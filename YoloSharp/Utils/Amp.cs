using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;

public class MixedPrecisionTrainer : IDisposable
{
	private float _lossScale;
	private float _growthFactor;
	private float _backoffFactor;
	private int _growthInterval;
	private int _growthCounter;
	private bool _foundInf;
	private Device _device;
	private ScalarType _precision;

	public MixedPrecisionTrainer(
		float initScale = 65536.0f,
		float growthFactor = 2.0f,
		float backoffFactor = 0.5f,
		int growthInterval = 2000,
		Device device = null,
		ScalarType precision = ScalarType.Float16)
	{
		if (precision != ScalarType.Float16 && precision != ScalarType.BFloat16)
		{
			throw new ArgumentException("Precision must be either Float16 or BFloat16", nameof(precision));
		}

		_lossScale = initScale;
		_growthFactor = growthFactor;
		_backoffFactor = backoffFactor;
		_growthInterval = growthInterval;
		_growthCounter = 0;
		_foundInf = false;
		_device = device ?? (torch.cuda_is_available() ? CUDA : CPU);
		_precision = precision;


	}

	public ScalarType Precision => _precision;

	// Scale Loss
	public Tensor ScaleLoss(Tensor loss)
	{
		return loss * _lossScale;
	}

	// Scale Gradients
	public void ScaleGradients(IEnumerable<TorchSharp.Modules.Parameter> parameters)
	{
		if (!_foundInf)
		{
			foreach (var param in parameters)
			{
				if (param.grad is not null)
				{
					param.grad.mul_(_lossScale);
				}
			}
		}
	}

	// Unscale Gradients
	public void UnscaleGradients(IEnumerable<TorchSharp.Modules.Parameter> parameters)
	{
		var invScale = 1.0f / _lossScale;
		foreach (var param in parameters)
		{
			if (param.grad is not null)
			{
				param.grad.mul_(invScale);
			}
		}
	}

	// Check Gradients if is Inf or Nan
	public bool CheckGradientsForInfNan(IEnumerable<TorchSharp.Modules.Parameter> parameters)
	{
		_foundInf = false;

		foreach (var param in parameters)
		{
			if (param.grad is not null)
			{
				var grad = param.grad;

				// Check if is NaN or Inf
				if (grad.isinf().any().item<bool>() || grad.isnan().any().item<bool>())
				{
					_foundInf = true;
					break;
				}
			}
		}

		return _foundInf;
	}

	// Update Scaler
	public void Update()
	{
		_growthCounter++;

		if (_foundInf)
		{
			// If is Inf or NaN, reduce the scale.
			_lossScale = _lossScale * _backoffFactor;
			_growthCounter = 0;
			Console.WriteLine($"Reducing loss scale to: {_lossScale}");
		}
		else if (_growthCounter >= _growthInterval)
		{
			// If there is a long time with no Inf or NaN, upscale the scale.
			_lossScale = _lossScale * _growthFactor;
			_growthCounter = 0;
			Console.WriteLine($"Increasing loss scale to: {_lossScale}");
		}
	}

	// Skip current update
	public void SkipStep(Optimizer optimizer)
	{
		optimizer.zero_grad();
	}

	// Change tensor to mixed
	public Tensor ToMixedPrecision(Tensor tensor)
	{
		if (tensor.dtype == ScalarType.Float32)
		{
			return tensor.to(_precision, copy: true);
		}
		return tensor;
	}

	// Change the tensor to Float
	public Tensor ToFloat32(Tensor tensor)
	{
		if (tensor.dtype == _precision)
		{
			return tensor.to(ScalarType.Float32, copy: true);
		}
		return tensor;
	}

	// Get Current Scale
	public float CurrentScale => _lossScale;

	public void Dispose()
	{
		
	}
}

public class AMPWrapper : IDisposable
{
	private MixedPrecisionTrainer _scaler;
	private torch.nn.Module<Tensor, Tensor[]> _model;
	private Optimizer _optimizer;
	private Dictionary<string, Tensor> _originalWeights;
	private bool _weightsConverted;

	public AMPWrapper(
		torch.nn.Module<Tensor, Tensor[]> model,
		Optimizer optimizer,
		ScalarType precision = ScalarType.Float16,
		Device device = null)
	{
		_model = model;
		_optimizer = optimizer;
		_scaler = new MixedPrecisionTrainer(precision: precision, device: device);
		_originalWeights = new Dictionary<string, Tensor>();
		_weightsConverted = false;
	}

	// Convert the weight to Mixed
	private void ConvertWeightsToMixedPrecision()
	{
		if (_weightsConverted) return;

		foreach (var (name, param) in _model.named_parameters())
		{
			if (param.dtype == ScalarType.Float32)
			{
				// Get the org weight
				_originalWeights[name] = param.detach().clone();

				// Convert to mixed weight
				var mixedPrecisionParam = param.to(_scaler.Precision);
				param.set_(mixedPrecisionParam);
			}
		}

		_weightsConverted = true;
	}

	// Restore Weights To Float
	private void RestoreWeightsToFloat32()
	{
		if (!_weightsConverted) return;

		foreach (var (name, original) in _originalWeights)
		{
			var param = _model.named_parameters().FirstOrDefault(a => a.name == name).parameter;
			if (param.IsInvalid)
			{
				param.set_(original);
			}
		}

		// Clean the cache
		foreach (var weight in _originalWeights.Values)
		{
			weight.Dispose();
		}
		_originalWeights.Clear();
		_weightsConverted = false;
	}

	public Tensor[] Forward(Tensor input)
	{
		// Convert the weight to float
		ConvertWeightsToMixedPrecision();

		// Convert the weight to mixed
		Tensor inputMixed = _scaler.ToMixedPrecision(input);

		try
		{
			// Foward with mixed weight.
			var output = _model.forward(inputMixed);

			// Convert the weight to float
			return output.Select(_scaler.ToFloat32).ToArray();
		}
		finally
		{
			// Clean the cache
			if (inputMixed.Handle == input.Handle)
			{
				inputMixed.Dispose();
			}
		}
	}

	// Forward
	public Tensor[] Forward(IEnumerable<Tensor> inputs)
	{
		ConvertWeightsToMixedPrecision();

		var mixedInputs = inputs.Select(_scaler.ToMixedPrecision).ToList();
		var outputs = new List<Tensor>();

		try
		{
			foreach (var input in mixedInputs)
			{
				var output = _model.forward(input);
				outputs.AddRange(output.Select(_scaler.ToFloat32));
			}
		}
		finally
		{
			// Clean the cache.
			foreach (var input in mixedInputs)
			{
				if (input.IsInvalid)
				{
					input.Dispose();
				}
			}
		}

		return outputs.ToArray();
	}

	public void Step(Tensor loss)
	{
		if (loss.dtype != ScalarType.Float32)
		{
			loss = loss.to(ScalarType.Float32);
		}

		try
		{
			// scale Loss
			var scaledLoss = _scaler.ScaleLoss(loss);

			// Get the backward
			scaledLoss.backward();

			// Check Gradients if is Inf or Nan
			bool hasInfNan = _scaler.CheckGradientsForInfNan(_model.parameters());

			if (!hasInfNan)
			{
				// Cancel the scale to grad.
				_scaler.UnscaleGradients(_model.parameters());

				// Opt
				_optimizer.step();
			}
			else
			{
				// Skip if is NaN or Inf
				_scaler.SkipStep(_optimizer);
			}

			// Update the scaler.
			_scaler.Update();

			// 清除缩放损失
			if (scaledLoss.Handle == loss.Handle)
			{
				scaledLoss.Dispose();
			}
		}
		catch (Exception ex)
		{
			Console.WriteLine($"Error in training step: {ex.Message}");
			_scaler.SkipStep(_optimizer);
			throw;
		}
	}

	// Train
	public Tensor[] TrainStep(Tensor input, Tensor target, Func<Tensor[], Tensor, Tensor> lossFunction)
	{
		var outputs = Forward(input);

		var loss = lossFunction(outputs, target);
		Step(loss);

		return outputs;
	}

	// Eval
	public Tensor[] Evaluate(Tensor input)
	{
		RestoreWeightsToFloat32();

		try
		{
			return _model.forward(input).ToArray();
		}
		finally
		{
			ConvertWeightsToMixedPrecision();
		}
	}

	public MixedPrecisionTrainer Scaler => _scaler;

	public ScalarType Precision => _scaler.Precision;

	public void Dispose()
	{
		RestoreWeightsToFloat32();
		_scaler?.Dispose();
	}
}