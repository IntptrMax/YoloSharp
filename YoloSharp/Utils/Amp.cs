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
	private bool _isMixedPrecision;

	public MixedPrecisionTrainer(
		float initScale = 65536.0f,
		float growthFactor = 2.0f,
		float backoffFactor = 0.5f,
		int growthInterval = 2000,
		Device device = null,
		ScalarType precision = ScalarType.Float32)  
	{
		if (precision != ScalarType.Float16 && precision != ScalarType.BFloat16 && precision != ScalarType.Float32)
		{
			throw new ArgumentException("Precision must be Float16, BFloat16, or Float32", nameof(precision));
		}

		_lossScale = initScale;
		_growthFactor = growthFactor;
		_backoffFactor = backoffFactor;
		_growthInterval = growthInterval;
		_growthCounter = 0;
		_foundInf = false;
		_device = device ?? (torch.cuda_is_available() ? CUDA : CPU);
		_precision = precision;
		_isMixedPrecision = (precision == ScalarType.Float16 || precision == ScalarType.BFloat16);

		if (!_isMixedPrecision)
		{
			_lossScale = 1.0f;
		}
	}

	public ScalarType Precision => _precision;
	public bool IsMixedPrecision => _isMixedPrecision;

	// Scale Loss
	public Tensor ScaleLoss(Tensor loss)
	{
		if (!_isMixedPrecision)
			return loss;

		return loss * _lossScale;
	}

	// Scale Gradients
	public void ScaleGradients(IEnumerable<TorchSharp.Modules.Parameter> parameters)
	{
		if (!_isMixedPrecision || _foundInf)
			return;

		foreach (var param in parameters)
		{
			if (param.grad is not null)
			{
				param.grad.mul_(_lossScale);
			}
		}
	}

	// Unscale Gradients
	public void UnscaleGradients(IEnumerable<TorchSharp.Modules.Parameter> parameters)
	{
		if (!_isMixedPrecision)
			return;

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
		if (!_isMixedPrecision)
			return;

		_growthCounter++;

		if (_foundInf)
		{
			// If is Inf or NaN, reduce the scale.
			_lossScale = _lossScale * _backoffFactor;
			_growthCounter = 0;
			//Console.Write($"Reducing loss scale to: {_lossScale}");
		}
		else if (_growthCounter >= _growthInterval)
		{
			// If there is a long time with no Inf or NaN, upscale the scale.
			_lossScale = _lossScale * _growthFactor;
			_growthCounter = 0;
			//Console.Write($"Increasing loss scale to: {_lossScale}");
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
		if (!_isMixedPrecision)
			return tensor;

		if (tensor.dtype == ScalarType.Float32)
		{
			return tensor.to(_precision, copy: true);
		}
		return tensor;
	}

	// Change the tensor to Float
	public Tensor ToFloat32(Tensor tensor)
	{
		if (!_isMixedPrecision)
			return tensor;

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
	private bool _isMixedPrecision;

	public AMPWrapper(
		torch.nn.Module<Tensor, Tensor[]> model,
		Optimizer optimizer,
		ScalarType precision = ScalarType.Float32,  
		Device device = null)
	{
		_model = model;
		_optimizer = optimizer;
		_scaler = new MixedPrecisionTrainer(precision: precision, device: device);
		_originalWeights = new Dictionary<string, Tensor>();
		_weightsConverted = false;
		_isMixedPrecision = _scaler.IsMixedPrecision;
	}

	// Convert the weight to Mixed
	private void ConvertWeightsToMixedPrecision()
	{
		if (!_isMixedPrecision || _weightsConverted)
			return;

		foreach (var (name, param) in _model.named_parameters())
		{
			if (param.dtype == ScalarType.Float32)
			{
				_originalWeights[name] = param.detach().clone();

				var mixedPrecisionParam = param.to(_scaler.Precision);
				param.set_(mixedPrecisionParam);
			}
		}

		_weightsConverted = true;
	}

	// Restore Weights To Float
	private void RestoreWeightsToFloat32()
	{
		if (!_isMixedPrecision || !_weightsConverted)
			return;

		foreach (var (name, original) in _originalWeights)
		{
			var namedParam = _model.named_parameters().FirstOrDefault(a => a.name == name);
			if (namedParam.parameter is not null && !namedParam.parameter.IsInvalid)
			{
				var param = namedParam.parameter;
				param.set_(original);
			}
		}

		foreach (var weight in _originalWeights.Values)
		{
			weight.Dispose();
		}
		_originalWeights.Clear();
		_weightsConverted = false;
	}

	public Tensor[] Forward(Tensor input)
	{
		if (_isMixedPrecision)
		{
			ConvertWeightsToMixedPrecision();
			Tensor inputMixed = _scaler.ToMixedPrecision(input);

			try
			{
				var output = _model.forward(inputMixed);
				return output.Select(_scaler.ToFloat32).ToArray();
			}
			finally
			{
				if (inputMixed.Handle != input.Handle)
				{
					inputMixed.Dispose();
				}
			}
		}
		else
		{
			return _model.forward(input).ToArray();
		}
	}

	// Forward
	public Tensor[] Forward(IEnumerable<Tensor> inputs)
	{
		if (!_isMixedPrecision)
		{
			var opts = new List<Tensor>();
			foreach (var input in inputs)
			{
				opts.AddRange(_model.forward(input));
			}
			return opts.ToArray();
		}

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
			foreach (var input in mixedInputs)
			{
				if (input.Handle != inputs.ElementAt(mixedInputs.IndexOf(input)).Handle)
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
			if (_isMixedPrecision)
			{
				var scaledLoss = _scaler.ScaleLoss(loss);
				scaledLoss.backward();

				bool hasInfNan = _scaler.CheckGradientsForInfNan(_model.parameters());

				if (!hasInfNan)
				{
					_scaler.UnscaleGradients(_model.parameters());
					_optimizer.step();
				}
				else
				{
					_scaler.SkipStep(_optimizer);
				}

				_scaler.Update();

				if (scaledLoss.Handle != loss.Handle)
				{
					scaledLoss.Dispose();
				}
			}
			else
			{
				loss.backward();
				_optimizer.step();
				_optimizer.zero_grad();
			}
		}
		catch (Exception ex)
		{
			Console.WriteLine($"Error in training step: {ex.Message}");
			if (_isMixedPrecision)
			{
				_scaler.SkipStep(_optimizer);
			}
			else
			{
				_optimizer.zero_grad();
			}
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
		if (_isMixedPrecision)
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
		else
		{
			return _model.forward(input).ToArray();
		}
	}

	public void TrainStepFloat32(Tensor input, Tensor target, Func<Tensor[], Tensor, Tensor> lossFunction)
	{
		if (_isMixedPrecision)
		{
			RestoreWeightsToFloat32();
		}

		var outputs = _model.forward(input).ToArray();
		var loss = lossFunction(outputs, target);

		loss.backward();
		_optimizer.step();
		_optimizer.zero_grad();
	}

	public MixedPrecisionTrainer Scaler => _scaler;
	public ScalarType Precision => _scaler.Precision;
	public bool IsMixedPrecision => _isMixedPrecision;

	public void Dispose()
	{
		RestoreWeightsToFloat32();
		_scaler?.Dispose();
	}
}