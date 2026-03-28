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
	private float _minScale;      // New: minimum scaling factor
	private float _maxScale;      // New: maximum scaling factor

	public MixedPrecisionTrainer(
		float initScale = 65536.0f,
		float growthFactor = 2.0f,
		float backoffFactor = 0.5f,
		int growthInterval = 2000,
		Device device = null,
		ScalarType precision = ScalarType.Float32,
		float minScale = 1e-4f,      // default minimum scale
		float maxScale = 16777216f)  // default maximum scale (2^24)
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
		_minScale = minScale;
		_maxScale = maxScale;

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
				// Use any().item<bool>() to check
				if (grad.isinf().any().item<bool>() || grad.isnan().any().item<bool>())
				{
					_foundInf = true;
					break;
				}
			}
		}

		return _foundInf;
	}

	// Update Scaler with boundary protection
	public void Update()
	{
		if (!_isMixedPrecision)
			return;

		_growthCounter++;

		if (_foundInf)
		{
			// Found inf/nan, reduce scaling factor but not below minimum
			_lossScale = Math.Max(_lossScale * _backoffFactor, _minScale);
			_growthCounter = 0;
		}
		else if (_growthCounter >= _growthInterval)
		{
			// No issues for a while, increase scaling factor but not above maximum
			_lossScale = Math.Min(_lossScale * _growthFactor, _maxScale);
			_growthCounter = 0;
		}
	}

	// Skip current update
	public void SkipStep(Optimizer optimizer)
	{
		optimizer.zero_grad();
	}

	// Change tensor to mixed precision (if needed)
	public Tensor ToMixedPrecision(Tensor tensor)
	{
		if (!_isMixedPrecision)
			return tensor;

		// If already at target precision, return the original tensor (avoid copy)
		if (tensor.dtype == _precision)
			return tensor;

		return tensor.to(_precision, copy: true);
	}

	// Change tensor to float32 (if needed)
	public Tensor ToFloat32(Tensor tensor)
	{
		if (!_isMixedPrecision)
			return tensor;

		// If already float32, return the original tensor
		if (tensor.dtype == ScalarType.Float32)
			return tensor;

		return tensor.to(ScalarType.Float32, copy: true);
	}

	// Get Current Scale
	public float CurrentScale => _lossScale;

	public void Dispose()
	{
		// No additional cleanup needed
	}
}

public class AMPWrapper : IDisposable
{
	private MixedPrecisionTrainer _scaler;
	private torch.nn.Module<Tensor, Tensor[]> _model;
	private Optimizer _optimizer;
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
		_isMixedPrecision = _scaler.IsMixedPrecision;

		// If mixed precision is enabled, convert model parameters to target precision (in-place)
		if (_isMixedPrecision)
		{
			ConvertWeightsToMixedPrecision();
		}
	}

	// Convert model weights to mixed precision (in-place)
	private void ConvertWeightsToMixedPrecision()
	{
		var targetType = _scaler.Precision;
		foreach (var (_, param) in _model.named_parameters())
		{
			if (param.dtype != targetType)
			{
				var converted = param.to(targetType);
				param.set_(converted);
				converted.Dispose(); // Release temporary tensor
			}
		}
	}

	// Convert model weights back to float32 (in-place)
	private void RestoreWeightsToFloat32()
	{
		foreach (var (_, param) in _model.named_parameters())
		{
			if (param.dtype != ScalarType.Float32)
			{
				var converted = param.to(ScalarType.Float32);
				param.set_(converted);
				converted.Dispose();
			}
		}
	}

	// Forward pass (single input)
	public Tensor[] Forward(Tensor input)
	{
		if (!_isMixedPrecision)
		{
			return _model.forward(input).ToArray();
		}

		// Convert input to mixed precision, convert outputs back to float32
		using var mixedInput = _scaler.ToMixedPrecision(input);
		var outputs = _model.forward(mixedInput);
		return outputs.Select(_scaler.ToFloat32).ToArray();
	}

	// Forward pass (multiple inputs)
	public Tensor[] Forward(IEnumerable<Tensor> inputs)
	{
		if (!_isMixedPrecision)
		{
			var results = new List<Tensor>();
			foreach (var input in inputs)
			{
				results.AddRange(_model.forward(input));
			}
			return results.ToArray();
		}

		var outputs = new List<Tensor>();
		foreach (var input in inputs)
		{
			using var mixedInput = _scaler.ToMixedPrecision(input);
			var outTensors = _model.forward(mixedInput);
			outputs.AddRange(outTensors.Select(_scaler.ToFloat32));
		}
		return outputs.ToArray();
	}

	// Single training step
	public void Step(Tensor loss)
	{
		// Ensure loss is float32
		if (loss.dtype != ScalarType.Float32)
		{
			loss = loss.to(ScalarType.Float32);
		}

		try
		{
			if (_isMixedPrecision)
			{
				// Scale loss and backpropagate
				var scaledLoss = _scaler.ScaleLoss(loss);
				scaledLoss.backward();

				// Check if gradients contain inf/nan
				bool hasInfNan = _scaler.CheckGradientsForInfNan(_model.parameters());

				if (!hasInfNan)
				{
					// Unscale gradients
					_scaler.UnscaleGradients(_model.parameters());
					// Update optimizer
					_optimizer.step();
					// Zero gradients (fix gradient accumulation issue)
					_optimizer.zero_grad();
				}
				else
				{
					// Skip this update and zero gradients
					_scaler.SkipStep(_optimizer);
				}

				// Update loss scaler
				_scaler.Update();

				// Dispose scaled loss tensor if it is different from original loss
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

	// Training step (includes forward, loss computation, backward)
	public Tensor[] TrainStep(Tensor input, Tensor target, Func<Tensor[], Tensor, Tensor> lossFunction)
	{
		var outputs = Forward(input);
		var loss = lossFunction(outputs, target);
		Step(loss);
		return outputs;
	}

	// Evaluation mode: forward pass with mixed precision weights (input automatically converted, outputs converted to float32)
	public Tensor[] Evaluate(Tensor input)
	{
		if (!_isMixedPrecision)
		{
			return _model.forward(input).ToArray();
		}

		// Convert input to mixed precision, convert outputs back to float32
		using var mixedInput = _scaler.ToMixedPrecision(input);
		var outputs = _model.forward(mixedInput);
		return outputs.Select(_scaler.ToFloat32).ToArray();
	}

	// Force a training step using float32 (kept for backward compatibility)
	public void TrainStepFloat32(Tensor input, Tensor target, Func<Tensor[], Tensor, Tensor> lossFunction)
	{
		// Temporarily convert model weights to float32
		if (_isMixedPrecision)
		{
			RestoreWeightsToFloat32();
		}

		var outputs = _model.forward(input).ToArray();
		var loss = lossFunction(outputs, target);

		loss.backward();
		_optimizer.step();
		_optimizer.zero_grad();

		// If originally mixed precision, revert to mixed precision
		if (_isMixedPrecision)
		{
			ConvertWeightsToMixedPrecision();
		}
	}

	public MixedPrecisionTrainer Scaler => _scaler;
	public ScalarType Precision => _scaler.Precision;
	public bool IsMixedPrecision => _isMixedPrecision;

	public void Dispose()
	{
		// Convert model weights back to float32 for subsequent use
		if (_isMixedPrecision)
		{
			RestoreWeightsToFloat32();
		}
		_scaler?.Dispose();
	}
}