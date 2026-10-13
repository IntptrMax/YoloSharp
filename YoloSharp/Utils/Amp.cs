using TorchSharp;


public class MixedPrecisionTrainer : IDisposable
{
	private float _lossScale;
	private readonly float _growthFactor;
	private readonly float _backoffFactor;
	private readonly int _growthInterval;
	private int _successCounter;
	private bool _foundInf;
	private readonly torch.ScalarType _precision;
	private readonly bool _isMixedPrecision;
	private readonly float _minScale;
	private readonly float _maxScale;

	public MixedPrecisionTrainer(float initScale = 512.0f, float growthFactor = 2.0f, float backoffFactor = 0.5f, int growthInterval = 2000, torch.Device device = null, torch.ScalarType precision = torch.ScalarType.Float32, float minScale = 1e-4f, float maxScale = 16777216f)
	{
		if (precision != torch.ScalarType.Float16 && precision != torch.ScalarType.BFloat16 && precision != torch.ScalarType.Float32)
		{
			throw new ArgumentException("Precision must be Float16, BFloat16, or Float32", nameof(precision));
		}

		_lossScale = initScale;
		_growthFactor = growthFactor;
		_backoffFactor = backoffFactor;
		_growthInterval = growthInterval;
		_successCounter = 0;
		_foundInf = false;
		_precision = precision;
		_isMixedPrecision = precision == torch.ScalarType.Float16 || precision == torch.ScalarType.BFloat16;
		_minScale = minScale;
		_maxScale = maxScale;

		if (!_isMixedPrecision)
		{
			_lossScale = 1.0f;
		}
	}

	public torch.ScalarType Precision => _precision;
	public bool IsMixedPrecision => _isMixedPrecision;
	public float CurrentScale => _lossScale;

	public torch.Tensor ScaleLoss(torch.Tensor loss)
	{
		if (!_isMixedPrecision) return loss;
		return loss * _lossScale;
	}

	public void UnscaleGradients(IEnumerable<TorchSharp.Modules.Parameter> parameters)
	{
		if (!_isMixedPrecision) return;
		float invScale = 1.0f / _lossScale;
		foreach (var param in parameters)
		{
			if (param.grad is null) continue;

			var fp32Grad = param.grad.to(torch.ScalarType.Float32);
			fp32Grad.mul_(invScale);

			param.grad.Dispose();
			param.grad = fp32Grad;
		}
	}


	public bool CheckGradientsForInfNan(IEnumerable<TorchSharp.Modules.Parameter> parameters)
	{
		_foundInf = false;
		foreach (var param in parameters)
		{
			if (param.grad is null) continue;

			using var g = param.grad.to(torch.ScalarType.Float32);
			if (g.isinf().any().item<bool>() || g.isnan().any().item<bool>())
			{
				_foundInf = true;
				break;
			}
		}
		return _foundInf;
	}

	public void Update()
	{
		if (!_isMixedPrecision) return;

		if (_foundInf)
		{
			_lossScale = Math.Max(_lossScale * _backoffFactor, _minScale);
			_successCounter = 0;
		}
		else
		{
			_successCounter++;
			if (_successCounter >= _growthInterval)
			{
				_lossScale = Math.Min(_lossScale * _growthFactor, _maxScale);
				_successCounter = 0;
			}
		}
	}

	public void Dispose() { }
}

public class AMPWrapper : IDisposable
{
	private readonly torch.nn.Module<torch.Tensor, (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds)?> _model;
	private readonly torch.optim.Optimizer _optimizer;
	private readonly MixedPrecisionTrainer _scaler;
	private readonly bool _isMixedPrecision;
	private readonly float _gradClipNorm;

	public torch.optim.Optimizer Optimizer => _optimizer;
	public MixedPrecisionTrainer Scaler => _scaler;
	public torch.ScalarType Precision => _scaler.Precision;
	public bool IsMixedPrecision => _isMixedPrecision;

	public AMPWrapper(torch.nn.Module<torch.Tensor, (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds)?> model, torch.optim.Optimizer optimizer, torch.ScalarType precision = torch.ScalarType.Float32, torch.Device device = null, float gradClipNorm = 10.0f, float initScale = 512.0f)
	{
		_model = model;
		_optimizer = optimizer;
		_scaler = new MixedPrecisionTrainer(precision: precision, device: device, initScale: initScale);
		_gradClipNorm = gradClipNorm;
		_isMixedPrecision = _scaler.IsMixedPrecision;
	}


	public (torch.Tensor loss, torch.Tensor lossItems) TrainStep(torch.Tensor input, Dictionary<string, torch.Tensor> batch, torch.nn.Module<Dictionary<string, object>, Dictionary<string, torch.Tensor>, (torch.Tensor loss, torch.Tensor loss_items)> lossFunc)
	{
		_model.train();

		torch.Tensor modelInput = input;
		if (_isMixedPrecision && input.dtype != _scaler.Precision)
		{
			modelInput = input.to(_scaler.Precision);
		}

		var outputs = _model.forward(modelInput);
		if (outputs is null)
		{
			throw new InvalidOperationException("Model output is null.");
		}

		var (_, preds) = outputs.Value;

		var predsFp32 = ConvertObjectDictToFP32(preds);
		var batchFp32 = EnsureBatchFp32(batch);

		var (loss, lossItems) = lossFunc.forward(predsFp32, batchFp32);

		Step(loss);

		return (loss, lossItems);
	}


	public void Step(torch.Tensor loss)
	{
		if (loss.numel() != 1) loss = loss.sum();
		if (loss.dtype != torch.ScalarType.Float32) loss = loss.to(torch.ScalarType.Float32);

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
					if (_gradClipNorm > 0)
						torch.nn.utils.clip_grad_norm_(_model.parameters(), _gradClipNorm);
					_optimizer.step();
					_optimizer.zero_grad();
				}
				else
				{
					//Console.WriteLine($"[AMP] Found Inf/NaN gradients, skipping step. Current scale={_scaler.CurrentScale}");
					_optimizer.zero_grad();
				}

				_scaler.Update();

				if (!ReferenceEquals(scaledLoss, loss))
					scaledLoss.Dispose();
			}
			else
			{
				loss.backward();
				if (_gradClipNorm > 0)
					torch.nn.utils.clip_grad_norm_(_model.parameters(), _gradClipNorm);
				_optimizer.step();
				_optimizer.zero_grad();
			}
		}
		catch (Exception ex)
		{
			Console.WriteLine($"Error in training step: {ex.Message}");
			_optimizer.zero_grad();
			throw;
		}
	}


	public (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds)? Evaluate(torch.Tensor input)
	{
		_model.eval();

		torch.Tensor modelInput = input;
		if (_isMixedPrecision && input.dtype != _scaler.Precision)
		{
			modelInput = input.to(_scaler.Precision);
		}

		var raw = _model.forward(modelInput);
		if (!raw.HasValue)
		{
			return null;
		}

		var (rawInference, rawPreds) = raw.Value;

		var inferenceFloat = new Dictionary<string, torch.Tensor>();
		foreach (var kv in rawInference)
		{
			inferenceFloat[kv.Key] = ToFp32Tensor(kv.Value);
		}

		var predsFloat = new Dictionary<string, object>();
		foreach (var kv in rawPreds)
		{
			predsFloat[kv.Key] = ConvertToFp32(kv.Value);
		}

		return (inferenceFloat, predsFloat);
	}


	private static torch.Tensor ToFp32Tensor(torch.Tensor t)
	{
		if (t is null) return null;
		if (t.dtype == torch.ScalarType.Float32) return t;
		if (t.dtype == torch.ScalarType.Float16 ||
			t.dtype == torch.ScalarType.BFloat16 ||
			t.dtype == torch.ScalarType.Float64)
		{
			return t.to(torch.ScalarType.Float32);
		}
		return t;
	}

	private static object ConvertToFp32(object obj)
	{
		switch (obj)
		{
			case null:
				return null;
			case torch.Tensor t:
				return ToFp32Tensor(t);
			case torch.Tensor[] arr:
				return arr.Select(ToFp32Tensor).ToArray();
			case Dictionary<string, torch.Tensor> dt:
				{
					var r = new Dictionary<string, torch.Tensor>();
					foreach (var kv in dt) r[kv.Key] = ToFp32Tensor(kv.Value);
					return r;
				}
			case Dictionary<string, object> dobj:
				{
					var r = new Dictionary<string, object>();
					foreach (var kv in dobj) r[kv.Key] = ConvertToFp32(kv.Value);
					return r;
				}
			default:
				return obj;
		}
	}

	private static Dictionary<string, torch.Tensor> ConvertDictTensorsToFP32(Dictionary<string, torch.Tensor> dict)
	{
		if (dict is null) return null;
		var result = new Dictionary<string, torch.Tensor>();
		foreach (var kv in dict)
			result[kv.Key] = ToFp32Tensor(kv.Value);
		return result;
	}

	private static Dictionary<string, object> ConvertObjectDictToFP32(Dictionary<string, object> dict)
	{
		if (dict is null) return null;
		var result = new Dictionary<string, object>();
		foreach (var kv in dict)
			result[kv.Key] = ConvertToFp32(kv.Value);
		return result;
	}


	private static Dictionary<string, torch.Tensor> EnsureBatchFp32(Dictionary<string, torch.Tensor> batch)
	{
		if (batch is null) return null;
		var result = new Dictionary<string, torch.Tensor>();
		foreach (var kv in batch)
		{
			var t = kv.Value;
			if (t.dtype == torch.ScalarType.Float16 ||
				t.dtype == torch.ScalarType.BFloat16 ||
				t.dtype == torch.ScalarType.Float64)
			{
				t = t.to(torch.ScalarType.Float32);
			}
			result[kv.Key] = t;
		}
		return result;
	}

	public void Dispose()
	{
		_scaler?.Dispose();
	}
}