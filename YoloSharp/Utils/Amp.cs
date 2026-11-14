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
		// [Add 2026-11-05] Gradient accumulation (ultralytics nbs=64 semantics): python accumulates
		// `round(nbs/batch)` batches before stepping (effective batch 64). Without it, bs8 stepping
		// every batch = effective batch 8 → 8× noisier SGD gradients → jagged mAP curve
		// (observed 0.99→0.43 single-epoch swings). When accumulating, only the LAST batch of a
		// window runs clip+step+zero_grad; the loss is pre-scaled by 1/accumulate.
		// 梯度累积对齐python(nbs=64)：等效batch翻8倍，平滑mAP毛刺。窗口内仅末批step，loss按1/n预缩放。
		private int _accumulateFld;   // [Fix 2026-11-09] 不再readonly——warmup期SetAccumulate渐进调整
		private long _accumCounter;

		/// <summary>
		/// [Add 2026-11-09] Runtime accumulate-window adjustment for warmup (python trainer.py L535:
		/// `accumulate = max(1, interp(ni, xi, [1, nbs/bs]).round())` grows the effective batch
		/// progressively during warmup — early epochs step more often, matching ultralytics).
		/// warmup期累积窗口动态调整：1→n渐进（等效batch从小长大，早期更新频率高学得快）。
		/// </summary>
		public void SetAccumulate(int value)
		{
			int v = Math.Max(1, value);
			if (v == _accumulateFld) return;
			_accumulateFld = v;
			_accumCounter = 0;   // 新窗口重新累积（半窗口误差≤1批，可接受）
		}

		public torch.optim.Optimizer Optimizer => _optimizer;
		public MixedPrecisionTrainer Scaler => _scaler;
	public torch.ScalarType Precision => _scaler.Precision;
	public bool IsMixedPrecision => _isMixedPrecision;

	public AMPWrapper(torch.nn.Module<torch.Tensor, (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds)?> model, torch.optim.Optimizer optimizer, torch.ScalarType precision = torch.ScalarType.Float32, torch.Device device = null, float gradClipNorm = 10.0f, float initScale = 512.0f, int accumulate = 1)
	{
		_model = model;
		_optimizer = optimizer;
		_scaler = new MixedPrecisionTrainer(precision: precision, device: device, initScale: initScale);
		_gradClipNorm = gradClipNorm;
		_isMixedPrecision = _scaler.IsMixedPrecision;
		_accumulateFld = Math.Max(1, accumulate);   // [Add 2026-11-05] nbs/batch梯度累积窗口（11-09改可变字段供warmup渐进）
		_accumCounter = 0;
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

		// [Add 2026-11-05] gradient-accumulation window: python ultralytics accumulates
		// round(nbs/batch) batches per optimizer step. Scale this batch's loss by 1/accumulate
		// so the summed gradient averages (not sums) over the window; step only at window end.
		// 梯度累积：loss按1/n预缩放（窗口内求平均），仅窗口末批执行clip+step+zero。
		bool isWindowEnd = _accumulateFld <= 1 || (++_accumCounter >= _accumulateFld);
		if (_accumulateFld > 1)
			loss = loss.div(_accumulateFld);

		try
		{
			if (_isMixedPrecision)
			{
				var scaledLoss = _scaler.ScaleLoss(loss);
				scaledLoss.backward();

				bool hasInfNan = _scaler.CheckGradientsForInfNan(_model.parameters());
				if (!hasInfNan)
				{
					if (isWindowEnd)
					{
						_scaler.UnscaleGradients(_model.parameters());
						if (_gradClipNorm > 0)
							torch.nn.utils.clip_grad_norm_(_model.parameters(), _gradClipNorm);
						_optimizer.step();
						_optimizer.zero_grad();
						_accumCounter = 0;
					}
				}
				else
				{
					// Inf/NaN: discard the half-built window (python skips the step entirely)
					_optimizer.zero_grad();
					_accumCounter = 0;
				}

				_scaler.Update();

				if (!ReferenceEquals(scaledLoss, loss))
					scaledLoss.Dispose();
			}
			else
			{
				loss.backward();
				if (isWindowEnd)
				{
					if (_gradClipNorm > 0)
						torch.nn.utils.clip_grad_norm_(_model.parameters(), _gradClipNorm);
					_optimizer.step();
					_optimizer.zero_grad();
					_accumCounter = 0;
				}
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