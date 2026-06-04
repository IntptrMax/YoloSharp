using TorchSharp;

public class MixedPrecisionTrainer : IDisposable
{
    private float _lossScale;
    private float _growthFactor;
    private float _backoffFactor;
    private int _growthInterval;
    private int _growthCounter;
    private bool _foundInf;
    private torch.Device _device;
    private torch.ScalarType _precision;
    private bool _isMixedPrecision;
    private float _minScale;      // New: minimum scaling factor
    private float _maxScale;      // New: maximum scaling factor

    public MixedPrecisionTrainer(
        float initScale = 65536.0f,
        float growthFactor = 2.0f,
        float backoffFactor = 0.5f,
        int growthInterval = 2000,
        torch.Device device = null,
       torch.ScalarType precision = torch.ScalarType.Float32,
        float minScale = 1e-4f,      // default minimum scale
        float maxScale = 16777216f)  // default maximum scale (2^24)
    {
        if (precision != torch.ScalarType.Float16 && precision != torch.ScalarType.BFloat16 && precision != torch.ScalarType.Float32)
        {
            throw new ArgumentException("Precision must be Float16, BFloat16, or Float32", nameof(precision));
        }

        _lossScale = initScale;
        _growthFactor = growthFactor;
        _backoffFactor = backoffFactor;
        _growthInterval = growthInterval;
        _growthCounter = 0;
        _foundInf = false;
        _device = device ?? (torch.cuda_is_available() ? torch.CUDA : torch.CPU);
        _precision = precision;
        _isMixedPrecision = (precision == torch.ScalarType.Float16 || precision == torch.ScalarType.BFloat16);
        _minScale = minScale;
        _maxScale = maxScale;

        if (!_isMixedPrecision)
        {
            _lossScale = 1.0f;
        }
    }

    public torch.ScalarType Precision => _precision;
    public bool IsMixedPrecision => _isMixedPrecision;

    // Scale Loss
    public torch.Tensor ScaleLoss(torch.Tensor loss)
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
    public void SkipStep(torch.optim.Optimizer optimizer)
    {
        optimizer.zero_grad();
    }

    // Change tensor to mixed precision (if needed)
    public torch.Tensor ToMixedPrecision(torch.Tensor tensor)
    {
        if (!_isMixedPrecision)
            return tensor;

        // If already at target precision, return the original tensor (avoid copy)
        if (tensor.dtype == _precision)
            return tensor;

        return tensor.to(_precision, copy: true);
    }

    // Change tensor to float32 (if needed)
    public torch.Tensor ToFloat32(torch.Tensor tensor)
    {
        if (!_isMixedPrecision)
            return tensor;

        // If already float32, return the original tensor
        if (tensor.dtype == torch.ScalarType.Float32)
            return tensor;

        return tensor.to(torch.ScalarType.Float32, copy: true);
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
    private readonly MixedPrecisionTrainer _scaler;
    private readonly torch.nn.Module<torch.Tensor, (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds)?> _model;
    private readonly torch.optim.Optimizer _optimizer;
    private readonly bool _isMixedPrecision;

    private Dictionary<string, torch.Tensor> _fp32Weights;

    public torch.optim.Optimizer Optimizer => _optimizer;
    public MixedPrecisionTrainer Scaler => _scaler;
    public torch.ScalarType Precision => _scaler.Precision;
    public bool IsMixedPrecision => _isMixedPrecision;

    public AMPWrapper(
        torch.nn.Module<torch.Tensor, (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds)?> model,
        torch.optim.Optimizer optimizer,
       torch.ScalarType precision = torch.ScalarType.Float32,
       torch.Device device = null)
    {
        _model = model;
        _optimizer = optimizer;
        _scaler = new MixedPrecisionTrainer(precision: precision, device: device);
        _isMixedPrecision = _scaler.IsMixedPrecision;

        if (_isMixedPrecision)
        {
            using (torch.no_grad())
            {
                _fp32Weights = new Dictionary<string, torch.Tensor>();
                foreach (var (name, param) in _model.named_parameters())
                {
                    var masterParam = param.detach().clone().to(torch.ScalarType.Float32).requires_grad_(false);
                    _fp32Weights[name] = masterParam;
                    using var halfParam = param.to(precision);
                    param.set_(halfParam);
                }
            }
        }
    }

    public IEnumerable<torch.Tensor> MasterParameters => _fp32Weights?.Values;

    private void SyncMasterToModel()
    {
        if (!_isMixedPrecision || _fp32Weights == null) return;

        using (torch.no_grad())
        {
            foreach (var (name, param) in _model.named_parameters())
            {
                using var halfParam = _fp32Weights[name].to(_scaler.Precision);
                param.set_(halfParam);
            }
        }
    }

    private void SyncModelGradsToMaster()
    {
        if (!_isMixedPrecision || _fp32Weights == null) return;

        foreach (var (name, param) in _model.named_parameters())
        {
            if (param.grad is null) continue;

            var masterParam = _fp32Weights[name];
            using var fp32Grad = param.grad.to(torch.ScalarType.Float32);
            if (masterParam.grad is null)
            {
                masterParam.grad = fp32Grad;
            }
            else
            {
                masterParam.grad.add_(fp32Grad);
                masterParam.grad.set_(fp32Grad);
            }

            param.grad.Dispose();
            param.grad = null;
        }
    }

    public (torch.Tensor loss, torch.Tensor lossItems) TrainStep(
        torch.Tensor input,
        Dictionary<string, torch.Tensor> batch,
        torch.nn.Module<Dictionary<string, object>, Dictionary<string, torch.Tensor>, (torch.Tensor loss, torch.Tensor loss_items)> lossFunc)
    {
        _model.train();
        SyncMasterToModel();

        (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds) outputs;
        if (_isMixedPrecision)
        {
            torch.Tensor mixedInput = input.to(_scaler.Precision);
            var rawOutputs = _model.forward(mixedInput);
            outputs = ConvertOutputsForLoss(rawOutputs);
        }
        else
        {
            outputs = _model.forward(input).Value;
        }

        Dictionary<string, object> preds = outputs.preds;
        var (loss, lossItems) = lossFunc.forward(preds, batch);

        Step(loss);

        return (loss, lossItems);
    }

    private (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds) ConvertOutputsForLoss((Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds)? rawOutputs)
    {
        if (!rawOutputs.HasValue)
            throw new InvalidOperationException("Model output is null.");

        var (rawY, rawPreds) = rawOutputs.Value;
        return (ConvertDictTensorsToFP32(rawY),
                ConvertDictTensorsToFP32(rawPreds));
    }

    private object ConvertObjectTensorsToFP32(object obj)
    {
        if (obj is torch.Tensor t)
            return t.to(torch.ScalarType.Float32);
        if (obj is Dictionary<string, object> dict)
            return ConvertDictTensorsToFP32(dict);
        return obj;
    }

    private torch.Tensor ConvertObjectTensorsToFP32(torch.Tensor obj)
    {
        if (obj is torch.Tensor t)
            return t.to(torch.ScalarType.Float32);
        return obj;
    }

    private Dictionary<string, object> ConvertDictTensorsToFP32(Dictionary<string, object> dict)
    {
        var newDict = new Dictionary<string, object>();
        foreach (var kvp in dict)
        {
            newDict[kvp.Key] = ConvertObjectTensorsToFP32(kvp.Value);
        }
        return newDict;
    }

    private Dictionary<string, torch.Tensor> ConvertDictTensorsToFP32(Dictionary<string, torch.Tensor> dict)
    {
        if (dict is null)
        {
            return dict;
        }
        var newDict = new Dictionary<string, torch.Tensor>();
        foreach (var kvp in dict)
        {
            newDict[kvp.Key] = ConvertObjectTensorsToFP32(kvp.Value);
        }
        return newDict;
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
                    SyncModelGradsToMaster();

                    _optimizer.step();
                    _optimizer.zero_grad();
                }
                else
                {
                    _scaler.SkipStep(_optimizer);
                }

                _scaler.Update();

                if (!ReferenceEquals(scaledLoss, loss))
                    scaledLoss.Dispose();
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
                _scaler.SkipStep(_optimizer);
            else
                _optimizer.zero_grad();
            throw;
        }
    }


    public (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds)? Evaluate(torch.Tensor input)
    {
        _model.eval();
        if (!_isMixedPrecision)
            return _model.forward(input);

        SyncMasterToModel();
        using var mixedInput = input.to(_scaler.Precision);
        var raw = _model.forward(mixedInput);
        if (!raw.HasValue) return null;
        var (rawInference, rawPreds) = raw.Value;

        var inferenceFloat = new Dictionary<string, torch.Tensor>();
        foreach (var y in rawInference)
        {
            if (y.Value is torch.Tensor ty)
            {
                inferenceFloat[y.Key] = ty.to(torch.ScalarType.Float32);
                ty.Dispose();
            }
            else
            {
                inferenceFloat[y.Key] = y.Value;
            }
        }
        rawInference.Clear();

        var predsFloat = new Dictionary<string, object>();
        foreach (var kvp in rawPreds)
        {
            if (kvp.Value is torch.Tensor pt)
            {
                predsFloat[kvp.Key] = pt.to(torch.ScalarType.Float32);
                pt.Dispose();
            }
            else
            {
                predsFloat[kvp.Key] = kvp.Value;
            }
        }
        rawPreds.Clear();

        return (inferenceFloat, predsFloat);
    }

    public void Dispose()
    {
        if (_isMixedPrecision && _fp32Weights != null)
        {
            SyncMasterToModel();
            foreach (var (_, param) in _model.named_parameters())
            {
                using var fp32Param = param.to(torch.ScalarType.Float32);
                param.set_(fp32Param);
            }

            foreach (var kvp in _fp32Weights) kvp.Value.Dispose();
            _fp32Weights = null;
        }
        _scaler?.Dispose();
    }
}