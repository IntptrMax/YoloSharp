using System;
using System.Collections.Generic;
using TorchSharp;

namespace YoloSharp.Utils
{
    /// <summary>
    /// 指数移动平均（EMA）权重管理器。
    /// 
    /// 作用：跟踪模型训练过程中参数的平滑版本，用于验证和保存 best 权重。
    /// 因为 SGD 的权重震荡很大，直接用当前权重做 Val 会导致 mAP 剧烈波动；
    /// EMA 相当于对权重做指数平滑，让验证和保存结果更稳定。
    /// 
    /// 与 ultralytics 的 ModelEMA 语义一致：
    ///   decay = min(decay, (1 + updates) / (10 + updates))     # 前期快速跟踪
    ///   实际实现用 exp 形式：decay_dynamic = decay * (1 - exp(-updates / tau))
    /// </summary>
    internal class ModelEMA
    {
        private readonly Dictionary<string, torch.Tensor> _emaWeights;
        private readonly float _decay;
        private readonly float _tau;
        private long _updates;

        /// <param name="model">需要跟踪的模型</param>
        /// <param name="decay">EMA 衰减系数（默认 0.9999）</param>
        /// <param name="tau">前期动态 decay 的 warm-up 参数（默认 2000）</param>
        internal ModelEMA(torch.nn.Module model, float decay = 0.9999f, float tau = 2000f)
        {
            _decay = decay;
            _tau = tau;
            _updates = 0;
            _emaWeights = new Dictionary<string, torch.Tensor>();

            using (torch.no_grad())
            {
                foreach (var kv in model.state_dict())
                {
                    var t = kv.Value;
                    // 只跟踪浮点类型的张量（参数 + BN 的 running_mean/var）
                    if (t.dtype == torch.ScalarType.Float32 ||
                        t.dtype == torch.ScalarType.Float16 ||
                        t.dtype == torch.ScalarType.BFloat16)
                    {
                        _emaWeights[kv.Key] = t.detach().to(torch.ScalarType.Float32).clone();
                    }
                }
            }
        }

        /// <summary>每训练一步调用一次，用当前模型更新 EMA。</summary>
        internal void Update(torch.nn.Module model)
        {
            _updates++;

            // 动态 decay：前期快速跟踪，后期趋于稳定
            float d = _decay * (1.0f - (float)Math.Exp(-(double)_updates / _tau));

            using (torch.no_grad())
            {
                foreach (var kv in model.state_dict())
                {
                    if (!_emaWeights.TryGetValue(kv.Key, out var ema)) continue;

                    using var curr = kv.Value.detach().to(torch.ScalarType.Float32);
                    // ema = ema * d + curr * (1 - d)
                    ema.mul_(d).add_(curr, alpha: 1.0f - d);
                }
            }
        }

        /// <summary>把 EMA 权重写回模型的参数（用于验证）。</summary>
        internal void ApplyTo(torch.nn.Module model)
        {
            using (torch.no_grad())
            {
                var target = model.state_dict();
                foreach (var kv in _emaWeights)
                {
                    if (target.TryGetValue(kv.Key, out var param))
                    {
                        param.copy_(kv.Value.to(param.dtype, param.device));
                    }
                }
            }
        }

        /// <summary>返回 EMA 权重的 state_dict 副本（用于保存 best.bin）。</summary>
        internal Dictionary<string, torch.Tensor> GetStateDict(torch.nn.Module model)
        {
            var result = new Dictionary<string, torch.Tensor>();
            using (torch.no_grad())
            {
                foreach (var kv in model.state_dict())
                {
                    if (_emaWeights.TryGetValue(kv.Key, out var ema))
                    {
                        result[kv.Key] = ema.to(kv.Value.dtype, kv.Value.device).clone();
                    }
                    else
                    {
                        result[kv.Key] = kv.Value.clone();
                    }
                }
            }
            return result;
        }

        public long Updates => _updates;
    }
}