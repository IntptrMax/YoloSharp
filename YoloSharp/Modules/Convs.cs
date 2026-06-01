using System.Numerics;
using TorchSharp;
using TorchSharp.Modules;

namespace YoloSharp.Modules
{
    internal class Convs
    {
        /// <summary>
        /// Pad to 'same' shape outputs (int version).
        /// </summary>
        private static int autopad(int k, int? p = null, int d = 1)
        {
            if (d > 1)
                k = d * (k - 1) + 1;
            if (p is null)
                p = k / 2;
            return p.Value;
        }

        /// <summary>
        /// Pad to 'same' shape outputs (int[] version).
        /// </summary>
        private static int[] autopad(int[] k, int[]? p = null, int d = 1)
        {
            if (d > 1)
                k = k.Select(x => d * (x - 1) + 1).ToArray();
            if (p is null)
                p = k.Select(x => x / 2).ToArray();
            return p;
        }

        /// <summary>
        /// Standard convolution module with batch normalization and activation.
        /// </summary>
        internal class Conv : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            public readonly Conv2d conv;
            public readonly BatchNorm2d bn;
            protected readonly torch.nn.Module<torch.Tensor, torch.Tensor> act;
            private readonly double eps = 0.001;
            private readonly double momentum = 0.03;

            internal Conv(int c1, int c2, int k = 1, int s = 1, int? p = null, int g = 1, int d = 1, bool bias = false, torch.nn.Module<torch.Tensor, torch.Tensor>? act = null, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(Conv))
            {
                p = p ?? k / 2;
                this.conv = torch.nn.Conv2d(c1, c2, k, s, p.Value, groups: g, bias: bias, dilation: d, device: device, dtype: dtype);
                this.bn = torch.nn.BatchNorm2d(c2, eps: eps, momentum: momentum, track_running_stats: true, device: device, dtype: dtype);
                this.act = act ?? torch.nn.SiLU();
                RegisterComponents();
            }

            public override torch.Tensor forward(torch.Tensor x)
            {
                return this.act.forward(this.bn.forward(this.conv.forward(x)));
            }

            public virtual torch.Tensor forward_fuse(torch.Tensor x)
            {
                return this.act.forward(this.conv.forward(x));
            }
        }

        /// <summary>
        /// Simplified RepConv module with Conv fusing.
        /// </summary>
        internal class Conv2 : Conv
        {
            private readonly Conv2d cv2;
            private bool fuse = false;

            internal Conv2(int c1, int c2, int k = 3, int s = 1, int? p = null, int g = 1, int d = 1, bool bias = false, torch.nn.Module<torch.Tensor, torch.Tensor>? act = null, torch.Device? device = null, torch.ScalarType? dtype = null) : base(c1, c2, k, s, p, g, d, bias, act, device, dtype)
            {
                this.cv2 = torch.nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups: g, dilation: d, bias: false, device: device, dtype: dtype);
            }

            public override torch.Tensor forward(torch.Tensor x)
            {
                if (fuse)
                {
                    return this.act.forward(this.bn.forward(this.conv.forward(x) + this.cv2.forward(x)));
                }
                else
                {
                    fuse_convs();
                    return forward_fuse(x);
                }
            }

            public override torch.Tensor forward_fuse(torch.Tensor x)
            {
                return this.act.forward(this.bn.forward(this.conv.forward(x)));
            }

            public void fuse_convs()
            {
                var w = torch.zeros_like(this.conv.weight);
                int i0 = (int)w.shape[2] / 2;
                int i1 = (int)w.shape[3] / 2;
                w[torch.TensorIndex.Ellipsis, torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(i0, (i0 + 1)), torch.TensorIndex.Slice(i1, (i1 + 1))] = cv2.weight.clone();
                this.conv.weight.copy_(this.conv.weight + w);
            }
        }

        /// <summary>
        /// Depth-wise convolution module.
        /// </summary>
        internal class DWConv : Conv
        {
            internal DWConv(int in_channels, int out_channels, int kernel_size = 1, int stride = 1, int d = 1, torch.nn.Module<torch.Tensor, torch.Tensor>? act = null, torch.Device? device = null, torch.ScalarType? dtype = null) : base(in_channels, out_channels, kernel_size, stride, g: (int)BigInteger.GreatestCommonDivisor(in_channels, out_channels), d: d, act: act, device: device, dtype: dtype)
            {

            }
        }

        /// <summary>
        /// Light convolution module with 1x1 and depthwise convolutions.
        /// </summary>
        internal class LightConv : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly Conv conv1;
            private readonly DWConv conv2;

            internal LightConv(int c1, int c2, int k = 1, torch.nn.Module<torch.Tensor, torch.Tensor>? act = null, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(LightConv))
            {
                this.conv1 = new Conv(c1, c2, 1, act: torch.nn.Identity(), device: device, dtype: dtype);
                this.conv2 = new DWConv(c2, c2, k, act: act ?? torch.nn.ReLU(), device: device, dtype: dtype);
            }

            public override torch.Tensor forward(torch.Tensor x)
            {
                return this.conv2.forward(conv1.forward(x));
            }
        }

        /// <summary>
        /// Depth-wise transpose convolution module.
        /// </summary>
        internal class DWConvTranspose2d : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly ConvTranspose2d conv;

            internal DWConvTranspose2d(int c1, int c2, int k = 1, int s = 1, int p1 = 0, int p2 = 0, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(DWConvTranspose2d))
            {
                conv = torch.nn.ConvTranspose2d(c1, c2, k, s, p1, p2, groups: (int)BigInteger.GreatestCommonDivisor(c1, c2), device: device, dtype: dtype);
            }

            public override torch.Tensor forward(torch.Tensor x)
            {
                return conv.forward(x);
            }
        }

        /// <summary>
        /// Convolution transpose module with optional batch normalization and activation.
        /// </summary>
        internal class ConvTranspose : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly ConvTranspose2d conv_transpose;
            private readonly torch.nn.Module<torch.Tensor, torch.Tensor> bn;
            private readonly torch.nn.Module<torch.Tensor, torch.Tensor> act;
            private readonly double eps = 0.001;
            private readonly double momentum = 0.03;

            internal ConvTranspose(int c1, int c2, int k = 2, int s = 2, int p = 0, bool bn = true, torch.nn.Module<torch.Tensor, torch.Tensor>? act = null, torch.Device? device = null, torch.ScalarType? dtype = null)
                : base(nameof(ConvTranspose))
            {
                this.conv_transpose = torch.nn.ConvTranspose2d(c1, c2, k, s, p, bias: !bn, device: device, dtype: dtype);
                this.bn = bn ? torch.nn.BatchNorm2d(c2, eps: eps, momentum: momentum, track_running_stats: true, device: device, dtype: dtype) : torch.nn.Identity();
                this.act = act ?? torch.nn.SiLU();
            }

            public override torch.Tensor forward(torch.Tensor x)
            {
                return act.forward(bn.forward(conv_transpose.forward(x)));
            }

            public torch.Tensor forward_fuse(torch.Tensor x)
            {
                return act.forward(conv_transpose.forward(x));
            }
        }

        /// <summary>
        /// Focus module for concentrating feature information.
        /// </summary>
        internal class Focus : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly Conv conv;

            internal Focus(int c1, int c2, int k = 1, int s = 1, int? p = null, int g = 1, torch.nn.Module<torch.Tensor, torch.Tensor>? act = null, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(Focus))
            {
                this.conv = new Conv(c1 * 4, c2, k, s, p, g, act: act, device: device, dtype: dtype);
            }

            public override torch.Tensor forward(torch.Tensor x)
            {
                // Slices: x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]
                torch.Tensor slice1 = x[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(null, null, 2), torch.TensorIndex.Slice(null, null, 2)];
                torch.Tensor slice2 = x[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(1, null, 2), torch.TensorIndex.Slice(null, null, 2)];
                torch.Tensor slice3 = x[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(0, null, 2), torch.TensorIndex.Slice(1, null, 2)];
                torch.Tensor slice4 = x[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(1, null, 2), torch.TensorIndex.Slice(1, null, 2)];
                var cat = torch.cat(new[] { slice1, slice2, slice3, slice4 }, 1);
                return conv.forward(cat);
            }
        }

        /// <summary>
        /// Ghost Convolution module.
        /// </summary>
        internal class GhostConv : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly Conv cv1;
            private readonly Conv cv2;

            internal GhostConv(int c1, int c2, int k = 1, int s = 1, int g = 1, torch.nn.Module<torch.Tensor, torch.Tensor>? act = null, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(GhostConv))
            {
                int c_ = c2 / 2;
                this.cv1 = new Conv(c1, c_, k, s, null, g, act: act, device: device, dtype: dtype);
                this.cv2 = new Conv(c_, c_, 5, 1, null, c_, act: act, device: device, dtype: dtype);
            }

            public override torch.Tensor forward(torch.Tensor x)
            {
                var y = this.cv1.forward(x);
                return torch.cat(new[] { y, this.cv2.forward(y) }, 1);
            }
        }

        /// <summary>
        /// RepConv module with training and deploy modes.
        /// </summary>
        internal class RepConv : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly int g, c1, c2;
            private readonly torch.nn.Module<torch.Tensor, torch.Tensor> act;
            private Conv conv1;
            private Conv conv2;
            private BatchNorm2d bn;
            private Conv2d conv;          // fused conv used in deploy mode
            private torch.Tensor id_tensor;

            internal RepConv(int c1, int c2, int k = 3, int s = 1, int p = 1, int g = 1, int d = 1, torch.nn.Module<torch.Tensor, torch.Tensor>? act = null, bool bn = false, bool deploy = false, torch.Device? device = null, torch.ScalarType? dtype = null)
                : base(nameof(RepConv))
            {
                if (k != 3 || p != 1) throw new ArgumentException("RepConv only supports k=3 and p=1");
                this.g = g;
                this.c1 = c1;
                this.c2 = c2;
                this.act = act ?? torch.nn.SiLU();

                if (!deploy)
                {
                    this.conv1 = new Conv(c1, c2, k, s, p, g, d, act: torch.nn.Identity(), device: device, dtype: dtype);
                    this.conv2 = new Conv(c1, c2, 1, s, p - k / 2, g, d, act: torch.nn.Identity(), device: device, dtype: dtype);
                    if (bn && c2 == c1 && s == 1)
                    {
                        this.bn = torch.nn.BatchNorm2d(c1, eps: 0.001, momentum: 0.03, track_running_stats: true, device: device, dtype: dtype);
                    }
                }
                else
                {
                    // In deploy mode, we expect fuse_convs to be called or kernel/bias set externally
                    this.conv = torch.nn.Conv2d(c1, c2, k, s, p, d, groups: g, device: device, dtype: dtype);
                }
            }

            public override torch.Tensor forward(torch.Tensor x)
            {
                if (conv != null)  // deploy mode
                    return act.forward(conv.forward(x));

                torch.Tensor id_out = (bn is null) ? 0 : bn.forward(x);
                return act.forward(conv1!.forward(x) + conv2!.forward(x) + id_out);
            }

            public torch.Tensor forward_fuse(torch.Tensor x)
            {
                if (conv == null) throw new InvalidOperationException("RepConv not fused yet");
                return act.forward(conv.forward(x));
            }

            public void fuse_convs()
            {
                if (conv != null) return;
                using (torch.no_grad())
                {
                    var (kernel, bias) = get_equivalent_kernel_bias();
                    conv = torch.nn.Conv2d(
                        in_channels: conv1!.conv.in_channels,
                        out_channels: conv1.conv.out_channels,
                        kernel_size: conv1.conv.kernel_size[0],
                        stride: conv1.conv.stride[0],
                        padding: conv1.conv.padding[0],
                        dilation: conv1.conv.dilation[0],
                        groups: conv1.conv.groups,
                        bias: true
                    );
                    conv.weight.copy_(kernel);
                    conv.bias.copy_(bias);
                }
            }

            private (torch.Tensor kernel, torch.Tensor bias) get_equivalent_kernel_bias()
            {
                var (kernel3x3, bias3x3) = fuse_bn_tensor(conv1);
                var (kernel1x1, bias1x1) = fuse_bn_tensor(conv2);
                var (kernelId, biasId) = fuse_bn_tensor(bn);
                return (kernel3x3 + pad_1x1_to_3x3_tensor(kernel1x1) + kernelId, bias3x3 + bias1x1 + biasId);
            }

            private torch.Tensor pad_1x1_to_3x3_tensor(torch.Tensor kernel1x1)
            {
                if (kernel1x1 is null) return torch.zeros(1);
                return torch.nn.functional.pad(kernel1x1, new long[] { 1, 1, 1, 1 });
            }

            private (torch.Tensor kernel, torch.Tensor bias) fuse_bn_tensor(torch.nn.Module? branch)
            {
                if (branch is null) return (torch.zeros(1), torch.zeros(1));

                torch.Tensor kernel, running_mean, running_var, gamma, beta;
                double eps;

                if (branch is Conv convBranch)
                {
                    kernel = convBranch.conv.weight;
                    var bnLayer = convBranch.bn;
                    running_mean = bnLayer.running_mean;
                    running_var = bnLayer.running_var;
                    gamma = bnLayer.weight;
                    beta = bnLayer.bias;
                    eps = bnLayer.eps;
                }
                else if (branch is BatchNorm2d bnBranch)
                {
                    if (id_tensor is null)
                    {
                        int input_dim = c1 / g;
                        var kernelValue = new float[c1, input_dim, 3, 3];
                        for (int i = 0; i < c1; i++)
                            kernelValue[i, i % input_dim, 1, 1] = 1.0f;
                        id_tensor = torch.from_array(kernelValue).to(bnBranch.weight.device);
                    }
                    kernel = id_tensor!;
                    running_mean = bnBranch.running_mean;
                    running_var = bnBranch.running_var;
                    gamma = bnBranch.weight;
                    beta = bnBranch.bias;
                    eps = bnBranch.eps;
                }
                else
                    throw new ArgumentException("Unsupported branch type");

                var std = torch.sqrt(running_var + eps);
                var t = (gamma / std).reshape(-1, 1, 1, 1);
                return (kernel * t, beta - running_mean * gamma / std);
            }
        }


        /// <summary>
        /// Channel-attention module for feature recalibration.
        /// </summary>
        internal class ChannelAttention : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly AdaptiveAvgPool2d pool;
            private readonly Conv2d fc;
            private readonly Sigmoid act;

            internal ChannelAttention(int channels, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(ChannelAttention))
            {
                this.pool = torch.nn.AdaptiveAvgPool2d(1);
                fc = torch.nn.Conv2d(channels, channels, 1, 1, 0, bias: true, device: device, dtype: dtype);
                act = torch.nn.Sigmoid();
            }

            public override torch.Tensor forward(torch.Tensor x)
            {
                return x * act.forward(fc.forward(pool.forward(x)));
            }
        }

        /// <summary>
        /// Spatial-attention module for feature recalibration.
        /// </summary>
        internal class SpatialAttention : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly Conv2d cv1;
            private readonly Sigmoid act;

            internal SpatialAttention(int kernel_size = 7, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(SpatialAttention))
            {
                if (kernel_size != 3 && kernel_size != 7)
                {
                    throw new ArgumentException("kernel size must be 3 or 7");
                }
                int padding = kernel_size == 7 ? 3 : 1;
                cv1 = torch.nn.Conv2d(2, 1, kernel_size, padding: padding, bias: false, device: device, dtype: dtype);
                act = torch.nn.Sigmoid();
            }

            public override torch.Tensor forward(torch.Tensor x)
            {
                var mean = x.mean(new long[] { 1 }, keepdim: true);
                var max = x.max(1, keepdim: true).values;
                var concat = torch.cat(new[] { mean, max }, 1);
                return x * act.forward(cv1.forward(concat));
            }
        }

        /// <summary>
        /// Convolutional Block Attention Module.
        /// </summary>
        internal class CBAM : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly ChannelAttention channel_attention;
            private readonly SpatialAttention spatial_attention;

            internal CBAM(int c1, int kernel_size = 7, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(CBAM))
            {
                channel_attention = new ChannelAttention(c1, device: device, dtype: dtype);
                spatial_attention = new SpatialAttention(kernel_size, device: device, dtype: dtype);
            }

            public override torch.Tensor forward(torch.Tensor x)
            {
                return spatial_attention.forward(channel_attention.forward(x));
            }
        }

        /// <summary>
        /// Concatenate a list of torch.Tensors along specified dimension.
        /// </summary>
        internal class Concat : torch.nn.Module<List<torch.Tensor>, torch.Tensor>
        {
            private readonly int d;

            internal Concat(int dimension = 1) : base(nameof(Concat))
            {
                d = dimension;
            }

            public override torch.Tensor forward(List<torch.Tensor> x)
            {
                return torch.cat(x, d);
            }
        }

        /// <summary>
        /// Returns a particular index of the input.
        /// </summary>
        internal class Index : torch.nn.Module<List<torch.Tensor>, torch.Tensor>
        {
            private readonly int index;

            internal Index(int index = 0) : base(nameof(Index))
            {
                this.index = index;
            }

            public override torch.Tensor forward(List<torch.Tensor> x)
            {
                return x[index];
            }
        }
    }
}