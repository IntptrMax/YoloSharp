using System.Numerics;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Modules
{
    internal class Convs
    {
        private static int autopad(int k, int? p = null, int d = 1)
        {
            if (d > 1)
            {
                k = d * (k - 1) + 1;
            }
            if (p is null)
            {
                p = k / 2;
            }
            return p.Value;
        }

        private static int[] autopad(int[] k, int[]? p = null, int d = 1)
        {
            if (d > 1)
            {
                k = k.Select(x => d * (x - 1) + 1).ToArray();
            }
            if (p is null)
            {
                p = k.Select(x => x / 2).ToArray();
            }
            return p;
        }

        internal class Conv : Module<Tensor, Tensor>
        {
            protected readonly Conv2d conv;
            protected readonly BatchNorm2d bn;
            protected readonly Module<Tensor, Tensor> act;
            private readonly double eps = 0.001;
            private readonly double momentum = 0.03;

            internal Conv(int c1, int c2, int k = 1, int s = 1, int? p = null, int g = 1, int d = 1, Func<Module<Tensor, Tensor>>? act = null, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(Conv))
            {
                p = p ?? k / 2;
                conv = Conv2d(c1, c2, k, s, p.Value, groups: g, bias: false, dilation: d, device: device, dtype: dtype);
                bn = BatchNorm2d(c2, eps: eps, momentum: momentum, track_running_stats: true, device: device, dtype: dtype);
                this.act = (act ?? SiLU)();
                RegisterComponents();
            }

            public override Tensor forward(Tensor x)
            {
                return act.forward(bn.forward(conv.forward(x)));
            }

            public virtual Tensor forward_fuse(Tensor x)
            {
                return act.forward(conv.forward(x));
            }

        }

        internal class Conv2 : Conv
        {
            private readonly Conv2d cv2;

            internal Conv2(int c1, int c2, int k = 3, int s = 1, int? p = null, int g = 1, int d = 1, Func<Module<Tensor, Tensor>>? act = null, Device? device = null, torch.ScalarType? dtype = null) : base(c1: c1, c2: c2, k: k, s: s, p: p, g: g, d: d, act: act, device: device, dtype: dtype)
            {
                this.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups: g, dilation: d, bias: false);  // add 1x1 conv
                RegisterComponents();
            }

            public override Tensor forward(Tensor x)
            {
                return this.act.forward(this.bn.forward(this.conv.forward(x) + this.cv2.forward(x)));
            }

            public override Tensor forward_fuse(Tensor x)
            {
                return act.forward(bn.forward(conv.forward(x)));
            }
        }

        internal class DWConv : Conv
        {
            internal DWConv(int in_channels, int out_channels, int kernel_size = 1, int stride = 1, int d = 1, Func<Module<Tensor, Tensor>>? act = null, bool bias = false, Device? device = null, torch.ScalarType? dtype = null) : base(in_channels, out_channels, kernel_size, stride, g: (int)BigInteger.GreatestCommonDivisor(in_channels, out_channels), d: d, act: act, device: device, dtype: dtype)
            {

            }
        }
    }
}
