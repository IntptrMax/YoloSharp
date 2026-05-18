using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static YoloSharp.Modules.Modules;

namespace Modules
{
    internal class Block
    {
        internal class Proto : Module<Tensor, Tensor>
        {
            private readonly Conv cv1;
            private readonly Conv cv2;
            private readonly Conv cv3;
            private readonly ConvTranspose2d upsample;
            internal Proto(int c1, int c_ = 256, int c2 = 32, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(Proto))
            {
                cv1 = new Conv(c1, c_, kernel_size: 3, device: device, dtype: dtype);
                upsample = ConvTranspose2d(c_, c_, 2, 2, 0, bias: true, device: device, dtype: dtype);  // nn.Upsample(scale_factor=2, mode='nearest')
                cv2 = new Conv(c_, c_, kernel_size: 3, device: device, dtype: dtype);
                cv3 = new Conv(c_, c2, kernel_size: 1, device: device, dtype: dtype);
                RegisterComponents();
            }

            public override Tensor forward(Tensor x)
            {
                return cv3.forward(cv2.forward(upsample.forward(cv1.forward(x))));
            }
        }

    }
}
