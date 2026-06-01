using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static YoloSharp.Modules.Block;

namespace YoloSharp.Modules
{
    internal class Block
    {
        /// <summary>
        /// Integral module of Distribution Focal Loss (DFL).
        /// </summary>
        /// <remarks>
        /// Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        /// </remarks>
        internal class DFL : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly Conv2d conv;
            private readonly int c1;

            /// <summary>
            /// Initialize a convolutional layer with a given number of input channels.
            /// </summary>
            /// <param name="c1">Number of input channels.</param>
            /// <param name="device">Device type.</param>
            /// <param name="dtype">Scaler type.</param>
            internal DFL(int c1 = 16, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(DFL))
            {
                this.conv = torch.nn.Conv2d(c1, 1, 1, bias: false, device: device, dtype: dtype);
                torch.Tensor x = torch.arange(c1, device: device, dtype: torch.ScalarType.Float32);
                this.conv.weight = (torch.nn.Parameter(x.view(1, c1, 1, 1)));
                this.c1 = c1;
            }

            /// <summary>
            /// Apply the DFL module to input tensor and return transformed output.
            /// </summary>
            /// <param name="x"></param>
            /// <returns></returns>
            public override torch.Tensor forward(torch.Tensor x)
            {
                long b = x.shape[0];  // batch, channels, anchors
                long a = x.shape[2];
                return conv.forward(x.view(b, 4, c1, a).transpose(2, 1).softmax(1)).view(b, 4, a);
            }
        }

        /// <summary>
        /// Ultralytics YOLO models mask Proto module for segmentation models.
        /// </summary>
        internal class Proto : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly Convs.Conv cv1;
            private readonly Convs.Conv cv2;
            private readonly Convs.Conv cv3;
            private readonly ConvTranspose2d upsample;

            /// <summary>
            /// Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.
            /// </summary>
            /// <param name="c1">Input channels.</param>
            /// <param name="c_">Intermediate channels.</param>
            /// <param name="c2">Output channels (number of protos).</param>
            /// <param name="device">Device type.</param>
            /// <param name="dtype">Scaler type.</param>
            internal Proto(int c1, int c_ = 256, int c2 = 32, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(Proto))
            {
                cv1 = new Convs.Conv(c1, c_, k: 3, device: device, dtype: dtype);
                upsample = torch.nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias: true, device: device, dtype: dtype);  // nn.Upsample(scale_factor=2, mode='nearest')
                cv2 = new Convs.Conv(c_, c_, k: 3, device: device, dtype: dtype);
                cv3 = new Convs.Conv(c_, c2, k: 1, device: device, dtype: dtype);

            }

            /// <summary>
            /// Perform a forward pass through layers using an upsampled input image.
            /// </summary>
            /// <param name="x"></param>
            /// <returns></returns>
            public override torch.Tensor forward(torch.Tensor x)
            {
                return cv3.forward(cv2.forward(upsample.forward(cv1.forward(x))));
            }
        }

        /// <summary>
        /// StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.
        /// </summary>
        /// <see cref="https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py"/>
        internal class HGStem : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly Convs.Conv stem1;
            private readonly Convs.Conv stem2a;
            private readonly Convs.Conv stem2b;
            private readonly Convs.Conv stem3;
            private readonly Convs.Conv stem4;
            private readonly MaxPool2d pool;

            /// <summary>
            /// Initialize the StemBlock of PPHGNetV2.
            /// </summary>
            /// <param name="c1">Input channels.</param>
            /// <param name="cm">Middle channels.</param>
            /// <param name="c2">Output channels.</param>
            internal HGStem(int c1, int cm, int c2, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(HGStem))
            {
                this.stem1 = new Convs.Conv(c1, cm, 3, 2, act: torch.nn.ReLU(), device: device, dtype: dtype);
                this.stem2a = new Convs.Conv(cm, cm / 2, 2, 1, 0, act: torch.nn.ReLU(), device: device, dtype: dtype);
                this.stem2b = new Convs.Conv(cm / 2, cm, 2, 1, 0, act: torch.nn.ReLU(), device: device, dtype: dtype);
                this.stem3 = new Convs.Conv(cm * 2, cm, 3, 2, act: torch.nn.ReLU(), device: device, dtype: dtype);
                this.stem4 = new Convs.Conv(cm, c2, 1, 1, act: torch.nn.ReLU(), device: device, dtype: dtype);
                this.pool = torch.nn.MaxPool2d(kernel_size: 2, stride: 1, padding: 0, ceil_mode: true);
            }

            /// <summary>
            /// Forward pass of a PPHGNetV2 backbone layer.
            /// </summary>
            /// <param name="x"></param>
            /// <returns></returns>
            public override torch.Tensor forward(torch.Tensor x)
            {
                using (torch.NewDisposeScope())
                {
                    x = this.stem1.forward(x);
                    x = torch.nn.functional.pad(x, new long[] { 0, 1, 0, 1 });
                    torch.Tensor x2 = this.stem2a.forward(x);
                    x2 = torch.nn.functional.pad(x2, new long[] { 0, 1, 0, 1 });
                    x2 = this.stem2b.forward(x2);
                    torch.Tensor x1 = this.pool.forward(x);
                    x = torch.cat(new torch.Tensor[] { x1, x2 }, dim: 1);
                    x = this.stem3.forward(x);
                    x = this.stem4.forward(x);
                    return x.MoveToOuterDisposeScope();
                }
            }
        }

        /// <summary>
        /// HG_Block of PPHGNetV2 with 2 convolutions and LightConv.
        /// </summary>
        /// <see cref="https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py"/>
        internal class HGBlock : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly ModuleList<torch.nn.Module<torch.Tensor, torch.Tensor>> m;
            private readonly Convs.Conv sc;
            private readonly Convs.Conv ec;
            private readonly bool add;

            internal HGBlock(int c1, int cm, int c2, int k = 3, int n = 6, bool lightconv = false, bool shortcut = false, torch.nn.Module<torch.Tensor, torch.Tensor>? act = null, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(HGBlock))
            {
                torch.nn.Module<torch.Tensor, torch.Tensor> localAct = act ?? torch.nn.ReLU();

                this.m = new ModuleList<torch.nn.Module<torch.Tensor, torch.Tensor>>(
                    Enumerable.Range(0, n).Select(i =>
                        lightconv
                            ? new Convs.LightConv(i == 0 ? c1 : cm, cm, k: k, act: localAct, device: device, dtype: dtype)
                            : (torch.nn.Module<torch.Tensor, torch.Tensor>)new Convs.Conv(i == 0 ? c1 : cm, cm, k: k, act: localAct, device: device, dtype: dtype)
                    ).ToArray()
                );

                this.sc = new Convs.Conv(c1 + n * cm, c2 / 2, 1, 1, act: localAct, device: device, dtype: dtype);
                this.ec = new Convs.Conv(c2 / 2, c2, 1, 1, act: localAct, device: device, dtype: dtype);
                this.add = shortcut && c1 == c2;
            }

            /// <summary>
            /// Forward pass of a PPHGNetV2 backbone layer.
            /// </summary>
            /// <param name="x"></param>
            /// <returns></returns>
            public override torch.Tensor forward(torch.Tensor x)
            {
                using (torch.NewDisposeScope())
                {
                    List<torch.Tensor> y = new List<torch.Tensor> { x };
                    foreach (var layer in m)
                    {
                        y.Add(layer.forward(y[y.Count - 1]));
                    }

                    torch.Tensor cat = torch.cat(y.ToArray(), dim: 1);
                    torch.Tensor @out = ec.forward(sc.forward(cat));

                    return (add ? @out + x : @out).MoveToOuterDisposeScope();
                }
            }
        }

        /// <summary>
        /// Spatial Pyramid Pooling (SPP) layer.
        /// </summary>
        /// <see cref="https://arxiv.org/abs/1406.4729"/>
        internal class SPP : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly Convs.Conv cv1;
            private readonly Convs.Conv cv2;
            private readonly ModuleList<MaxPool2d> m;

            /// <summary>
            /// Initialize the SPP layer with input/output channels and pooling kernel sizes.
            /// </summary>
            /// <param name="c1">Input channels.</param>
            /// <param name="c2">Output channels.</param>
            /// <param name="k">Kernel sizes for max pooling.</param>
            internal SPP(int c1, int c2, int[]? k = null, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(SPP))
            {
                int[] localK = k ?? new int[] { 5, 9, 13 };
                int c_ = c1 / 2;  // hidden channels
                this.cv1 = new Convs.Conv(c1, c_, 1, 1, device: device, dtype: dtype);
                this.cv2 = new Convs.Conv(c_ * (k.Length + 1), c2, 1, 1, device: device, dtype: dtype);
                this.m = new ModuleList<MaxPool2d>(k.Select(x => torch.nn.MaxPool2d(kernel_size: x, stride: 1, padding: x / 2)).ToArray());
            }

            /// <summary>
            /// Forward pass of the SPP layer, performing spatial pyramid pooling.
            /// </summary>
            /// <param name="x"></param>
            /// <returns></returns>
            public override torch.Tensor forward(torch.Tensor x)
            {
                x = this.cv1.forward(x);
                //return this.cv2.forward(torch.cat([x] + [m(x) for m in this.m], 1))
                List<torch.Tensor> y = new List<torch.Tensor> { x };
                y.AddRange(this.m.Select(m => m.forward(x)).ToArray());
                return this.cv2.forward(torch.cat(y, 1));

            }
        }

        /// <summary>
        /// Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher.
        /// </summary>
        internal class SPPF : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly Convs.Conv cv1;
            private readonly Convs.Conv cv2;
            private readonly MaxPool2d m;
            private readonly int n;
            private readonly bool add;

            /// <summary>
            /// Initialize the SPPF layer with given input/output channels and kernel size.
            /// </summary>
            /// <param name="c1">Input channels.</param>
            /// <param name="c2">Output channels.</param>
            /// <param name="k">Kernel size.</param>
            /// <param name="n">Number of pooling iterations.</param>
            /// <param name="shortcut">Whether to use shortcut connection.</param>
            /// <param name="device">Device type.</param>
            /// <param name="dtype">Scaler type.</param>
            internal SPPF(int c1, int c2, int k = 5, int n = 3, bool shortcut = false, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(SPPF))
            {
                int c_ = c1 / 2; // hidden channels
                this.cv1 = new Convs.Conv(c1, c_, 1, 1, act: torch.nn.Identity(), device: device, dtype: dtype);
                this.cv2 = new Convs.Conv(c_ * (n + 1), c2, 1, 1, device: device, dtype: dtype);
                this.m = torch.nn.MaxPool2d(kernel_size: k, stride: 1, padding: k / 2);
                this.n = n;
                this.add = shortcut && (c1 == c2);
            }

            /// <summary>
            /// Apply sequential pooling operations to input and return concatenated feature maps.
            /// </summary>
            /// <param name="x"></param>
            /// <returns></returns>
            public override torch.Tensor forward(torch.Tensor x)
            {
                using (torch.NewDisposeScope())
                {
                    List<torch.Tensor> y = new List<torch.Tensor> { this.cv1.forward(x) };
                    for (int i = 0; i < this.n; i++)
                    {
                        y.Add(this.m.forward(y[y.Count - 1]));
                    }
                    torch.Tensor result = this.cv2.forward(torch.cat(y, 1));
                    return (this.add ? result + x : result).MoveToOuterDisposeScope();
                }
            }


        }

        /// <summary>
        /// CSP Bottleneck with 1 convolution.
        /// </summary>
        internal class C1 : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly Convs.Conv cv1;
            private readonly Sequential m;

            /// <summary>
            /// Initialize the CSP Bottleneck with 1 convolution.
            /// </summary>
            /// <param name="c1">Input channels.</param>
            /// <param name="c2">Output channels.</param>
            /// <param name="n">Number of convolutions.</param>
            /// <param name="device">Device type.</param>
            /// <param name="dtype">Scaler type.</param>
            internal C1(int c1, int c2, int n = 1, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(C1))
            {
                this.cv1 = new Convs.Conv(c1, c2, 1, 1, device: device, dtype: dtype);
                this.m = torch.nn.Sequential(Enumerable.Range(0, 1).Select(_ => new Convs.Conv(c2, c2, 3, device: device, dtype: dtype)).ToArray());
            }

            /// <summary>
            /// Apply convolution and residual connection to input tensor.
            /// </summary>
            /// <param name="x"></param>
            /// <returns></returns>
            public override torch.Tensor forward(torch.Tensor x)
            {
                torch.Tensor y = this.cv1.forward(x);
                return this.m.forward(y) + y;
            }
        }

        /// <summary>
        /// CSP Bottleneck with 2 convolutions.
        /// </summary>
        internal class C2 : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly int c;
            private readonly Convs.Conv cv1;
            private readonly Convs.Conv cv2;
            private readonly Sequential m;

            /// <summary>
            /// Initialize a CSP Bottleneck with 2 convolutions.
            /// </summary>
            /// <param name="c1">Input channels.</param>
            /// <param name="c2">Output channels.</param>
            /// <param name="n">Number of Bottleneck blocks.</param>
            /// <param name="shortcut">Whether to use shortcut connections.</param>
            /// <param name="g">Groups for convolutions.</param>
            /// <param name="e">Expansion ratio.</param>
            /// <param name="device">Device type.</param>
            /// <param name="dtype">Scaler type.</param>
            internal C2(int c1, int c2, int n = 1, bool shortcut = true, int g = 1, float e = 0.5f, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(C2))
            {
                this.c = (int)(c2 * e);  // hidden channels
                this.cv1 = new Convs.Conv(c1, 2 * this.c, 1, 1, device: device, dtype: dtype);
                this.cv2 = new Convs.Conv(2 * this.c, c2, 1, device: device, dtype: dtype); // optional act=FReLU(c2)
                this.m = torch.nn.Sequential(Enumerable.Range(0, n).Select(_ => new Bottleneck(this.c, this.c, shortcut, g, k: new int[] { 3, 3 }, e: 1.0f, device: device, dtype: dtype)).ToArray());
            }

            /// <summary>
            /// Forward pass through the CSP bottleneck with 2 convolutions.
            /// </summary>
            /// <param name="x"></param>
            /// <returns></returns>
            public override torch.Tensor forward(torch.Tensor x)
            {
                using (torch.NewDisposeScope())
                {
                    torch.Tensor[] ab = this.cv1.forward(x).chunk(2, 1);
                    torch.Tensor a = ab[0];
                    torch.Tensor b = ab[1];
                    return this.cv2.forward(torch.cat(new torch.Tensor[] { this.m.forward(a), b }, 1)).MoveToOuterDisposeScope();
                }
            }
        }

        /// <summary>
        /// Faster Implementation of CSP Bottleneck with 2 convolutions.
        /// </summary>
        internal class C2f : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly int c;
            private readonly Convs.Conv cv1;
            private readonly Convs.Conv cv2;
            private readonly ModuleList<Bottleneck> m;

            internal C2f(int c1, int c2, int n = 1, bool shortcut = false, int g = 1, float e = 0.5f, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(C2f))
            {
                this.c = (int)(c2 * e);  // hidden channels
                this.cv1 = new Convs.Conv(c1, 2 * this.c, 1, 1);
                this.cv2 = new Convs.Conv((2 + n) * this.c, c2, 1);  // optional act=FReLU(c2)
                this.m = nn.ModuleList<Bottleneck>(Enumerable.Range(0, n).Select(_ => new Bottleneck(this.c, this.c, shortcut, g, k: new int[] { 3, 3 }, e = 1.0f)).ToArray());
            }

            public override Tensor forward(Tensor input)
            {
                throw new NotImplementedException();
            }
        }

        /// <summary>
        /// Standard bottleneck.
        /// </summary>
        internal class Bottleneck : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly Convs.Conv cv1;
            private readonly Convs.Conv cv2;
            private readonly bool add;

            /// <summary>
            /// Initialize a standard bottleneck module.
            /// </summary>
            /// <param name="c1">Input channels.</param>
            /// <param name="c2">Output channels.</param>
            /// <param name="shortcut">Whether to use shortcut connection.</param>
            /// <param name="g">Groups for convolutions.</param>
            /// <param name="k">Kernel sizes for convolutions.</param>
            /// <param name="e">Expansion ratio.</param>
            /// <param name="device">Device type.</param>
            /// <param name="dtype">Scaler type.</param>
            internal Bottleneck(int c1, int c2, bool shortcut = true, int g = 1, int[]? k = null, float e = 0.5f, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(Bottleneck))
            {
                int[] localk = k ?? new int[] { 3, 3 };
                int c_ = (int)(c2 * e);  // hidden channels
                this.cv1 = new Convs.Conv(c1, c_, k[0], 1, device: device, dtype: dtype);
                this.cv2 = new Convs.Conv(c_, c2, k[1], 1, g: g, device: device, dtype: dtype);
                this.add = shortcut && c1 == c2;
                RegisterComponents();
            }

            /// <summary>
            /// Apply bottleneck with optional shortcut connection.
            /// </summary>
            /// <param name="x"></param>
            /// <returns></returns>
            public override torch.Tensor forward(torch.Tensor x)
            {
                return this.add ? (x + this.cv2.forward(this.cv1.forward(x))) : (this.cv2.forward(this.cv1.forward(x)));
            }
        }

    }
}
