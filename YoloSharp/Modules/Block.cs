using TorchSharp;
using TorchSharp.Modules;
using YoloSharp.Types;

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
                RegisterComponents();
            }

            /// <summary>
            /// Apply the DFL module to input torch.Tensor and return transformed output.
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
                upsample = torch.nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias: true, device: device, dtype: dtype);  // torch.nn.Upsample(scale_factor=2, mode='nearest')
                cv2 = new Convs.Conv(c_, c_, k: 3, device: device, dtype: dtype);
                cv3 = new Convs.Conv(c_, c2, k: 1, device: device, dtype: dtype);
                RegisterComponents();
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
                RegisterComponents();
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
                RegisterComponents();
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
                RegisterComponents();
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
                RegisterComponents();
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
                this.m = torch.nn.Sequential(Enumerable.Range(0, 1).Select(_ => new Convs.Conv(c2, c2, 3, device: device, dtype: dtype)));
                RegisterComponents();
            }

            /// <summary>
            /// Apply convolution and residual connection to input torch.Tensor.
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
                this.m = torch.nn.Sequential(Enumerable.Range(0, n).Select(_ => new Bottleneck(this.c, this.c, shortcut, g, k: new int[] { 3, 3 }, e: 1.0f, device: device, dtype: dtype)));
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
                this.cv1 = new Convs.Conv(c1, 2 * this.c, 1, 1, device: device, dtype: dtype);
                this.cv2 = new Convs.Conv((2 + n) * this.c, c2, 1, device: device, dtype: dtype);  // optional act=FReLU(c2)
                this.m = torch.nn.ModuleList<Bottleneck>(Enumerable.Range(0, n).Select(_ => new Bottleneck(this.c, this.c, shortcut, g, k: new int[] { 3, 3 }, e = 1.0f, device: device, dtype: dtype)).ToArray());
                RegisterComponents();
            }

            public override torch.Tensor forward(torch.Tensor x)
            {
                using (torch.NewDisposeScope())
                {
                    List<torch.Tensor> y = this.cv1.forward(x).chunk(2, 1).ToList();
                    for (int i = 0; i < m.Count; i++)
                    {
                        y.Add(m[i].forward(y.Last()));
                    }
                    return this.cv2.forward(torch.cat(y.ToArray(), 1)).MoveToOuterDisposeScope();
                }
            }
        }

        /// <summary>
        /// CSP Bottleneck with 3 convolutions.
        /// </summary>
        internal class C3 : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            internal readonly Convs.Conv cv1;
            internal readonly Convs.Conv cv2;
            internal readonly Convs.Conv cv3;
            protected torch.nn.Module<torch.Tensor, torch.Tensor> m;

            /// <summary>
            /// Initialize the CSP Bottleneck with 3 convolutions.
            /// </summary>
            /// <param name="c1">Input channels.</param>
            /// <param name="c2">Output channels.</param>
            /// <param name="n">Number of Bottleneck blocks.</param>
            /// <param name="shortcut">Whether to use shortcut connections.</param>
            /// <param name="g">Groups for convolutions.</param>
            /// <param name="e">Expansion ratio.</param>
            /// <param name="device">Device type.</param>
            /// <param name="dtype">Scalar type.</param>
            internal C3(int c1, int c2, int n = 1, bool shortcut = true, int g = 1, float e = 0.5f, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(C3))
            {
                int c_ = (int)(c2 * e);
                cv1 = new Convs.Conv(c1, c_, 1, 1, device: device, dtype: dtype);
                cv2 = new Convs.Conv(c1, c_, 1, 1, device: device, dtype: dtype);
                cv3 = new Convs.Conv(2 * c_, c2, 1, device: device, dtype: dtype);

                m = torch.nn.Sequential(Enumerable.Range(0, n).Select(i => new Block.Bottleneck(c_, c_, k: new int[] { 1, 3 }, shortcut: shortcut, g: g, e: 1.0f, device: device, dtype: dtype)));
                RegisterComponents();
            }

            /// <summary>
            /// Forward pass through the CSP bottleneck with 3 convolutions.
            /// </summary>
            /// <param name="input"></param>
            /// <returns></returns>
            public override torch.Tensor forward(torch.Tensor input)
            {
                return cv3.forward(torch.cat(new torch.Tensor[] { m.forward(cv1.forward(input)), cv2.forward(input) }, 1));
            }
        }

        internal class C3x : C3
        {
            internal C3x(int c1, int c2, int n = 1, bool shortcut = true, int g = 1, float e = 0.5f, torch.Device? device = null, torch.ScalarType? dtype = null) : base(c1, c2, n, shortcut, g, e, device, dtype)
            {
                // Override the m with a more efficient implementation of BottleneckCSP with 3 convolutions.
                int c_ = (int)(c2 * e);
                base._internal_submodules.Remove("m");
                m = torch.nn.Sequential(Enumerable.Range(0, n).Select(i => new Block.Bottleneck(c_, c_, k: new int[] { 1, 3 }, shortcut: shortcut, g: g, e: 1.0f, device: device, dtype: dtype)));
                register_module("m", m);
            }
        }

        /// <summary>
        /// Rep C3.
        /// </summary>
        internal class RepC3 : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly Convs.Conv cv1;
            private readonly Convs.Conv cv2;
            private readonly torch.nn.Module<torch.Tensor, torch.Tensor> cv3;
            private readonly Sequential m;

            /// <summary>
            /// Initialize RepC3 module with RepConv blocks.
            /// </summary>
            /// <param name="c1">Input channels.</param>
            /// <param name="c2">Output channels.</param>
            /// <param name="n">Number of RepConv blocks.</param>
            /// <param name="e">Expansion ratio.</param>
            /// <param name="device">Device type.</param>
            /// <param name="dtype">Scalar type.</param>
            internal RepC3(int c1, int c2, int n = 3, float e = 1.0f, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(RepC3))
            {
                int c_ = (int)(c2 * e);  // hidden channels
                this.cv1 = new Convs.Conv(c1, c_, 1, 1, device: device, dtype: dtype);
                this.cv2 = new Convs.Conv(c1, c_, 1, 1, device: device, dtype: dtype);
                this.m = torch.nn.Sequential(Enumerable.Range(0, n).Select(_ => new Convs.RepConv(c_, c_, device: device, dtype: dtype)));
                this.cv3 = c_ != c2 ? new Convs.Conv(c_, c2, 1, 1, device: device, dtype: dtype) : torch.nn.Identity();
                RegisterComponents();
            }

            /// <summary>
            /// Forward pass of RepC3 module.
            /// </summary>
            /// <param name="x"></param>
            /// <returns></returns>
            public override torch.Tensor forward(torch.Tensor x)
            {
                return cv3.forward(torch.cat(new torch.Tensor[] { m.forward(cv1.forward(x)), cv2.forward(x) }, 1));
            }
        }

        /// <summary>
        /// Initialize C3 module with TransformerBlock.
        /// </summary>
        internal class C3TR : C3
        {
            /// <summary>
            /// Initialize C3 module with TransformerBlock.
            /// </summary>
            /// <param name="c1">Input channels.</param>
            /// <param name="c2">Output channels.</param>
            /// <param name="n">Number of Transformer blocks.</param>
            /// <param name="shortcut">Whether to use shortcut connections.</param>
            /// <param name="g">Groups for convolutions.</param>
            /// <param name="e">Expansion ratio.</param>
            /// <param name="device">Device type.</param>
            /// <param name="dtype">Scalar type.</param>
            internal C3TR(int c1, int c2, int n = 1, bool shortcut = true, int g = 1, float e = 0.5f, torch.Device? device = null, torch.ScalarType? dtype = null) : base(c1, c2, n, shortcut, g, e, device, dtype)
            {
                // Override the m with a more efficient implementation of BottleneckCSP with 3 convolutions.
                int c_ = (int)(c2 * e);
                base._internal_submodules.Remove("m");
                m = new Transformer.TransformerBlock(c_, c_, 4, n, device: device, dtype: dtype);
                register_module("m", m);
            }
        }

        /// <summary>
        /// "C3 module with GhostBottleneck().
        /// </summary>
        internal class C3Ghost : C3
        {
            internal C3Ghost(int c1, int c2, int n = 1, bool shortcut = true, int g = 1, float e = 0.5f, torch.Device? device = null, torch.ScalarType? dtype = null) : base(c1, c2, n, shortcut, g, e, device, dtype)
            {
                // Override the m with a more efficient implementation of BottleneckCSP with 3 convolutions.
                int c_ = (int)(c2 * e);
                base._internal_submodules.Remove("m");
                m = torch.nn.Sequential(Enumerable.Range(0, n).Select(i => new GhostBottleneck(c_, c_, device: device, dtype: dtype)));
                register_module("m", m);
            }
        }

        /// <summary>
        /// Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones.
        /// </summary>
        internal class GhostBottleneck : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly Sequential conv;
            private readonly torch.nn.Module<torch.Tensor, torch.Tensor> shortcut;

            internal GhostBottleneck(int c1, int c2, int k = 3, int s = 1, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(GhostBottleneck))
            {
                int c_ = c2 / 2;  // hidden channels
                this.conv = torch.nn.Sequential(
                            new Convs.GhostConv(c1, c_, 1, 1, device: device, dtype: dtype),  // pw
                            s == 2 ? new Convs.DWConv(c_, c_, k, s, act: torch.nn.Identity(), device: device, dtype: dtype) : torch.nn.Identity(),  // dw
                            new Convs.GhostConv(c_, c2, 1, 1, act: torch.nn.Identity(), device: device, dtype: dtype)  // pw-linear
                            );
                this.shortcut = s == 2 ? torch.nn.Sequential(new Convs.DWConv(c1, c1, k, s, act: torch.nn.Identity(), device: device, dtype: dtype), new Convs.Conv(c1, c2, 1, 1, act: torch.nn.Identity(), device: device, dtype: dtype)) : torch.nn.Identity();

                RegisterComponents();
            }

            /// <summary>
            /// Apply skip connection and addition to input torch.Tensor.
            /// </summary>
            /// <param name="x"></param>
            /// <returns></returns>
            public override torch.Tensor forward(torch.Tensor x)
            {
                return this.conv.forward(x) + this.shortcut.forward(x);
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


        internal class C3k : C3
        {
            internal C3k(int inChannels, int outChannels, int n = 1, bool shortcut = true, int groups = 1, float e = 0.5f, torch.Device? device = null, torch.ScalarType? dtype = null) : base(inChannels, outChannels, n, shortcut, groups, e, device, dtype)
            {
                int c = (int)(outChannels * e);
                base._internal_submodules.Remove("m");
                m = torch.nn.Sequential(Enumerable.Range(0, n).Select(_ => new Block.Bottleneck(c, c, k: new int[] { 3, 3 }, shortcut: shortcut, g: groups, e: 1.0f, device: device, dtype: dtype)).ToArray());
                register_module("m", m);
            }
        }


        internal class C3k2 : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            internal readonly Convs.Conv cv1;
            internal readonly Convs.Conv cv2;
            internal readonly ModuleList<torch.nn.Module> m;
            internal readonly int c;
            internal C3k2(int inChannels, int outChannels, int n = 1, bool c3k = false, float e = 0.5f, int groups = 1, bool shortcut = true, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(C3k2))
            {
                c = (int)(outChannels * e);
                cv1 = new Convs.Conv(inChannels, 2 * c, 1, 1, device: device, dtype: dtype);
                cv2 = new Convs.Conv((2 + n) * c, outChannels, 1, device: device, dtype: dtype);  // optional act=FReLU(outChannels)
                m = new ModuleList<torch.nn.Module>();
                for (int i = 0; i < n; i++)
                {
                    if (c3k)
                    {
                        m.append(new Block.C3k(c, c, 2, shortcut, groups, device: device, dtype: dtype));
                    }
                    else
                    {
                        m.append(new Block.Bottleneck(c, c, k: new int[] { 3, 3 }, shortcut: shortcut, g: groups, device: device, dtype: dtype));
                    }
                }
                RegisterComponents();
            }

            public override torch.Tensor forward(torch.Tensor input)
            {
                using (torch.NewDisposeScope())
                {
                    List<torch.Tensor> y = cv1.forward(input).chunk(2, 1).ToList();
                    for (int i = 0; i < m.Count; i++)
                    {
                        y.Add(((torch.nn.Module<torch.Tensor, torch.Tensor>)m[i]).forward(y.Last()));
                    }
                    torch.Tensor result = cv2.forward(torch.cat(y, 1));
                    return result.MoveToOuterDisposeScope();
                }
            }
        }

        internal class C2PSA : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly int c;
            private readonly Convs.Conv cv1;
            private readonly Convs.Conv cv2;
            private readonly Sequential m;

            internal C2PSA(int inChannel, int outChannel, int n = 1, float e = 0.5f, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(C2PSA))
            {
                if (inChannel != outChannel)
                {
                    throw new ArgumentException("in channel not equals to out channel");
                }
                c = (int)(inChannel * e);
                cv1 = new Convs.Conv(inChannel, 2 * c, 1, 1, device: device, dtype: dtype);
                cv2 = new Convs.Conv(2 * c, outChannel, 1, device: device, dtype: dtype);

                m = torch.nn.Sequential(Enumerable.Range(0, n).Select(_ => new PSABlock(c, attn_ratio: 0.5f, num_heads: c / 64, device: device, dtype: dtype)));

                RegisterComponents();
            }

            public override torch.Tensor forward(torch.Tensor x)
            {
                using (torch.NewDisposeScope())
                {
                    torch.Tensor[] ab = cv1.forward(x).split(new long[] { c, c }, dim: 1);
                    torch.Tensor a = ab[0];
                    torch.Tensor b = ab[1];
                    b = m.forward(b);
                    return cv2.forward(torch.cat(new torch.Tensor[] { a, b }, 1)).MoveToOuterDisposeScope();
                }
            }
        }

        internal class PSABlock : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly Attention attn; // can use ScaledDotProductAttention instead
            private readonly Sequential ffn;
            private readonly bool add;

            internal PSABlock(int c, float attn_ratio = 0.5f, int num_heads = 8, bool shortcut = true, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(PSABlock))
            {
                attn = new Attention(c, num_heads, attn_ratio, attentionType: AttentionType.SelfAttention, device: device, dtype: dtype);
                ffn = torch.nn.Sequential(new Convs.Conv(c, c * 2, 1, device: device, dtype: dtype), new Convs.Conv(c * 2, c, 1, device: device, dtype: dtype));
                add = shortcut;
                RegisterComponents();
            }

            public override torch.Tensor forward(torch.Tensor x)
            {
                x = add ? x + attn.forward(x) : attn.forward(x);
                x = add ? x + ffn.forward(x) : ffn.forward(x);
                return x;
            }
        }

        internal class Attention : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly int num_heads;
            private readonly int head_dim;
            private readonly int key_dim;
            private readonly float scale;

            private readonly Convs.Conv qkv;
            private readonly Convs.Conv proj;
            private readonly Convs.Conv pe;

            private AttentionType attentionType;

            internal Attention(int dim, int num_heads = 8, float attn_ratio = 0.5f, AttentionType attentionType = AttentionType.SelfAttention, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(Attention))
            {
                this.num_heads = num_heads;
                head_dim = dim / num_heads;
                key_dim = (int)(head_dim * attn_ratio);
                scale = (float)Math.Pow(key_dim, -0.5);

                int nh_kd = key_dim * num_heads;
                int h = dim + nh_kd * 2;

                qkv = new Convs.Conv(dim, h, 1, device: device, dtype: dtype);
                proj = new Convs.Conv(dim, dim, 1, device: device, dtype: dtype);
                pe = new Convs.Conv(dim, dim, 3, 1, g: dim, device: device, dtype: dtype);

                this.attentionType = attentionType;
                RegisterComponents();
            }

            public override torch.Tensor forward(torch.Tensor x)
            {
                using (torch.NewDisposeScope())
                {
                    long B = x.shape[0];
                    long C = x.shape[1];
                    long H = x.shape[2];
                    long W = x.shape[3];

                    long N = H * W;

                    torch.Tensor qkv = this.qkv.forward(x);

                    torch.Tensor[] qkv_mix = qkv.view(B, num_heads, key_dim * 2 + head_dim, N).split(new long[] { key_dim, key_dim, head_dim }, dim: 2);
                    torch.Tensor q = qkv_mix[0];
                    torch.Tensor k = qkv_mix[1];
                    torch.Tensor v = qkv_mix[2];

                    switch (attentionType)
                    {
                        case AttentionType.SelfAttention:
                            {
                                torch.Tensor attn = q.transpose(-2, -1).matmul(k) * scale;
                                attn = attn.softmax(dim: -1);
                                x = v.matmul(attn.transpose(-2, -1)).view(B, C, H, W) + pe.forward(v.reshape(B, C, H, W));
                                break;
                            }
                        case AttentionType.ScaledDotProductAttention:
                            {
                                q = q.transpose(-2, -1); // [B, num_heads, N, key_dim]
                                k = k.transpose(-2, -1); // [B, num_heads, N, key_dim]
                                v = v.transpose(-2, -1); // [B, num_heads, N, head_dim]

                                torch.Tensor attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_casual: false);

                                attn_output = attn_output.transpose(-2, -1); // [B, num_heads, N, head_dim]
                                attn_output = attn_output.contiguous();

                                if (B * num_heads * N * head_dim != B * C * H * W)
                                {
                                    throw new InvalidOperationException("Shape mismatch: Cannot reshape attn_output to [B, C, H, W].");
                                }

                                attn_output = attn_output.view(B, C, H, W);
                                x = attn_output + pe.forward(attn_output);
                                break;
                            }
                        default:
                            {
                                throw new NotImplementedException($"Attention type {attentionType} is not implemented.");
                            }
                    }

                    x = proj.forward(x);

                    return x.MoveToOuterDisposeScope();
                }
            }
        }

        internal class SCDown : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly Convs.Conv cv1;
            private readonly Convs.Conv cv2;
            internal SCDown(int inChannel, int outChannel, int k, int s, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(SCDown))
            {
                cv1 = new Convs.Conv(inChannel, outChannel, 1, 1, device: device, dtype: dtype);
                cv2 = new Convs.Conv(outChannel, outChannel, k: k, s: s, g: outChannel, device: device, dtype: dtype);
                RegisterComponents();
            }

            public override torch.Tensor forward(torch.Tensor x)
            {
                return cv2.forward(cv1.forward(x));
            }
        }

        internal class C2fCIB : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly Convs.Conv cv1;
            private readonly Convs.Conv cv2;
            internal readonly Sequential m;
            internal C2fCIB(int inChannels, int outChannels, int n = 1, bool shortcut = false, bool lk = false, int g = 1, float e = 0.5f, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(C2fCIB))
            {
                int c = (int)(outChannels * e);
                cv1 = new Convs.Conv(inChannels, 2 * c, 1, 1, device: device, dtype: dtype);
                cv2 = new Convs.Conv((2 + n) * c, outChannels, 1, device: device, dtype: dtype);  // optional act=FReLU(outChannels)
                m = torch.nn.Sequential();
                for (int i = 0; i < n; i++)
                {
                    m = m.append(new CIB(c, c, shortcut, e: 1.0f, lk: lk, device: device, dtype: dtype));
                }
                RegisterComponents();
            }

            public override torch.Tensor forward(torch.Tensor input)
            {
                using (torch.NewDisposeScope())
                {
                    List<torch.Tensor> y = cv1.forward(input).chunk(2, 1).ToList();
                    for (int i = 0; i < m.Count; i++)
                    {
                        y.Add(m[i].call(y.Last()));
                    }
                    return cv2.forward(torch.cat(y, 1)).MoveToOuterDisposeScope();
                }
            }
        }

        internal class CIB : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly Sequential cv1;
            private readonly bool add;
            internal CIB(int inChannels, int outChannels, bool shortcut = true, float e = 0.5f, bool lk = false, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(CIB))
            {
                int c = (int)(outChannels * e);  // hidden channels
                cv1 = torch.nn.Sequential(
                    new Convs.Conv(inChannels, inChannels, 3, g: inChannels, device: device, dtype: dtype),
                    new Convs.Conv(inChannels, 2 * c, 1, device: device, dtype: dtype),
                    lk ? new RepVGGDW(2 * c, device: device, dtype: dtype) : new Convs.Conv(2 * c, 2 * c, 3, g: 2 * c, device: device, dtype: dtype),
                    new Convs.Conv(2 * c, outChannels, 1, device: device, dtype: dtype),
                    new Convs.Conv(outChannels, outChannels, 3, g: outChannels, device: device, dtype: dtype));
                add = shortcut && inChannels == outChannels;

                RegisterComponents();
            }

            public override torch.Tensor forward(torch.Tensor x)
            {
                return add ? x + cv1.forward(x) : cv1.forward(x);
            }
        }


        /// <summary>
        /// Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.
        /// This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
        /// processing.It supports both area-attention and standard convolution modes.
        /// </summary>
        internal class A2C2f : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            /// <summary>
            /// Initial 1x1 convolution layer that reduces x channels to hidden channels.
            /// </summary>
            private readonly Convs.Conv cv1;

            /// <summary>
            /// Final 1x1 convolution layer that processes concatenated features.
            /// </summary>
            private readonly Convs.Conv cv2;

            /// <summary>
            /// Learnable parameter for residual scaling when using area attention.
            /// </summary>
            private readonly Parameter? gamma;

            /// <summary>
            /// List of either ABlock or C3k modules for feature processing.
            /// </summary>
            private readonly Sequential m;

            /// <summary>
            /// Initialize Area-Attention C2f module.
            /// </summary>
            /// <param name="c1">Number of x channels.</param>
            /// <param name="c2">Number of output channels.</param>
            /// <param name="n">Number of ABlock or C3k modules to stack.</param>
            /// <param name="a2">Whether to use area attention blocks. If False, uses C3k blocks instead.</param>
            /// <param name="area">Number of areas the feature map is divided.</param>
            /// <param name="residual">Whether to use residual connections with learnable gamma parameter.</param>
            /// <param name="mlp_ratio">Expansion ratio for MLP hidden dimension.</param>
            /// <param name="e">Channel expansion ratio for hidden channels.</param>
            /// <param name="g">Number of groups for grouped convolutions.</param>
            /// <param name="shortcut">Whether to use shortcut connections in C3k blocks.</param>
            /// <param name="device"></param>
            /// <param name="dtype"></param>
            /// <exception cref="Exception"></exception>
            internal A2C2f(int c1, int c2, int n = 1, bool a2 = true, int area = 1, bool residual = false, float mlp_ratio = 2.0f, float e = 0.5f, int g = 1, bool shortcut = true, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(A2C2f))
            {
                int c_ = (int)(c2 * e);
                if (c_ % 32 != 0)
                {
                    throw new Exception("Dimension of ABlock be a multiple of 32.");
                }
                cv1 = new Convs.Conv(c1, c_, 1, 1, device: device, dtype: dtype);
                cv2 = new Convs.Conv((1 + n) * c_, c2, 1, device: device, dtype: dtype);

                gamma = a2 && residual ? torch.nn.Parameter(0.01 * torch.ones(c2, device: device, dtype: dtype), requires_grad: true) : null;
                m = torch.nn.Sequential();
                for (int i = 0; i < n; i++)
                {
                    if (a2)
                    {
                        var seq = torch.nn.Sequential();
                        for (int j = 0; j < 2; j++)
                        {
                            seq.append(new ABlock(c_, c_ / 32, mlp_ratio, area, device: device, dtype: dtype));
                        }
                        m.append(seq);
                    }
                    else
                    {
                        C3k c3k = new C3k(c_, c_, 2, shortcut, g, device: device, dtype: dtype);
                        m.append(c3k);
                    }
                }
                RegisterComponents();
            }

            public override torch.Tensor forward(torch.Tensor x)
            {
                using (torch.NewDisposeScope())
                {
                    List<torch.Tensor> y = new List<torch.Tensor> { cv1.forward(x) };

                    foreach (var module in m.children())
                    {
                        y.Add(((torch.nn.Module<torch.Tensor, torch.Tensor>)module).forward(y.Last()));
                    }

                    torch.Tensor y_cat = torch.cat(y.ToArray(), 1);
                    torch.Tensor output = cv2.forward(y_cat);

                    if (gamma is not null)
                    {
                        torch.Tensor gamma_view = gamma.view(new long[] { -1, gamma.shape[0], 1, 1 });
                        return (x + gamma_view * output).MoveToOuterDisposeScope();
                    }
                    return output.MoveToOuterDisposeScope();
                }
            }
        }

        /// <summary>
        /// Area-attention block module for efficient feature extraction in YOLO models.
        /// This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
        /// It uses a novel area-based attention approach that is more efficient than traditional self-attention while
        /// maintaining effectiveness
        /// </summary>
        internal class ABlock : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly AAttn attn;
            private readonly Sequential mlp;
            private readonly Action<torch.nn.Module> initWeights;  // Weight initialization function
            internal ABlock(int dim, int num_heads, float mlp_ratio = 1.2f, int area = 1, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(ABlock))
            {
                attn = new AAttn(dim, num_heads: num_heads, area: area, attentionType: AttentionType.SelfAttention, device: device, dtype: dtype);
                int mlp_hidden_dim = (int)(dim * mlp_ratio);
                mlp = torch.nn.Sequential(new Convs.Conv(dim, mlp_hidden_dim, 1, device: device, dtype: dtype), new Convs.Conv(mlp_hidden_dim, dim, 1, device: device, dtype: dtype));
                // Initialize weights
                initWeights = m =>
                {
                    if (m is Conv2d conv)
                    {
                        torch.nn.init.trunc_normal_(conv.weight, std: 0.02);
                        if (conv.bias is not null)
                            torch.nn.init.constant_(conv.bias, 0);
                    }
                };
                apply(initWeights);
                RegisterComponents();
            }

            public override torch.Tensor forward(torch.Tensor x)
            {
                x = x + attn.forward(x);
                return x + mlp.forward(x);
            }
        }


        /// <summary>
        /// Area-attention module for YOLO models, providing efficient attention mechanisms.
        /// This module implements an area-based attention mechanism that processes x features in a spatially-aware manner,
        /// making it particularly effective for object detection tasks.

        /// </summary>
        internal class AAttn : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly int area;
            private readonly int num_heads;
            private readonly int head_dim;

            private readonly Convs.Conv qkv;
            private readonly Convs.Conv proj;
            private readonly Convs.Conv pe;
            private readonly AttentionType attentionType;

            internal AAttn(int dim, int num_heads, int area = 1, AttentionType attentionType = AttentionType.SelfAttention, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(AAttn))
            {
                this.attentionType = attentionType;
                this.area = area;
                this.num_heads = num_heads;
                head_dim = dim / num_heads;
                int all_head_dim = head_dim * this.num_heads;

                qkv = new Convs.Conv(dim, all_head_dim * 3, 1, device: device, dtype: dtype);
                proj = new Convs.Conv(all_head_dim, dim, 1, device: device, dtype: dtype);
                pe = new Convs.Conv(all_head_dim, dim, 7, 1, 3, g: dim, bias: true, device: device, dtype: dtype);
                RegisterComponents();
            }

            public override torch.Tensor forward(torch.Tensor x)
            {
                using (torch.NewDisposeScope())
                {

                    long B = x.shape[0];
                    long C = x.shape[1];
                    long H = x.shape[2];
                    long W = x.shape[3];
                    long N = H * W;

                    torch.Tensor qkv = this.qkv.forward(x).flatten(2).transpose(1, 2);

                    if (area > 1)
                    {
                        qkv = qkv.reshape(B * area, N / area, C * 3);
                        B = qkv.shape[0];
                        N = qkv.shape[1];
                    }
                    torch.Tensor[] qkv_mix = qkv.view(B, N, num_heads, head_dim * 3).permute(0, 2, 3, 1).split(new long[] { head_dim, head_dim, head_dim }, dim: 2);
                    torch.Tensor q = qkv_mix[0];
                    torch.Tensor k = qkv_mix[1];
                    torch.Tensor v = qkv_mix[2];
                    if (attentionType == AttentionType.SelfAttention)
                    {
                        torch.Tensor attn = q.transpose(-2, -1).matmul(k) * (float)Math.Pow(head_dim, -0.5);
                        attn = attn.softmax(dim: -1);
                        x = v.matmul(attn.transpose(-2, -1));
                        x = x.permute(0, 3, 1, 2);
                        v = v.permute(0, 3, 1, 2);
                    }
                    else if (attentionType == AttentionType.ScaledDotProductAttention)
                    {
                        q = q.permute(0, 1, 3, 2); // [B, nh, N, hd]
                        k = k.permute(0, 1, 3, 2);
                        v = v.permute(0, 1, 3, 2);

                        x = torch.nn.functional.scaled_dot_product_attention(q, k, v);

                        x = x.permute(0, 2, 1, 3)  // [B, N, nh, hd]
                            .reshape(B, N, -1);     // [B, N, C]

                        v = v.permute(0, 2, 1, 3)  // [B, N, nh, hd]
                            .reshape(B, N, -1);     // [B, N, C]
                    }
                    else
                    {
                        throw new NotImplementedException($"Attention type {attentionType} is not implemented.");
                    }

                    if (area > 1)
                    {
                        x = x.reshape(B / area, N * area, C);
                        v = v.reshape(B / area, N * area, C);
                        B = x.shape[0];
                        N = x.shape[1];
                    }

                    x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous();
                    v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous();
                    x = x + pe.forward(v);
                    return proj.forward(x).MoveToOuterDisposeScope();
                }
            }
        }

        internal class RepVGGDW : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly Convs.Conv conv;
            private readonly Convs.Conv conv1;
            private readonly int dim;
            private readonly torch.nn.Module<torch.Tensor, torch.Tensor> act;
            internal RepVGGDW(int ed, torch.Device? device = null, torch.nn.Module<torch.Tensor, torch.Tensor>? act = null, torch.ScalarType? dtype = null) : base(nameof(RepVGGDW))
            {
                conv = new Convs.Conv(ed, ed, 7, 1, 3, g: ed, act: act, device: device, dtype: dtype);
                conv1 = new Convs.Conv(ed, ed, 3, 1, 1, g: ed, act: act, device: device, dtype: dtype);
                dim = ed;
                this.act = act ?? torch.nn.SiLU();

                RegisterComponents();
            }
            public override torch.Tensor forward(torch.Tensor x)
            {
                return act.forward(conv.forward(x) + conv1.forward(x));
            }
        }

    }
}
