using TorchSharp;
using TorchSharp.Modules;
using YoloSharp.Modules;

namespace YoloSharp.Models
{
    /// <summary>
    /// GELAN building blocks for YOLOv9 — contributed back to the open-source YoloSharp project.
    /// Structure verified 1:1 against official WongKinYiu/yolov9 models/common.py and real trained
    /// yolov9-s weights (1774 tensors, layer-by-layer channel match).
    /// </summary>
    internal static class Gelan
    {
        /// <summary>Conv + avg_pool2d(k=2) downsample (v9 AConv, common.py:60)</summary>
        internal class AConv : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            readonly Convs.Conv cv1;
            internal AConv(int c1, int c2, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(AConv))
            {
                cv1 = new Convs.Conv(c1, c2, 3, 2, 1, device: device, dtype: dtype);
                RegisterComponents();
            }
            public override torch.Tensor forward(torch.Tensor x)
            {
                using var pooled = torch.nn.functional.avg_pool2d(x, 2, 1, 0, false, true);
                return cv1.forward(pooled);
            }
        }

        /// <summary>ELAN1 block (common.py:580). c1=ch_in, c2=ch_out, c3=hidden, c4=branch width.
        /// forward: cv1→chunk2; [cv2,cv3]链; cat(2+2*c4)→cv4</summary>
        internal class ELAN1 : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            readonly Convs.Conv cv1, cv2, cv3, cv4;
            internal ELAN1(int c1, int c2, int c3, int c4, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(ELAN1))
            {
                cv1 = new Convs.Conv(c1, c3, 1, 1, device: device, dtype: dtype);
                cv2 = new Convs.Conv(c3 / 2, c4, 3, 1, device: device, dtype: dtype);
                cv3 = new Convs.Conv(c4, c4, 3, 1, device: device, dtype: dtype);
                cv4 = new Convs.Conv(c3 + 2 * c4, c2, 1, 1, device: device, dtype: dtype);
                RegisterComponents();
            }
            public override torch.Tensor forward(torch.Tensor x)
            {
                // 官方: y=list(cv1(x).chunk(2)); y.extend(m(y[-1]) for m in [cv2,cv3]); cv4(cat(y))
                // cat=[y0,y1,a,b]=c3/2+c3/2+c4+c4 = c3+2*c4（实测权重cv4[64,128]证实4块cat）
                var y = cv1.forward(x).chunk(2, 1);
                var a = cv2.forward(y[1]);
                var b = cv3.forward(a);
                return cv4.forward(torch.cat(new torch.Tensor[] { y[0], y[1], a, b }, 1));
            }
        }

        /// <summary>RepConvN (common.py:86): 3x3 branch + 1x1 branch (training mode, no fuse)</summary>
        internal class RepConvN : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            readonly Convs.Conv conv1, conv2;
            readonly torch.nn.Module<torch.Tensor, torch.Tensor> act;
            internal RepConvN(int c1, int c2, int k = 3, int s = 1, int p = 1, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(RepConvN))
            {
                conv1 = new Convs.Conv(c1, c2, k, s, p, device: device, dtype: dtype, act: torch.nn.Identity());
                conv2 = new Convs.Conv(c1, c2, 1, s, p - k / 2, device: device, dtype: dtype, act: torch.nn.Identity());
                act = torch.nn.SiLU();
                RegisterComponents();
            }
            public override torch.Tensor forward(torch.Tensor x)
                => act.forward(conv1.forward(x) + conv2.forward(x));
        }

        /// <summary>RepNBottleneck (common.py:299): RepConvN(3x3) + Conv(3x3) + optional shortcut</summary>
        internal class RepNBottleneck : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            readonly RepConvN cv1;
            readonly Convs.Conv cv2;
            readonly bool add;
            internal RepNBottleneck(int c1, int c2, bool shortcut = true, int g = 1, float e = 0.5f, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(RepNBottleneck))
            {
                int c_ = (int)(c2 * e);
                cv1 = new RepConvN(c1, c_, 3, 1, 1, device: device, dtype: dtype);
                cv2 = new Convs.Conv(c_, c2, 3, 1, 1, g: g, device: device, dtype: dtype);
                add = shortcut && c1 == c2;
                RegisterComponents();
            }
            public override torch.Tensor forward(torch.Tensor x)
            {
                var y = cv2.forward(cv1.forward(x));
                return add ? x + y : y;
            }
        }

        /// <summary>RepNCSP (common.py:373): two 1x1 reduce + RepNBottleneck chain + 1x1 merge</summary>
        internal class RepNCSP : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            readonly Convs.Conv cv1, cv2, cv3;
            readonly ModuleList<RepNBottleneck> m;
            internal RepNCSP(int c1, int c2, int n = 1, bool shortcut = true, float e = 0.5f, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(RepNCSP))
            {
                int c_ = (int)(c2 * e);
                cv1 = new Convs.Conv(c1, c_, 1, 1, device: device, dtype: dtype);
                cv2 = new Convs.Conv(c1, c_, 1, 1, device: device, dtype: dtype);
                cv3 = new Convs.Conv(2 * c_, c2, 1, device: device, dtype: dtype);
                m = new ModuleList<RepNBottleneck>();
                for (int i = 0; i < n; i++)
                    m.Add(new RepNBottleneck(c_, c_, shortcut, 1, 1.0f, device: device, dtype: dtype));
                RegisterComponents();
            }
            public override torch.Tensor forward(torch.Tensor x)
            {
                // 官方: cv3(cat(m(cv1(x)), cv2(x)))——m链吃cv1输出(c_)，cv2旁路
                var a = cv1.forward(x);
                foreach (var b in m) a = b.forward(a);
                return cv3.forward(torch.cat(new torch.Tensor[] { a, cv2.forward(x) }, 1));
            }
        }

        /// <summary>RepNCSPELAN4 (common.py:601, yaml: [out, hidden, branch, n]).
        /// cv1: c1→hidden(1x1)；chunk2 → 每半=hidden/2
        /// cv2/cv3: RepNCSP(hidden/2→branch, e=1.0) + Conv(branch→branch, 3x3)——权重名cv2.0/cv2.1
        /// cat = hidden/2 + branch + branch；cv4: cat→out
        /// 验证(层4, yaml[128,128,64,1]): cv1[128,128] cv2.0.cv1[32,64]=RepNCSP(64→32,e=1.0即c_=32)
        ///   cv2.1[64,64] cv4[128,256=64+64+128]✓</summary>
        internal class RepNCSPELAN4 : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            readonly Convs.Conv cv1, cv4;
            readonly Sequential cv2, cv3;
            internal RepNCSPELAN4(int c1, int c2, int c3, int c4, int n = 1, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(RepNCSPELAN4))
            {
                cv1 = new Convs.Conv(c1, c3, 1, 1, device: device, dtype: dtype);
                cv2 = torch.nn.Sequential(
                    new RepNCSP(c3 / 2, c4, n, e: 0.5f, device: device, dtype: dtype),
                    new Convs.Conv(c4, c4, 3, 1, 1, device: device, dtype: dtype));
                cv3 = torch.nn.Sequential(
                    new RepNCSP(c3 / 2, c4, n, e: 0.5f, device: device, dtype: dtype),
                    new Convs.Conv(c4, c4, 3, 1, 1, device: device, dtype: dtype));
                cv4 = new Convs.Conv(c3 + 2 * c4, c2, 1, 1, device: device, dtype: dtype);
                RegisterComponents();
            }
            public override torch.Tensor forward(torch.Tensor x)
            {
                // 官方: y=list(cv1(x).chunk(2)); y.extend(m(y[-1]) for m in [cv2,cv3]); cv4(cat(y))
                // → cat = [y0(c3/2), y1(c3/2), cv2(y1)(c4), cv3(cv2(y1))(c4)] = c3 + 2*c4
                var y = cv1.forward(x).chunk(2, 1);
                var a = cv2.forward(y[1]);
                var b = cv3.forward(a);
                return cv4.forward(torch.cat(new torch.Tensor[] { y[0], y[1], a, b }, 1));
            }
        }

        /// <summary>SPPELAN (common.py:563): 1x1 + MaxPool5 x3 chain + 1x1 merge(4*c3)</summary>
        internal class SPPELAN : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            readonly Convs.Conv cv1, cv5;
            readonly TorchSharp.Modules.MaxPool2d cv2, cv3, cv4;
            internal SPPELAN(int c1, int c2, int c3, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(SPPELAN))
            {
                cv1 = new Convs.Conv(c1, c3, 1, 1, device: device, dtype: dtype);
                cv2 = torch.nn.MaxPool2d(kernel_size: 5, stride: 1, padding: 2);
                cv3 = torch.nn.MaxPool2d(kernel_size: 5, stride: 1, padding: 2);
                cv4 = torch.nn.MaxPool2d(kernel_size: 5, stride: 1, padding: 2);
                cv5 = new Convs.Conv(4 * c3, c2, 1, 1, device: device, dtype: dtype);
                RegisterComponents();
            }
            public override torch.Tensor forward(torch.Tensor x)
            {
                var y0 = cv1.forward(x);
                var y1 = cv2.forward(y0);      // SP(5)
                var y2t = cv3.forward(y1);     // SP(5)
                var y3t = cv4.forward(y2t);    // SP(5)
                return cv5.forward(torch.cat(new torch.Tensor[] { y0, y1, y2t, y3t }, 1));
            }
        }

        /// <summary>ADown (common.py:70, gelan-c-seg降采样): avgpool→chunk2；
        /// 分支1=Conv3x3(s2)；分支2=maxpool3(s2)+Conv1x1；cat</summary>
        internal class ADown : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            readonly Convs.Conv cv1, cv2;
            internal ADown(int c1, int c2, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(ADown))
            {
                int c = c2 / 2;
                cv1 = new Convs.Conv(c1 / 2, c, 3, 2, 1, device: device, dtype: dtype);
                cv2 = new Convs.Conv(c1 / 2, c, 1, 1, 0, device: device, dtype: dtype);
                RegisterComponents();
            }
            public override torch.Tensor forward(torch.Tensor x)
            {
                using var pooled = torch.nn.functional.avg_pool2d(x, 2, 1, 0, false, true);
                var y = pooled.chunk(2, 1);
                var x1 = cv1.forward(y[0]);
                using var x2p = torch.nn.functional.max_pool2d(y[1], 3, 2, 1);
                var x2 = cv2.forward(x2p);
                return torch.cat(new torch.Tensor[] { x1, x2 }, 1);
            }
        }
    }
}
