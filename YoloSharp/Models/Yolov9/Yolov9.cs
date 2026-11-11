using TorchSharp;
using TorchSharp.Modules;
using YoloSharp.Modules;
using YoloSharp.Types;

namespace YoloSharp.Models
{
    internal partial class Yolo
    {
    /// <summary>
    /// YOLOv9 (GELAN) — contributed back to the open-source YoloSharp project.
    /// Architecture verified against official WongKinYiu/yolov9 (yolov9-s.yaml) and
    /// real trained weights (1774 tensors matched 1:1):
    ///   backbone: Conv x2, ELAN1, [AConv + RepNCSPELAN4] x3
    ///   head:     SPPELAN(9), up-concat RepNCSPELAN4(12,15), down-concat RepNCSPELAN4(18,21),
    ///             aux SPPELAN(22 from layer8), up-concat RepNCSPELAN4(25,28),
    ///             Detect(29) over [28,25,22,15,18,21] (nl=6, first 3 = aux train heads, last 3 = inference)
    /// Blocks: AConv / ELAN1 / RepNCSP / RepNBottleneck / RepConvN / SPPELAN implemented in Models/Gelan.cs.
    /// </summary>
    internal class Yolov9 : Yolo.Yolov8
    {
        // v9 topology layer cache: 4,6,8,9,12,15,18,21,22,25,28 (concat sources + Detect inputs), incl. aux branch / v9拓扑的层缓存（concat源+Detect输入），含aux支路
        protected override int[] outputIndexs => new int[] { 4, 6, 8, 9, 12, 15, 18, 21, 22, 25, 28 };
        // concat order (layers): 11(cat6) 14(cat4) 17(cat12) 20(cat9) 24(cat6) 27(cat4)
        // -> outputs index: 6->1, 4->0, 12->4, 9->3, 6->1, 4->0
        protected override int[] concatIndex => new int[] { 1, 0, 4, 3, 1, 0 };

        internal Yolov9(int nc = 80, int reg_max = 16, int? kpt_num = null, int? kpt_dim = null, YoloSize yoloSize = YoloSize.n, bool end2end = false, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nc, reg_max: reg_max, kpt_dim: kpt_dim, kpt_num: kpt_num, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
        {
        }

        /// <summary>v9 forward: aux branch (22) forks from layer 8 (yaml [8,1,SPPELAN]); Detect(29) inputs=[28,25,22,15,18,21] / v9 forward：aux支路(22)从层8分叉；Detect(29)输入=[28,25,22,15,18,21]</summary>
        public override (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds)? forward(torch.Tensor x)
        {
            (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds)? result = null;
            var outputs = new List<torch.Tensor>();
            int catCount = 0;
            // outputs collection order (outputIndexs): 4,6,8,9,12,15,18,21,22,25,28 -> indices 0..10
            // Detect inputs: [28(10), 25(9), 22(8), 15(5), 18(6), 21(7)]
            int[] detectSrc = { 10, 9, 8, 5, 6, 7 };
            for (int i = 0; i < model.Count; i++)
            {
                // aux branch fork: layer 22 (SPPELAN) input = layer 8 output (outputs index 2), not the sequential x / aux支路分叉：层22输入=层8输出，不是顺序流x
                if (i == 22)
                {
                    x = outputs[2];   // layer 8
                }
                switch (model[i])
                {
                    case torch.nn.Module<torch.Tensor, torch.Tensor> mod:
                        x = mod.forward(x);
                        break;
                    case Convs.Concat cat:
                        x = cat.forward(new torch.Tensor[] { x, outputs[concatIndex[catCount]] });
                        catCount++;
                        break;
                    case Head.DetectV9 detect:
                        result = detect.forward(detectSrc.Select(idx => outputs[idx]).ToArray());
                        break;
                    default:
                        throw new Exception("Yolov9 forward: unhandled module " + model[i].GetType().Name);
                }
                if (outputIndexs.Contains(i))
                    outputs.Add(x);
            }
            return result;
        }

        internal override ModuleList<torch.nn.Module> BuildModel()
        {
            // yolov9-s.yaml: depth/width multiple=1.0 (official v9 t/c/s/e tiers; C# ships the s baseline
            // first, 1:1 verified against real 1774-tensor weights; other tiers await community channel tables) / s基准已实测1:1吻合
            // RepNCSPELAN4 (yaml [out,hidden,branch,n]) exact channel table:
            //   L4=[128,128,64,1] 6=[192,192,96,1] 8/21=[256,256,128,1] 12/18/25=[192,192,96,1] 15/28=[128,128,64,1]
            int c2 = 128, c3 = 192, c4 = 256;

            ModuleList<torch.nn.Module> mod = new ModuleList<torch.nn.Module>(
                // ── backbone (gelan-s.yaml) ──
                new Convs.Conv(3, 32, 3, 2, device: device, dtype: dtype),                        // 0-P1/2
                new Convs.Conv(32, 64, 3, 2, device: device, dtype: dtype),                       // 1-P2/4
                new Gelan.ELAN1(64, 64, 64, 32, device: device, dtype: dtype),                    // 2 [64,64,32]
                new Gelan.AConv(64, c2, device: device, dtype: dtype),                            // 3-P3/8
                new Gelan.RepNCSPELAN4(c2, 128, 128, 64, 3, device: device, dtype: dtype),             // 4 [128,128,64,1]
                new Gelan.AConv(c2, c3, device: device, dtype: dtype),                            // 5-P4/16
                new Gelan.RepNCSPELAN4(c3, 192, 192, 96, 3, device: device, dtype: dtype),             // 6 [192,192,96,1]
                new Gelan.AConv(c3, c4, device: device, dtype: dtype),                            // 7-P5/32
                new Gelan.RepNCSPELAN4(c4, 256, 256, 128, 3, device: device, dtype: dtype),            // 8 [256,256,128,1]

                // ── head: main branch ──
                new Gelan.SPPELAN(c4, 256, 128, device: device, dtype: dtype),                    // 9 [256,128]
                torch.nn.Upsample(scale_factor: new double[] { 2, 2 }, mode: torch.UpsampleMode.Nearest),  // 10
                new Convs.Concat(),                                                                // 11 cat(6)
                new Gelan.RepNCSPELAN4(c4 + c3, 192, 192, 96, 3, device: device, dtype: dtype),         // 12
                torch.nn.Upsample(scale_factor: new double[] { 2, 2 }, mode: torch.UpsampleMode.Nearest),  // 13
                new Convs.Concat(),                                                                // 14 cat(4)
                new Gelan.RepNCSPELAN4(c3 + c2, 128, 128, 64, 3, device: device, dtype: dtype),         // 15

                // ── head: down path ──
                new Gelan.AConv(128, 96, device: device, dtype: dtype),                           // 16
                new Convs.Concat(),                                                                // 17 cat(12)
                new Gelan.RepNCSPELAN4(c3 + 96, 192, 192, 96, 3, device: device, dtype: dtype),        // 18 (P4/16-medium)
                new Gelan.AConv(192, c2, device: device, dtype: dtype),                           // 19
                new Convs.Concat(),                                                                // 20 cat(9)
                new Gelan.RepNCSPELAN4(c4 + c2, 256, 256, 128, 3, device: device, dtype: dtype),       // 21 (P5/32-large)

                // -- head: aux branch (ignored at inference, carries weights only) / head: aux分支（推理忽略，仅承载权重）--
                new Gelan.SPPELAN(c4, 256, 128, device: device, dtype: dtype),                    // 22 (from 8)
                torch.nn.Upsample(scale_factor: new double[] { 2, 2 }, mode: torch.UpsampleMode.Nearest),  // 23
                new Convs.Concat(),                                                                // 24 cat(6)
                new Gelan.RepNCSPELAN4(c4 + c3, 192, 192, 96, 3, device: device, dtype: dtype),         // 25
                torch.nn.Upsample(scale_factor: new double[] { 2, 2 }, mode: torch.UpsampleMode.Nearest),  // 26
                new Convs.Concat(),                                                                // 27 cat(4)
                new Gelan.RepNCSPELAN4(c3 + c2, 128, 128, 64, 3, device: device, dtype: dtype),         // 28

                // -- Detect(29): v9 head 6 branches [aux(28,25,22), main(15,18,21)], inference uses the last 3 / 推理用后3 --
                new Head.DetectV9(nc, new int[] { 128, 192, 256, 128, 192, 256 }, reg_max: 16, device: device, dtype: dtype)
            );
            return mod;
        }
    }

        /// <summary>
        /// YOLOv9-Segment (gelan-c-seg.yaml, 23 layers, no aux branch) — contributed back to the open-source YoloSharp project.
        /// Differences vs detect: downsample uses ADown (not AConv); layer 2 is RepNCSPELAN4 (not ELAN1); head = standard Segment (3-branch cv2/cv3/cv4 + proto). / 与detect版差异：降采样用ADown；层2为RepNCSPELAN4；头=标准Segment。
        /// Channel table calibrated against real weights (Dalian Chongda circle segmentation, 1000 tensors): / 通道表已对照真实权重校准：
        ///   backbone 64/128/256/512/512；RepNCSPELAN4: L2=[256,128,64] L4=[512,256,128] L6/8=[512,512,256]
        ///   L12/18/21=[512,512,256] L15=[256,256,128]; Segment head c2=64 c3=256(nc base) c4=64 nm=32 proto npr=256
        /// </summary>
        internal class Yolov9Segment : Yolov9
        {
            internal Yolov9Segment(int nc = 80, int reg_max = 16, YoloSize yoloSize = YoloSize.s, bool end2end = false, torch.Device? device = null, torch.ScalarType? dtype = null)
                : base(nc, reg_max: reg_max, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
            {
            }

            internal override ModuleList<torch.nn.Module> BuildModel()
        {
            // gelan-c-seg.yaml channels (calibrated layer-by-layer against Chongda 1000-tensor weights; yaml [out,hidden,branch,n] all n=1) / 通道对照权重逐层校准
            ModuleList<torch.nn.Module> mod = new ModuleList<torch.nn.Module>(
                // ── backbone ──
                new Convs.Conv(3, 64, 3, 2, device: device, dtype: dtype),                          // 0-P1/2
                new Convs.Conv(64, 128, 3, 2, device: device, dtype: dtype),                         // 1-P2/4
                new Gelan.RepNCSPELAN4(128, 256, 128, 64, 1, device: device, dtype: dtype),          // 2 [256,128,64,1]
                new Gelan.ADown(256, 256, device: device, dtype: dtype),                             // 3-P3/8
                new Gelan.RepNCSPELAN4(256, 512, 256, 128, 1, device: device, dtype: dtype),         // 4 [512,256,128,1]
                new Gelan.ADown(512, 512, device: device, dtype: dtype),                             // 5-P4/16
                new Gelan.RepNCSPELAN4(512, 512, 512, 256, 1, device: device, dtype: dtype),         // 6 [512,512,256,1]
                new Gelan.ADown(512, 512, device: device, dtype: dtype),                             // 7-P5/32
                new Gelan.RepNCSPELAN4(512, 512, 512, 256, 1, device: device, dtype: dtype),         // 8 [512,512,256,1]

                // ── head ──
                new Gelan.SPPELAN(512, 512, 256, device: device, dtype: dtype),                      // 9 [512,256]
                torch.nn.Upsample(scale_factor: new double[] { 2, 2 }, mode: torch.UpsampleMode.Nearest),  // 10
                new Convs.Concat(),                                                                    // 11 cat(6): 512+512
                new Gelan.RepNCSPELAN4(1024, 512, 512, 256, 1, device: device, dtype: dtype),        // 12 [512,512,256,1]
                torch.nn.Upsample(scale_factor: new double[] { 2, 2 }, mode: torch.UpsampleMode.Nearest),  // 13
                new Convs.Concat(),                                                                    // 14 cat(4): 512+512
                new Gelan.RepNCSPELAN4(1024, 256, 256, 128, 1, device: device, dtype: dtype),        // 15 [256,256,128,1]

                new Gelan.ADown(256, 256, device: device, dtype: dtype),                              // 16
                new Convs.Concat(),                                                                    // 17 cat(12): 256+512
                new Gelan.RepNCSPELAN4(768, 512, 512, 256, 1, device: device, dtype: dtype),          // 18 (P4/16)
                new Gelan.ADown(512, 512, device: device, dtype: dtype),                               // 19
                new Convs.Concat(),                                                                    // 20 cat(9): 512+512
                new Gelan.RepNCSPELAN4(1024, 512, 512, 256, 1, device: device, dtype: dtype),         // 21 (P5/32)

                // -- Segment head (layer 22): 3 branches [15,18,21] ch=[256,512,512], measured cv2=64/cv3=256/cv4=64/nm=32/proto npr=256 --
                new Head.Segment(nc: nc, nm: 32, npr: 256, reg_max: reg_max, ch: new int[] { 256, 512, 512 }, legacy: true, end2end: end2end, device: device, dtype: dtype)
            );
            return mod;
        }

            /// <summary>seg forward: 23-layer sequential flow (no aux fork); Segment at layer 22, inputs=[15,18,21] (last 3 of outputs) / seg版forward：23层顺序流；Segment在层22</summary>
            public override (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds)? forward(torch.Tensor x)
            {
                (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds)? result = null;
                var outputs = new List<torch.Tensor>();
                int catCount = 0;
                // cached layers (concat sources + Segment inputs): 4,6,9,12,15,18,21 -> indices 0..6
                // concat sources: 11(cat6=1) 14(cat4=0) 17(cat12=3) 20(cat9=2)
                int[] catSrc = { 1, 0, 3, 2 };
                for (int i = 0; i < model.Count; i++)
                {
                    switch (model[i])
                    {
                        case torch.nn.Module<torch.Tensor, torch.Tensor> m:
                            x = m.forward(x);
                            break;
                        case Convs.Concat cat:
                            x = cat.forward(new torch.Tensor[] { x, outputs[catSrc[catCount]] });
                            catCount++;
                            break;
                        case Head.Segment seg:
                            result = seg.forward(new torch.Tensor[] { outputs[4], outputs[5], outputs[6] });  // 15,18,21
                            break;
                        default:
                            throw new Exception("Yolov9Segment forward: unhandled " + model[i].GetType().Name);
                    }
                    if (i == 4 || i == 6 || i == 9 || i == 12 || i == 15 || i == 18 || i == 21)
                        outputs.Add(x);
                }
                return result;
            }
        }
}
}
