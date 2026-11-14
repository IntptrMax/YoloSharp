// Module processors: each YoloSharp module type → ONNX op sequence.
// Module processors: each YoloSharp module type -> an ONNX operator sequence / 模块处理器
// Reads module fields via reflection (conv/bn/cv1/m...), read-only; BN fused into Conv weights (inference-equivalent).
// Reads module fields (conv/bn/cv1/m...) via reflection, read-only; BN fused into Conv weights (inference-equivalent) / 反射只读+BN融合
using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

using Convs = YoloSharp.Modules.Convs;
using Block = YoloSharp.Modules.Block;
using Head = YoloSharp.Modules.Head;

namespace YoloSharp.OnnxExport
{
    internal static class Processors
    {
        // -- basic building blocks / 基础件 --

        /// <summary>Conv module (conv+bn+act) → fused Conv node. act supports only SiLU/Identity (v8/v9/v11/v12 defaults) / act仅支持SiLU/Identity（全家默认）</summary>
        public static string Conv(ExportContext c, Convs.Conv mod, string input, string hint)
        {
            var (w, b) = FuseConvBn(mod.conv, mod.bn);
            var outN = Ops.Conv(c, input, c.Temp(hint + "_w"), w, b,
                kernel: new long[] { (long)mod.conv.kernel_size[0], (long)mod.conv.kernel_size[1] },
                strides: new long[] { (long)mod.conv.stride[0], (long)mod.conv.stride[1] },
                pads: new long[] { (long)mod.conv.padding[0], (long)mod.conv.padding[1], (long)mod.conv.padding[0], (long)mod.conv.padding[1] },
                dilations: new long[] { (long)mod.conv.dilation[0], (long)mod.conv.dilation[1] },
                group: (long)mod.conv.groups, hint: hint);
            // act dispatch: SiLU expanded / ReLU->Relu node / Identity passthrough / act分派
            var act = mod.get_field("act");
            var an = act?.GetType().Name;
            if (an == "SiLU") return Ops.SiLU(c, outN);
            if (an == "ReLU" || an == "ReLU6")
            {
                var rN = c.Temp("relu");
                c.Node("Relu", rN, outN);
                return rN;
            }
            return outN;   // Identity / other linear / Identity其他线性
        }

        /// <summary>Conv+BN weight fusion (inference-equivalent): w' = w*γ/√(var+ε); b' = (b-μ)*γ/√(var+ε)+β / Conv+BN权重融合（推理等价）</summary>
        public static (torch.Tensor w, torch.Tensor b) FuseConvBn(Conv2d conv, BatchNorm2d bn)
        {
            using var noGrad = torch.no_grad();
            var w = conv.weight.detach().to_type(ScalarType.Float32);
            var g = bn.weight.detach().to_type(ScalarType.Float32);
            var beta = bn.bias.detach().to_type(ScalarType.Float32);
            var mu = bn.running_mean.detach().to_type(ScalarType.Float32);
            var var_ = bn.running_var.detach().to_type(ScalarType.Float32);
            float eps = (float)bn.eps;
            var denom = (var_ + eps).sqrt();
            var scale = (g / denom).reshape(-1, 1, 1, 1);   // scale by w's out-channel (dim 0) / 按w输出通道缩放通道(dim0)缩放
            var wF = w * scale;
            var b0 = conv.bias is null ? torch.zeros_like(mu) : conv.bias.detach().to_type(ScalarType.Float32);
            var bF = (b0 - mu) * g / denom + beta;
            return (wF.clone(), bF.clone());
        }

        /// <summary>Bottleneck: cv1→cv2→(optional residual +x) / Bottleneck：cv1→cv2→（可选残差+x）</summary>
        public static string Bottleneck(ExportContext c, Block.Bottleneck mod, string input)
        {
            var cv1 = (Convs.Conv)mod.get_field("cv1");
            var cv2 = (Convs.Conv)mod.get_field("cv2");
            var y = Conv(c, cv2, Conv(c, cv1, input, "bt1"), "bt2");
            var addF = typeof(Block.Bottleneck).GetField("add", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public);
            if (addF != null && (bool)addF.GetValue(mod)) return Ops.Add(c, input, y);
            return y;
        }

        /// <summary>C2f: cv1→chunk2→(a + n×Bottleneck(b))→cat(a,b...)→cv2</summary>
        public static string C2f(ExportContext c, Block.C2f mod, string input)
        {
            // Source semantics: y=chunk2(cv1)→loop y.Add(bottleneck(y.Last))→cat(y)→cv2
            // Source semantics: y=chunk2(cv1) -> loop y.Add(bottleneck(y.Last)) -> cat(y) -> cv2 / 源码语义
            // Segment count = 2+n, each channel = c (hidden channel field c, not out/2) / 段数2+n通道各段c
            var cv1 = (Convs.Conv)mod.get_field("cv1");
            var cv2 = (Convs.Conv)mod.get_field("cv2");
            var m = EnumModules(mod.get_field("m")).Cast<Block.Bottleneck>();
            int hidden = (int)cv1.conv.out_channels / 2;   // = c (half of 2*c output) / 即c
            var cv1Out = Conv(c, cv1, input, "c2f1");
            var sp = Ops.Split(c, cv1Out, new long[] { hidden, hidden });
            var y = new List<string> { sp[0], sp[1] };
            foreach (var blk in m) y.Add(Bottleneck(c, blk, y[y.Count - 1]));
            var cat = Ops.Concat(c, y.ToArray(), 1);
            return Conv(c, cv2, cat, "c2f2");
        }

        /// <summary>SPPF: cv1(Identity act)→MaxPool×n serial→cat(x,p1..pn)→cv2</summary>
        public static string SPPF(ExportContext c, Block.SPPF mod, string input)
        {
            // cv1 act=Identity: handled separately (no SiLU) / cv1的Identity单独处理
            var cv1 = (Convs.Conv)mod.get_field("cv1");
            var (w1, b1) = FuseConvBn(cv1.conv, cv1.bn);
            var cv1Out = Ops.Conv(c, input, c.Temp("sppf1_w"), w1, b1,
                new long[] { 1, 1 }, new long[] { 1, 1 }, new long[] { 0, 0, 0, 0 }, new long[] { 1, 1 }, 1, "sppf1");
            // SPPF: a single MaxPool2d(k) used n times serially — n = cv2.in_channels/cv1.out_channels - 1
            // SPPF: one MaxPool2d(k) applied serially n times — n = cv2.in/cv1.out - 1 / SPPF串行池化
            var poolField = typeof(Block.SPPF).GetField("m",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var pool = (MaxPool2d)poolField.GetValue(mod);
            long kk = (long)pool.kernel_size[0];
            int cv1OutCh = (int)cv1.conv.out_channels;
            var cv2 = (Convs.Conv)mod.get_field("cv2");
            int n = (int)(cv2.conv.in_channels / cv1OutCh) - 1;   // pool count / 池化次数
            var current = cv1Out;
            var outs = new List<string> { cv1Out };
            for (int i = 0; i < n; i++)
            {
                current = Ops.MaxPool(c, current,
                    new long[] { kk, kk }, new long[] { 1, 1 },
                    new long[] { kk / 2, kk / 2, kk / 2, kk / 2 });
                outs.Add(current);
            }
            var cat = Ops.Concat(c, outs.ToArray(), 1);
            return Conv(c, cv2, cat, "sppf2");
        }

        /// <summary>Detect head: box_head/cls_head per-level Conv(1x1)→view(bs,C,-1)→cat(3 levels)
        /// [Fix 2026-11-14 author-reported] Boxes are exported DECODED (ultralytics layout): DFL
        /// softmax+expectation and grid+stride are baked into the graph — output0 = [bs, 4+nc(+extras), A]
        /// (4 = decoded xyxy in pixels), matching python's export instead of the raw 64-ch DFL.
        /// / boxes导出改为解码版（python布局）：DFL解码+网格烘焙进图
        /// </summary>
        public static void Detect(ExportContext c, Head.Detect mod, string[] feats, Dictionary<string, string> outputs, int headIndexBase = 0)
        {
            var allBox = ((TorchSharp.Modules.ModuleList<TorchSharp.Modules.Sequential>)mod.get_field("cv2")).ToArray();
            var allCls = ((TorchSharp.Modules.ModuleList<TorchSharp.Modules.Sequential>)mod.get_field("cv3")).ToArray();
            // v9 (DetectV9): cv2/cv3 have 6 groups, inference uses indices 3..5 (main group) — headIndexBase=3
            // v9 (DetectV9): cv2/cv3 have 6 groups; inference uses indices 3..5 (main group) — headIndexBase=3 / v9六组取main组
            var boxHeads = allBox.Skip(headIndexBase).ToArray();
            var clsHeads = allCls.Skip(headIndexBase).ToArray();
            int nl = feats.Length;
            // [Fix 2026-11-11 19:58] Reshape bs dim uses -1 (dynamic) — batch propagates from the input:
            // with static input TRT/OV infer the real batch (equivalent to the old hardcode); with dynamic
            // input the [N,C,A] frame-contiguous layout is preserved (the old bs=1 hardcode folded dynamic
            // batch into the anchor dim, mis-slicing frames — trt4test measured 38/19 fluctuation) / bs维用-1动态
            long bs = -1;
            long regMax = 16;   // DFL bins per edge / 每边DFL bins
            var dflF = mod.get_field("dfl");
            if (dflF is Block.DFL dflMod)
            {
                // DFL's conv is Conv2d(c1, 1, 1): bins = IN channels (c1), NOT the 1x1 kernel size.
                // / DFL的conv是1x1卷积：bins=输入通道c1，不是kernel尺寸
                var dflConv = dflMod.get_field("conv") as Conv2d;
                if (dflConv != null && dflConv.in_channels > 1) regMax = dflConv.in_channels;
            }
            var boxViews = new List<string>();
            var clsViews = new List<string>();
            var strides = new List<long>();
            var levelHW = new List<long>();   // per-level H*W (concrete) / 每级H*W
            for (int i = 0; i < nl; i++)
            {
                // box branch head unit = Sequential(Conv,Conv,Conv2d) — emitted via SequentialDown / box分支头单元单元经SequentialDown发射
                var bOut = HeadConv(c, boxHeads[i], feats[i], $"bx{i}");
                var cOut = HeadConv(c, clsHeads[i], feats[i], $"cs{i}");
                long hw = HWOf(c, bOut);
                levelHW.Add(hw);
                // view(bs, C, -1): NCHW -> (bs,C,H*W) — via Reshape / NCHW转三维
                long bC = (long)GetHeadOutChannels(boxHeads[i]);
                long cC = (long)GetHeadOutChannels(clsHeads[i]);
                boxViews.Add(Ops.Reshape(c, bOut, new long[] { bs, bC, hw }));
                clsViews.Add(Ops.Reshape(c, cOut, new long[] { bs, cC, hw }));
                // stride from the feature/input size ratio (input dims are concrete) / 由特征图与输入尺寸求stride
                var fSh = c.ShapeOf(feats[i]);
                var inSh = c.ShapeOf("images");
                strides.Add(fSh != null && inSh != null && inSh.Length == 4 && fSh[3] > 0 && inSh[3] > 0 ? inSh[3] / fSh[3] : (long)(1L << (i + 3)));
            }
            // ── In-graph DFL decode (ultralytics layout) ──
            // boxes concat: [bs, 4*regMax, A] → reshape [bs, 4, regMax, A] → softmax(axis=2) →
            // MatMul(arange[regMax]) → [bs, 4, A] expected ltrb distances → grid+stride → xyxy pixels.
            // Level-wise decode keeps softmax bins per-level contiguous (A = ΣH·W per level order).
            string rawBoxes = Ops.Concat(c, boxViews.ToArray(), 2);
            string decoded;
            long totalA = 0;
            foreach (var hw in levelHW) totalA += hw;
            {
                // [bs, 64, A] → [bs, 4, 16, A]
                var reshaped = Ops.Reshape(c, rawBoxes, new long[] { bs, 4, regMax, totalA });
                var sm = Ops.Softmax(c, reshaped, axis: 2, outShape: new long[] { bs, 4, regMax, totalA });
                // expectation over bins: softmax × arange → ReduceSum(axis=2) → [bs,4,1,A] → [bs,4,A]
                // (NOT MatMul: trailing-dims matrix product can't broadcast the anchor dim) / 期望用乘加而非MatMul
                float[] proj = new float[regMax];
                for (int k = 0; k < regMax; k++) proj[k] = k;
                var projT = Ops.ConstTensor(c, proj, new long[] { 1, 1, regMax, 1 }, "dfp");
                var weighted = Ops.MulT(c, sm, projT, new long[] { bs, 4, regMax, totalA });   // broadcast bins
                var rsName = c.Temp("drs");
                // opset13+: ReduceSum takes axes as an INPUT initializer, not an attribute
                // opset13+：axes是输入initializer不是属性
                var axesI = c.AddInitializer(c.Temp("dax"), torch.tensor(new long[] { 2 }, ScalarType.Int64));
                var rsNode = c.Node("ReduceSum", rsName, weighted, axesI);
                rsNode.AttrInts.Add(("keepdims", 1));
                c.SetShape(rsName, new long[] { bs, 4, 1, totalA });
                var dist = Ops.Reshape(c, rsName, new long[] { bs, 4, totalA });
                // anchors/strides per anchor (constant, concrete A): [1, 2, A] grid centers (GRID units,
                // NOT pre-multiplied — the ×stride happens once at the end, like python's make_anchors
                // + dist2bbox×strides) + [1,1,A] stride. / 网格坐标不预乘stride（末尾统一乘）
                var anchorX = new List<float>(); var anchorY = new List<float>(); var strideArr = new List<float>();
                for (int lv = 0; lv < levelHW.Count; lv++)
                {
                    long hwi = levelHW[lv]; long s = strides[lv];
                    long w = fShOf(c, feats, lv, 3); long h = hwi / (w > 0 ? w : 1);
                    for (long y = 0; y < h; y++)
                        for (long x = 0; x < w; x++)
                        {
                            anchorX.Add((x + 0.5f) * 1.0f);   // grid units / 格坐标
                            anchorY.Add((y + 0.5f) * 1.0f);
                            strideArr.Add(s);
                        }
                }
                var axT = Ops.ConstTensor(c, anchorX.ToArray(), new long[] { 1, 1, totalA }, "ax");
                var ayT = Ops.ConstTensor(c, anchorY.ToArray(), new long[] { 1, 1, totalA }, "ay");
                var stT = Ops.ConstTensor(c, strideArr.ToArray(), new long[] { 1, 1, totalA }, "st");
                // dist [bs,4,A]: rows 0/1 = lt (anchor − d), rows 2/3 = rb (anchor + d) → xyxy in stride units → ×stride
                // Split rows: gather via Slice on axis 1 (0:2 lt, 2:4 rb)
                var lt = Slice(c, rawT: dist, starts: new long[] { 0 }, ends: new long[] { 2 }, axes: new long[] { 1 }, outSh: new long[] { bs, 2, totalA });
                var rb = Slice(c, rawT: dist, starts: new long[] { 2 }, ends: new long[] { 4 }, axes: new long[] { 1 }, outSh: new long[] { bs, 2, totalA });
                // anchor points stacked [1, 2, A] (x row, y row)
                var anchorT = ConcatConst2(c, axT, ayT, totalA);   // [1,2,A]
                var x1y1 = Ops.SubT(c, anchorT, lt, new long[] { bs, 2, totalA });   // broadcast [1,2,A] − [bs,2,A]
                var x2y2 = Ops.AddT(c, anchorT, rb, new long[] { bs, 2, totalA });
                var xyxy = Ops.Concat(c, new[] { x1y1, x2y2 }, 1);   // [bs,4,A] in stride units
                decoded = Ops.MulT(c, xyxy, stT, new long[] { bs, 4, totalA });   // → pixels
                c.SetShape(decoded, new long[] { bs, 4, totalA });
            }
            outputs["boxes"] = decoded;
            // [Fix 2026-11-02] Apply sigmoid to exported scores (aligns with libtorch inference
            // `cat(dbox, scores.sigmoid())` in Head.Detect.forward). The raw cls-branch logits were
            // exported before, so every runtime consumer (OnnxPredictCore/TxmLabeling/OV/TRT) read
            // logits as probabilities and deployment scores were systematically deflated
            // (observed 0.78 vs 0.9+ on the same weights).
            // Export scores with sigmoid (aligns with the libtorch chain's scores.sigmoid()): the previous
            // export emitted raw cls logits, which runtimes read as probabilities, deflating deployment scores. / scores加sigmoid对齐推理链
            outputs["scores"] = Ops.Sigmoid(c, Ops.Concat(c, clsViews.ToArray(), 2));
        }

        /// <summary>Concrete dim of feats[lv] (0-based from the last: 2=H,3=W in NCHW) — 0 when unknown / 特征图具体维</summary>
        static long fShOf(ExportContext c, string[] feats, int lv, int dim)
        {
            var sh = c.ShapeOf(feats[lv]);
            if (sh != null && sh.Length == 4 && sh[dim] > 0) return sh[dim];
            return 0;
        }

        /// <summary>Slice helper on a tensor dim — [Add 2026-11-14] DFL lt/rb row split / 维度切片</summary>
        static string Slice(ExportContext c, string rawT, long[] starts, long[] ends, long[] axes, long[] outSh)
        {
            var outN = c.Temp("sl");
            var startsI = c.AddInitializer(c.Temp("sls"), torch.tensor(starts, ScalarType.Int64));
            var endsI = c.AddInitializer(c.Temp("sle"), torch.tensor(ends, ScalarType.Int64));
            var axesI = c.AddInitializer(c.Temp("sla"), torch.tensor(axes, ScalarType.Int64));
            var n = c.Node("Slice", outN, rawT, startsI, endsI, axesI);
            if (outSh != null) c.SetShape(outN, outSh);
            return outN;
        }

        /// <summary>Concat two [1,k,A] constants along dim1 → [1,2k,A] — [Add 2026-11-14] anchor stack / 常量拼接</summary>
        static string ConcatConst2(ExportContext c, string a, string b, long A)
        {
            var outN = c.Temp("acat");
            var n = c.Node("Concat", outN, a, b);
            n.AttrInts.Add(("axis", 1));
            c.SetShape(outN, new long[] { 1, 2, A });
            return outN;
        }

        /// <summary>Feature-map H*W (from NCHW shape propagation dims [2]*[3]; falls back to -1 when unknown) — the Detect-head Reshape anchor dim / 特征图H*W</summary>
        static long HWOf(ExportContext c, string tensor)
        {
            var sh = c.ShapeOf(tensor);
            if (sh != null && sh.Length == 4 && sh[2] > 0 && sh[3] > 0) return sh[2] * sh[3];
            return -1;   // fall back to -1 when shape unknown / 未知时退-1
        }

        /// <summary>Sequential in head (Conv or plain Conv2d) → single Conv node (with optional BN fusion) / head里的Sequential→单Conv节点（含可选BN融合）</summary>
        static string HeadConv(ExportContext c, torch.nn.Module mod, string input, string hint)
        {
            // Detect head unit = Sequential(Conv,Conv,Conv2d) — emit per child: Conv with fusion, last layer plain Conv2d
            // Detect head unit = Sequential(Conv,Conv,Conv2d) — emit per child: Conv with fusion, last plain Conv2d / Detect头单元逐子级Emit
            return Processors.SequentialDown(c, mod, input);
        }

        static string HeadConvUnused(ExportContext c, torch.nn.Module mod, string input, string hint)
        {
            var conv = (Conv2d)mod.get_field("conv");
            BatchNorm2d bn = null;
            try { bn = (BatchNorm2d)mod.get_field("bn"); } catch { }
            torch.Tensor w, b;
            if (bn != null) { (w, b) = FuseConvBn(conv, bn); }
            else
            {
                w = conv.weight.detach().to_type(ScalarType.Float32);
                b = conv.bias is null ? null : conv.bias.detach().to_type(ScalarType.Float32);
            }
            return Ops.Conv(c, input, c.Temp(hint + "_w"), w, b,
                new long[] { (long)conv.kernel_size[0], (long)conv.kernel_size[1] },
                new long[] { (long)conv.stride[0], (long)conv.stride[1] },
                new long[] { (long)conv.padding[0], (long)conv.padding[1], (long)conv.padding[0], (long)conv.padding[1] },
                new long[] { 1, 1 }, (long)conv.groups, hint);
        }

        static long GetHeadOutChannels(torch.nn.Module seq)
        {
            // out_channels of the last child (plain Conv2d) / 末子级的out_channels
            foreach (var (name, child) in seq.named_children().Reverse())
                if (child is Conv2d cv) return (long)cv.out_channels;
            throw new NotSupportedException("head unit last child is not Conv2d / head单元末级非Conv2d");
        }

        // -- other backbone modules / 其余骨干模块 --

        /// <summary>C2: cv1→chunk2→Bottleneck×n on a→cat(a,y)→cv2</summary>
        public static string C2(ExportContext c, Block.C2 mod, string input)
        {
            var cv1 = (Convs.Conv)mod.get_field("cv1");
            var cv2 = (Convs.Conv)mod.get_field("cv2");
            var m = EnumModules(mod.get_field("m")).Cast<Block.Bottleneck>();
            int half = (int)cv1.conv.out_channels / 2;
            var cv1Out = Conv(c, cv1, input, "c2_1");
            var sp = Ops.Split(c, cv1Out, new long[] { half, half });
            string y = sp[1];
            foreach (var blk in m) y = Bottleneck(c, blk, y);
            var cat = Ops.Concat(c, new[] { sp[0], y }, 1);
            return Conv(c, cv2, cat, "c2_2");
        }

        /// <summary>C3: cv1,cv2 parallel→Bottleneck chain on cv1 branch→cat→cv3 / C3：cv1,cv2并行→Bottleneck链于cv1支→cat→cv3</summary>
        public static string C3(ExportContext c, Block.C3 mod, string input)
        {
            var cv1 = (Convs.Conv)mod.get_field("cv1");
            var cv2 = (Convs.Conv)mod.get_field("cv2");
            var cv3 = (Convs.Conv)mod.get_field("cv3");
            var m = EnumModules(mod.get_field("m")).Cast<Block.Bottleneck>();
            string y = Conv(c, cv1, input, "c3_1");
            foreach (var blk in m) y = Bottleneck(c, blk, y);
            var right = Conv(c, cv2, input, "c3_2");
            var cat = Ops.Concat(c, new[] { y, right }, 1);
            return Conv(c, cv3, cat, "c3_3");
        }

        /// <summary>Upsample(scale=2x2,nearest)→Resize</summary>
        public static string Upsample(ExportContext c, TorchSharp.Modules.Upsample mod, string input)
        {
            // v8-family Upsample is fixed scale_factor=2x2 nearest (hardcoded in Yolo.cs) — export as 2x2
            // v8-family Upsample is fixed scale_factor=2x2 nearest (hardcoded where Yolo.cs builds it) — exported as 2x2 / 按2x2导出
            return Ops.Resize(c, input, 2.0, 2.0, "nearest");
        }

        /// <summary>SCDown (v11 downsample): cv1(1x1)→cv2(DWConv s2) — exactly two steps in source, no pooling / 源码就两步，无池化</summary>
        public static string SCDown(ExportContext c, Block.SCDown mod, string input)
        {
            var cv1 = (Convs.Conv)mod.get_field("cv1");
            var cv2 = (Convs.DWConv)mod.get_field("cv2");
            return Conv(c, cv2, Conv(c, cv1, input, "scd1"), "scd2");
        }

        /// <summary>Pose head: box/score + raw kpts output (cv4 chain, bs,nk,-1, cat3) — kpts_decode left to runtime / Pose头：kpts原始输出——kpts_decode留运行时</summary>
        public static void Pose(ExportContext c, Head.Pose mod, string[] feats, Dictionary<string, string> outputs)
        {
            Detect(c, mod, feats, outputs);
            var cv4 = ((TorchSharp.Modules.ModuleList<TorchSharp.Modules.Sequential>)mod.get_field("cv4")).ToArray();
            int nk = 17;
            var nkField = typeof(Head.Pose).GetField("nk", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public);
            if (nkField != null) nk = (int)Convert.ChangeType(nkField.GetValue(mod), typeof(int));
            // [Fix 2026-11-13] bs=-1 like the Detect head (was hardcoded 1): dynamic batch must
            // propagate through the kpts branch too, else N folds into the anchor dim.
            // Anchor dim uses the CONCRETE H*W (like Detect's HWOf) — a Reshape may carry only ONE
            // -1 wildcard (ORT ShapeInferenceError otherwise). / 对齐Detect头bs=-1；anchor维用具体值
            long bs = -1;
            var kptViews = new List<string>();
            for (int i = 0; i < feats.Length; i++)
            {
                var kOut = SequentialDown(c, cv4[i], feats[i]);
                kptViews.Add(Ops.Reshape(c, kOut, new long[] { bs, nk, HWOf(c, kOut) }));   // nk already includes dims (num*dim) / nk已含维数(num*dim)
            }
            outputs["kpts"] = Ops.Concat(c, kptViews.ToArray(), 2);
        }

        /// <summary>Obb head: box/score + angle((sigmoid-0.25)·π, cv4 chain view(bs,ne,-1) cat3) / Obb头：angle=(sigmoid-0.25)·π</summary>
        public static void Obb(ExportContext c, Head.Obb mod, string[] feats, Dictionary<string, string> outputs)
        {
            Detect(c, mod, feats, outputs);
            var cv4 = ((TorchSharp.Modules.ModuleList<TorchSharp.Modules.Sequential>)mod.get_field("cv4")).ToArray();
            int ne = 1;   // ne = angle channel count / 角度通道数
            var neF = typeof(Head.Obb).GetField("ne", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (neF != null) ne = (int)Convert.ChangeType(neF.GetValue(mod), typeof(int));
            // [Fix 2026-11-13] bs=-1 like the Detect head (was hardcoded 1): dynamic batch must
            // propagate through the angle branch too. Anchor dim = concrete H*W (ONE -1 per Reshape).
            // 对齐Detect头bs=-1；anchor维用具体值（每个Reshape只允许一个-1）
            long bs = -1;
            var angViews = new List<string>();
            for (int i = 0; i < feats.Length; i++)
            {
                var aOut = SequentialDown(c, cv4[i], feats[i]);
                angViews.Add(Ops.Reshape(c, aOut, new long[] { bs, ne, HWOf(c, aOut) }));
            }
            var ang = Ops.Concat(c, angViews.ToArray(), 2);
            // (sigmoid-0.25)*π: Sigmoid→sub(0.25)→mul(π)
            // [Fix 2026-11-13] propagate the shape through this element-wise chain (was dropped —
            // the final output0 shape check then fell back to fully-symbolic dims). / 逐元素链补形状传播
            var sigN = c.Temp("osig");
            c.Node("Sigmoid", sigN, ang);
            var angSh = c.ShapeOf(ang); if (angSh != null) c.SetShape(sigN, angSh);
            var c025 = c.AddInitializer(c.Temp("oc0"), torch.tensor(0.25f));
            var subN = c.Temp("osub");
            c.Node("Sub", subN, sigN, c025);
            if (angSh != null) c.SetShape(subN, angSh);
            var cPi = c.AddInitializer(c.Temp("opi"), torch.tensor((float)Math.PI));
            var mulN = c.Temp("omul");
            c.Node("Mul", mulN, subN, cPi);
            if (angSh != null) c.SetShape(mulN, angSh);
            outputs["angle"] = mulN;
        }

        /// <summary>Segment head: box/score (base Detect logic) + mask coeffs (cv4) + proto — four outputs / Segment头：四输出</summary>
        public static void Segment(ExportContext c, Head.Segment mod, string[] feats, Dictionary<string, string> outputs)
        {
            Detect(c, mod, feats, outputs);   // boxes/scores

            // cv4 mask coefficients (same structure as cv2/cv3: Conv,Conv,Conv1x1) / cv4 mask系数结构同cv2cv3同）
            var cv4 = ((TorchSharp.Modules.ModuleList<TorchSharp.Modules.Sequential>)mod.get_field("cv4")).ToArray();
            int nl = feats.Length;
            // [Fix 2026-11-13] bs=-1 like the Detect head (was hardcoded 1): dynamic batch must
            // propagate through the mask branch too. Anchor dim = concrete H*W (ONE -1 per Reshape).
            // 对齐Detect头bs=-1；anchor维用具体值（每个Reshape只允许一个-1）
            long bs = -1;
            int nm = 32;   // nm default 32 (mask channels) / mask通道默认32
            var nmF = typeof(Head.Segment).GetField("nm", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (nmF != null) nm = (int)Convert.ChangeType(nmF.GetValue(mod), typeof(int));
            var maskViews = new List<string>();
            for (int i = 0; i < nl; i++)
            {
                var mOut = SequentialDown(c, cv4[i], feats[i]);
                long mC = (long)nm;
                maskViews.Add(Ops.Reshape(c, mOut, new long[] { bs, mC, HWOf(c, mOut) }));
            }
            outputs["masks"] = Ops.Concat(c, maskViews.ToArray(), 2);

            // proto: applied on feats[0] (P3 level), output (mask count, H/4, W/4) / proto作用于P3层3层）
            var proto = mod.get_field("proto");
            outputs["proto"] = SequentialDown(c, proto, feats[0]);
        }

        /// <summary>ABlock (v12 area attention): x+attn(x) → x+mlp(x)</summary>
        public static string ABlock(ExportContext c, Block.ABlock mod, string input)
        {
            var attn = (Block.AAttn)mod.get_field("attn");
            var mlp = mod.get_field("mlp");
            var x = AAttn(c, attn, input);
            var o1 = Ops.Add(c, input, x);
            var m = SequentialDown(c, mlp, o1);   // Conv1x1×2
            return Ops.Add(c, o1, m);
        }

        /// <summary>AAttn (v12): qkv→flatten2→T→[area partition]→view/permute/split→qk^T·scale→softmax→v·attn→permute→restore→NCHW+pe(v)→proj</summary>
        public static string AAttn(ExportContext c, Block.AAttn mod, string input)
        {
            var t = typeof(Block.AAttn);
            long F(string f) { var v = t.GetField(f, System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).GetValue(mod); return (long)Convert.ChangeType(v, typeof(long)); }
            long num_heads = F("num_heads"), head_dim = F("head_dim"), area = F("area");
            long C = num_heads * head_dim;
            var qkvConv = (Convs.Conv)mod.get_field("qkv");
            var peConv = (Convs.Conv)mod.get_field("pe");
            var projConv = (Convs.Conv)mod.get_field("proj");

            var sh = c.ShapeOf(input);
            if (sh == null) throw new NotSupportedException("AAttn input shape unknown / AAttn输入shape未知");
            long B = sh[0], H = sh[2], W = sh[3], N = H * W;

            var qkv = Conv(c, qkvConv, input, "aqkv");
            var flat = Ops.Reshape(c, qkv, new long[] { B, C * 3, N });
            var tr = Ops.Transpose(c, flat, new long[] { 0, 2, 1 });
            long Bq = B, Nq = N;
            if (area > 1)
            {
                tr = Ops.Reshape(c, tr, new long[] { B * area, N / area, C * 3 });
                Bq = B * area; Nq = N / area;
            }
            var vw = Ops.Reshape(c, tr, new long[] { Bq, Nq, num_heads, head_dim * 3 });
            var pm = Ops.Transpose(c, vw, new long[] { 0, 2, 3, 1 });
            var parts = Ops.Split(c, pm, new long[] { head_dim, head_dim, head_dim }, axis: 2);
            var q = parts[0]; var k = parts[1]; var v = parts[2];

            var qT = Ops.Transpose(c, q, new long[] { 0, 1, 3, 2 });
            var mm1 = c.Temp("amm1");
            c.Node("MatMul", mm1, qT, k);
            var scName = c.AddInitializer(c.Temp("aasc"), torch.tensor((float)Math.Pow(head_dim, -0.5)));
            var scM = c.Temp("aascm");
            c.Node("Mul", scM, mm1, scName);
            var sm = c.Temp("aasm");
            c.Node("Softmax", sm, scM).AttrInts.Add(("axis", -1));
            var smT = Ops.Transpose(c, sm, new long[] { 0, 1, 3, 2 });
            var mm2 = c.Temp("aamm2");
            c.Node("MatMul", mm2, v, smT);
            var xP = Ops.Transpose(c, mm2, new long[] { 0, 3, 1, 2 });
            var vP = Ops.Transpose(c, v, new long[] { 0, 3, 1, 2 });
            xP = Ops.Reshape(c, xP, new long[] { B, N, C });
            vP = Ops.Reshape(c, vP, new long[] { B, N, C });
            var xN = Ops.Transpose(c, Ops.Reshape(c, xP, new long[] { B, H, W, C }), new long[] { 0, 3, 1, 2 });
            var vN = Ops.Transpose(c, Ops.Reshape(c, vP, new long[] { B, H, W, C }), new long[] { 0, 3, 1, 2 });
            var peO = Conv(c, peConv, vN, "ape");
            var sum = Ops.Add(c, xN, peO);
            var pr = Conv(c, projConv, sum, "aproj");
            c.SetShape(pr, new long[] { B, C, H, W });
            return pr;
        }

        /// <summary>C2PSA (v11 tail attention): cv1→split(c,c)→[a ∥ m(PSABlock chain)]→cat→cv2 / C2PSA（v11尾部注意力）</summary>
        public static string C2PSA(ExportContext c, Block.C2PSA mod, string input)
        {
            var cv1 = (Convs.Conv)mod.get_field("cv1");
            var cv2 = (Convs.Conv)mod.get_field("cv2");
            var cF = typeof(Block.C2PSA).GetField("c", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            int cc = (int)(long)Convert.ChangeType(cF.GetValue(mod), typeof(long));
            var cv1Out = Conv(c, cv1, input, "psa1");
            var sp = Ops.Split(c, cv1Out, new long[] { cc, cc });
            var b = sp[1];
            foreach (var layer in EnumModules(mod.get_field("m")))
                b = layer is Block.PSABlock pb ? PSABlock(c, pb, b) : SequentialDown(c, layer, b);
            var cat = Ops.Concat(c, new[] { sp[0], b }, 1);
            return Conv(c, cv2, cat, "psa2");
        }

        /// <summary>PSABlock: x+attn(x) → x+ffn(x) (shortcut always true in v11 build) / shortcut恒true</summary>
        public static string PSABlock(ExportContext c, Block.PSABlock mod, string input)
        {
            var addF = typeof(Block.PSABlock).GetField("add", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            bool add = addF == null || (bool)addF.GetValue(mod);   // default true / 默认true
            var attn = (Block.Attention)mod.get_field("attn");
            var ffn = mod.get_field("ffn");
            var x = Attention(c, attn, input);
            if (add) x = Ops.Add(c, input, x);
            var f = SequentialDown(c, ffn, x);
            if (add) f = Ops.Add(c, x, f);
            return f;
        }

        /// <summary>Attention (self-attention): qkv→view(B,h,kd*2+hd,N)→split3→qT*k*scale→softmax→v*attnT→view+pe(v)</summary>
        public static string Attention(ExportContext c, Block.Attention mod, string input)
        {
            var t = typeof(Block.Attention);
            long F(string f) { var v = t.GetField(f, System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).GetValue(mod); return (long)Convert.ChangeType(v, typeof(long)); }
            long num_heads = F("num_heads"), key_dim = F("key_dim"), head_dim = F("head_dim");
            var scale = (float)Convert.ChangeType(t.GetField("scale", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).GetValue(mod), typeof(float));
            var qkvConv = (Convs.Conv)mod.get_field("qkv");
            var peConv = (Convs.Conv)mod.get_field("pe");
            var projConv = (Convs.Conv)mod.get_field("proj");

            var sh = c.ShapeOf(input);   // (B,C,H,W) — available via shape propagation / shape传播可得
            if (sh == null) throw new NotSupportedException("Attention input shape unknown — upstream layer missing shape propagation / 前置层缺shape传播");
            long B = sh[0], C = sh[1], H = sh[2], W = sh[3], N = H * W;

            var qkv = Conv(c, qkvConv, input, "qkv");           // actually (B, h*(2kd+hd), H, W) / 实为该shape, h*(2kd+hd), H, W)
            var flat = Ops.Reshape(c, qkv, new long[] { B, num_heads, 2 * key_dim + head_dim, N });
            var parts = Ops.Split(c, flat, new long[] { key_dim, key_dim, head_dim }, axis: 2);
            var q = parts[0]; var k = parts[1]; var v = parts[2];

            // attn = softmax(qT·k * scale)
            var qT = Ops.Transpose(c, q, new long[] { 0, 1, 3, 2 });
            var attnM = c.Temp("mm1");
            c.Node("MatMul", attnM, qT, k);
            var scName = c.AddInitializer(c.Temp("asc"), torch.tensor(scale));
            var attnS = c.Temp("ascm");
            c.Node("Mul", attnS, attnM, scName);
            var attnSm = c.Temp("asm");
            c.Node("Softmax", attnSm, attnS).AttrInts.Add(("axis", -1));
            // x = v·attnT → (B,h,hd,N)
            var attnT = Ops.Transpose(c, attnSm, new long[] { 0, 1, 3, 2 });
            var outM = c.Temp("mm2");
            c.Node("MatMul", outM, v, attnT);
            var outR = Ops.Reshape(c, outM, new long[] { B, C, H, W });
            // + pe(v.view(B,C,H,W))
            var vR = Ops.Reshape(c, v, new long[] { B, C, H, W });
            var peO = Conv(c, peConv, vR, "pe");
            var sum = Ops.Add(c, outR, peO);
            var pr = Conv(c, projConv, sum, "proj");
            c.SetShape(pr, new long[] { B, C, H, W });
            return pr;
        }

        /// <summary>A2C2f (v12 attention C2f): y={cv1(x)} no chunk→m chain→cat→cv2 (→optional gamma scale) / A2C2f：不chunk</summary>
        public static string A2C2f(ExportContext c, Block.A2C2f mod, string input)
        {
            var cv1 = (Convs.Conv)mod.get_field("cv1");
            var cv2 = (Convs.Conv)mod.get_field("cv2");
            var m = EnumModules(mod.get_field("m"));
            var y = new List<string> { Conv(c, cv1, input, "a21") };
            foreach (var layer in m)
            {
                var o = layer switch
                {
                    Block.C3k k3 => C3(c, k3, y[y.Count - 1]),      // C3k inherits C3 / C3k继承C3
                    Block.C2f f => C2f(c, f, y[y.Count - 1]),
                    Block.C2 c2 => C2(c, c2, y[y.Count - 1]),
                    Convs.Conv cv => Conv(c, cv, y[y.Count - 1], "a2m"),
                    _ => SequentialDown(c, layer, y[y.Count - 1]),
                };
                y.Add(o);
            }
            var cat = Ops.Concat(c, y.ToArray(), 1);
            var o2 = Conv(c, cv2, cat, "a22");
            // gamma: channel scale (Parameter) — read value and emit Mul / gamma通道缩放读值生成Mul
            var gammaF = typeof(Block.A2C2f).GetField("gamma", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var gamma = gammaF?.GetValue(mod) as torch.Tensor;   // Parameter inherits Tensor / Parameter继承Tensorr继承Tensor
            if (!ReferenceEquals(gamma, null))
            {
                var gName = c.AddInitializer(c.Temp("gm"), gamma);
                var mulN = c.Temp("gmul");
                var mn = c.Node("Mul", mulN, o2, gName);
                mn.AttrInts.Add(("axis", 1));
                o2 = mulN;
            }
            return o2;
        }

        /// <summary>C3k2/C2fCIB (v11/v12): same structure as C2f (chunk2→m chain→cat→cv2) — m is Sequential containing C2f/C3k/CIB / 与C2f同构</summary>
        public static string C2fLike(ExportContext c, torch.nn.Module mod, string input)
        {
            var cv1 = (Convs.Conv)mod.get_field("cv1");
            var cv2 = (Convs.Conv)mod.get_field("cv2");
            var m = EnumModules(mod.get_field("m"));
            int hidden = (int)cv1.conv.out_channels / 2;
            var cv1Out = Conv(c, cv1, input, "ck1");
            var sp = Ops.Split(c, cv1Out, new long[] { hidden, hidden });
            var y = new List<string> { sp[0], sp[1] };
            foreach (var layer in m)
            {
                var o = layer switch
                {
                    Block.Bottleneck bt => Bottleneck(c, bt, y[y.Count - 1]),
                    Block.C2f f => C2f(c, f, y[y.Count - 1]),
                    Block.C2 c2 => C2(c, c2, y[y.Count - 1]),
                    Block.C3 c3 => C3(c, c3, y[y.Count - 1]),
                    Convs.Conv cv => Conv(c, cv, y[y.Count - 1], "ckm"),
                    _ => SequentialDown(c, layer, y[y.Count - 1]),
                };
                y.Add(o);
            }
            var cat = Ops.Concat(c, y.ToArray(), 1);
            return Conv(c, cv2, cat, "ck2");
        }

        /// <summary>Generic module-list enumeration: ModuleList&lt;T&gt; / Sequential / IEnumerable&lt;Module&gt; unified / 模块列表通用枚举</summary>
        static System.Collections.Generic.IEnumerable<torch.nn.Module> EnumModules(object listObj)
        {
            // Prefer named_children (reliable for ModuleList/Sequential); IEnumerable may enumerate parameters, not submodules
            // 优先named_children；IEnumerable枚举的可能是参数不是子模块
            if (listObj is torch.nn.Module mm)
                foreach (var (_, child) in mm.named_children()) yield return child;
            else if (listObj is System.Collections.IEnumerable en)
                foreach (var o in en) if (o is torch.nn.Module m2) yield return m2;
        }

        /// <summary>Unrecognized module fallback: recursive drill-down via named_children (Sequential chains; v11/v12 C3k2 internals are mostly Conv/C2f/PSA combos) / 未识别模块兜底递归下钻</summary>
        public static string SequentialDown(ExportContext c, torch.nn.Module mod, string input)
        {
            var children = mod.named_children().ToList();
            if (children.Count == 0)
            {
                // Activation leaves: SiLU→expand; Identity/ReLU/LeakyReLU etc.→passthrough / 激活叶子处理
                var tn = mod.GetType().Name;
                if (tn == "SiLU") return Ops.SiLU(c, input);
                if (tn == "Identity" || tn == "ReLU" || tn == "LeakyReLU" || tn == "ReLU6") return input;
                if (mod is TorchSharp.Modules.MaxPool2d mp)
                {
                    long kk = (long)mp.kernel_size[0];
                    return Ops.MaxPool(c, input, new long[] { kk, kk },
                        new long[] { (long)mp.stride[0], (long)mp.stride[0] },
                        new long[] { (long)mp.padding[0], (long)mp.padding[0], (long)mp.padding[0], (long)mp.padding[0] });
                }
                if (mod is TorchSharp.Modules.Conv2d raw2) return RawConv2d(c, raw2, input, "sd_raw");
                throw new NotSupportedException($"Module {mod.GetType().Name} has no children and no processor — please add one / 无子级且无处理器——请补充处理器");
            }
            // Proto module handled as a whole (Block.Proto: cv1→ConvT(2x2s2)→cv2(3x3)→cv3(1x1)) / Proto模块整体处理
            if (mod.GetType().Name == "Proto")
            {
                var cv1 = (Convs.Conv)mod.get_field("cv1");
                var cv2 = (Convs.Conv)mod.get_field("cv2");
                var cv3 = (Convs.Conv)mod.get_field("cv3");
                var up = (TorchSharp.Modules.ConvTranspose2d)mod.get_field("upsample");
                var x = Conv(c, cv1, input, "pr1");
                var upN = c.Temp("prt");
                var un = c.Node("ConvTranspose", upN, x);
                un.AttrIntLists.Add(("kernel_shape", new long[] { 2, 2 }));
                un.AttrIntLists.Add(("strides", new long[] { 2, 2 }));
                un.AttrIntLists.Add(("pads", new long[] { 0, 0, 0, 0 }));
                un.Inputs.Add(c.AddInitializer(c.Temp("prtw"), up.weight.detach().to_type(ScalarType.Float32)));
                un.Inputs.Add(c.AddInitializer(c.Temp("prtb"), up.bias.detach().to_type(ScalarType.Float32)));
                var inSh = c.ShapeOf(x);
                if (inSh != null) c.SetShape(upN, new long[] { inSh[0], inSh[1], inSh[2] * 2, inSh[3] * 2 });
                var x2 = Conv(c, cv2, upN, "pr2");
                return Conv(c, cv3, x2, "pr3");
            }

            foreach (var (name, child) in children)
            {
                input = child switch
                {
                    Convs.Conv cv => Conv(c, cv, input, "sd"),
                    Block.C2f f => C2f(c, f, input),
                    Block.C2 c2 => C2(c, c2, input),
                    Block.C3k k3 => C3(c, k3, input),          // C3k inherits C3 — must precede C3 / C3k继承C3——必须放C3前
                    Block.C3 c3 => C3(c, c3, input),
                    Block.Bottleneck bt => Bottleneck(c, bt, input),
                    Block.SPPF sf => SPPF(c, sf, input),
                    TorchSharp.Modules.Upsample up => Upsample(c, up, input),
                    TorchSharp.Modules.Conv2d raw => RawConv2d(c, raw, input, "raw"),
                    Block.ABlock ab => ABlock(c, ab, input),
                    Block.PSABlock pb => PSABlock(c, pb, input),
                    _ => SequentialDown(c, child, input),   // deeper combos keep drilling / 深层组合继续下钻
                };
            }
            return input;
        }

        /// <summary>HGStem (v9 backbone stem): stem1→pad→[stem2a→pad→stem2b ∥ pool]→cat→stem3→stem4 / v9骨干头</summary>
        public static string HGStem(ExportContext c, Block.HGStem mod, string input)
        {
            var stem1 = (Convs.Conv)mod.get_field("stem1");
            var stem2a = (Convs.Conv)mod.get_field("stem2a");
            var stem2b = (Convs.Conv)mod.get_field("stem2b");
            var stem3 = (Convs.Conv)mod.get_field("stem3");
            var stem4 = (Convs.Conv)mod.get_field("stem4");
            var pool = (TorchSharp.Modules.MaxPool2d)mod.get_field("pool");

            var x = Conv(c, stem1, input, "hgs1");
            x = Ops.Pad(c, x, 0, 1, 0, 1);                       // pad(w right 1, h bottom 1) / pad(w右1,h下1)
            var x2 = Conv(c, stem2a, x, "hgs2a");
            x2 = Ops.Pad(c, x2, 0, 1, 0, 1);
            x2 = Conv(c, stem2b, x2, "hgs2b");
            long pk = (long)pool.kernel_size[0];
            var x1 = Ops.MaxPool(c, x, new long[] { pk, pk }, new long[] { (long)pool.stride[0], (long)pool.stride[0] }, new long[] { 0, 0, 0, 0 });
            var cat = Ops.Concat(c, new[] { x1, x2 }, 1);
            cat = Conv(c, stem3, cat, "hgs3");
            return Conv(c, stem4, cat, "hgs4");
        }

        /// <summary>HGBlock (v9 GELAN unit): y={x}+m chain→cat→sc→ec→(add? +x) / v9 GELAN单元</summary>
        public static string HGBlock(ExportContext c, Block.HGBlock mod, string input)
        {
            var sc = (Convs.Conv)mod.get_field("sc");
            var ec = (Convs.Conv)mod.get_field("ec");
            var m = EnumModules(mod.get_field("m"));
            var addF = typeof(Block.HGBlock).GetField("add", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            bool add = addF != null && (bool)addF.GetValue(mod);

            var y = new List<string> { input };
            var cur = input;
            foreach (var layer in m)
            {
                cur = layer switch
                {
                    Convs.Conv cv => Conv(c, cv, cur, "hgb"),
                    Convs.LightConv lc => LightConv(c, lc, cur),
                    _ => SequentialDown(c, layer, cur),
                };
                y.Add(cur);
            }
            var cat = Ops.Concat(c, y.ToArray(), 1);
            var o = Conv(c, ec, Conv(c, sc, cat, "hgsc"), "hgec");
            if (add) return Ops.Add(c, o, input);
            return o;
        }

        /// <summary>LightConv: conv1(1x1,Identity)→conv2(DWConv+act)</summary>
        public static string LightConv(ExportContext c, Convs.LightConv mod, string input)
        {
            var conv1 = (Convs.Conv)mod.get_field("conv1");
            var conv2 = (Convs.DWConv)mod.get_field("conv2");
            var x = Conv(c, conv1, input, "lc1");   // act=Identity auto passthrough / act=Identity自动透传
            return Conv(c, conv2, x, "lc2");
        }

        // ── GELAN (v9) family / GELAN(v9)家族 ──

        static string Cv(torch.nn.Module mod) => null;   // placeholder against misuse / 占位防误用

        static Convs.Conv GetConv(torch.nn.Module m, string f) => (Convs.Conv)m.get_field(f);

        /// <summary>ELAN1/RepNCSPELAN4 (v9 main unit): y=cv1(x).chunk2; a=cv2(y1); b=cv3(a); out=cv4(cat(y0,y1,a,b))
        /// cv2/cv3: ELAN1=direct Conv; RepNCSPELAN4=Sequential(RepNCSP+Conv)→SequentialDown</summary>
        public static string Elan(ExportContext c, torch.nn.Module mod, string input)
        {
            var cv1 = GetConv(mod, "cv1");
            var cv4 = GetConv(mod, "cv4");
            var cv2o = mod.get_field("cv2");
            var cv3o = mod.get_field("cv3");
            int half = (int)cv1.conv.out_channels / 2;
            var sp = Ops.Split(c, Conv(c, cv1, input, "el1"), new long[] { half, half });
            string Branch(torch.nn.Module b, string inp, string hint)
            {
                if (b is Convs.Conv bc) return Conv(c, bc, inp, hint);
                if (b is TorchSharp.Modules.Sequential seq)
                {
                    // RepNCSP+Conv chain / RepNCSP+Conv链
                    var cur = inp;
                    foreach (var (nm, ch) in seq.named_children())
                    {
                        cur = ch switch
                        {
                            YoloSharp.Models.Gelan.RepNCSP rncsp => RepNCSP(c, rncsp, cur),
                            Convs.Conv cv => Conv(c, cv, cur, hint + "s"),
                            _ => SequentialDown(c, ch, cur),
                        };
                    }
                    return cur;
                }
                return SequentialDown(c, b, inp);
            }
            var a = Branch(cv2o, sp[1], "el2");
            var b = Branch(cv3o, a, "el3");
            var cat = Ops.Concat(c, new[] { sp[0], sp[1], a, b }, 1);
            return Conv(c, cv4, cat, "el4");
        }

        /// <summary>RepNCSP (v9): a=m chain(cv1(x)); out=cv3(cat(a, cv2(x)))</summary>
        public static string RepNCSP(ExportContext c, YoloSharp.Models.Gelan.RepNCSP mod, string input)
        {
            var cv1 = GetConv(mod, "cv1");
            var cv2 = GetConv(mod, "cv2");
            var cv3 = GetConv(mod, "cv3");
            var a = Conv(c, cv1, input, "rnc1");
            foreach (var btk in EnumModules(mod.get_field("m")))
                a = btk is YoloSharp.Models.Gelan.RepNBottleneck rnb ? RepNBottleneck(c, rnb, a) : SequentialDown(c, btk, a);
            var byp = Conv(c, cv2, input, "rnc2");
            var cat = Ops.Concat(c, new[] { a, byp }, 1);
            return Conv(c, cv3, cat, "rnc3");
        }

        /// <summary>SPPELAN (v9 neck): cv1→[cv2→cv3→cv4 serial]→cat4→cv5 / v9颈部</summary>
        public static string SpPElan(ExportContext c, torch.nn.Module mod, string input)
        {
            // cv2/cv3/cv4 are MaxPool2d(5,s1,p2), not Conv (SPPF-style serial pooling) / cv2/cv3/cv4是MaxPool不是Conv
            var y0 = Conv(c, GetConv(mod, "cv1"), input, "sp1");
            string Step(string inp, string f, string hint)
            {
                var mp = (TorchSharp.Modules.MaxPool2d)mod.get_field(f);
                long kk = (long)mp.kernel_size[0];
                return Ops.MaxPool(c, inp, new long[] { kk, kk }, new long[] { (long)mp.stride[0], (long)mp.stride[0] },
                    new long[] { kk / 2, kk / 2, kk / 2, kk / 2 });
            }
            var y1 = Step(y0, "cv2", "sp2");
            var y2 = Step(y1, "cv3", "sp3");
            var y3 = Step(y2, "cv4", "sp4");
            var cat = Ops.Concat(c, new[] { y0, y1, y2, y3 }, 1);
            return Conv(c, GetConv(mod, "cv5"), cat, "sp5");
        }

        /// <summary>ADown (v9 downsample): avgpool2→chunk2→[cv1(y0) ∥ maxpool3s2+cv2(y1)]→cat / v9下采样</summary>
        public static string ADown(ExportContext c, YoloSharp.Models.Gelan.ADown mod, string input)
        {
            var pooled = c.Temp("adp");
            var ap = c.Node("AveragePool", pooled, input);
            ap.AttrIntLists.Add(("kernel_shape", new long[] { 2, 2 }));
            ap.AttrIntLists.Add(("strides", new long[] { 1, 1 }));   // k2s1: size unchanged (ceil aligned) / k2s1：尺寸不变(ceil对齐)
            ap.AttrIntLists.Add(("pads", new long[] { 0, 0, 0, 0 }));
            ap.AttrInts.Add(("ceil_mode", 1));
            // [Fix 2026-11-13] propagate the pool shape (k2s1+ceil_mode keeps H/W — identity) —
            // without the entry every downstream op loses propagation and the head Reshape falls
            // back to double -1 (ORT rejects the graph). / 池化形状传播（k2s1恒等，但必须有表项）
            var adIn = c.ShapeOf(input);
            if (adIn != null) c.SetShape(pooled, new long[] { adIn[0], adIn[1], adIn[2], adIn[3] });
            var cv1 = (Convs.Conv)mod.get_field("cv1");
            var cv2 = (Convs.Conv)mod.get_field("cv2");
            int half = (int)cv1.conv.in_channels;   // after chunk(2) each half = pooled channels/2 = cv1 input / chunk(2)后每半
            var sp = Ops.Split(c, pooled, new long[] { half, half });
            var x1 = Conv(c, cv1, sp[0], "ad1");
            var mp = Ops.MaxPool(c, sp[1], new long[] { 3, 3 }, new long[] { 2, 2 }, new long[] { 1, 1, 1, 1 });
            var x2 = Conv(c, cv2, mp, "ad2");
            return Ops.Concat(c, new[] { x1, x2 }, 1);
        }

        /// <summary>AConv (v9): avgpool2s1→cv1</summary>
        public static string AConv(ExportContext c, YoloSharp.Models.Gelan.AConv mod, string input)
        {
            var pooled = c.Temp("acp");
            var ap = c.Node("AveragePool", pooled, input);
            ap.AttrIntLists.Add(("kernel_shape", new long[] { 2, 2 }));
            ap.AttrIntLists.Add(("strides", new long[] { 1, 1 }));   // k2s1 same as ADown / k2s1同ADown
            ap.AttrIntLists.Add(("pads", new long[] { 0, 0, 0, 0 }));
            ap.AttrInts.Add(("ceil_mode", 1));
            // [Fix 2026-11-13] propagate the pool shape — k2s1+ceil keeps H/W (identity), but the
            // shape entry itself must exist or every downstream op loses propagation. / 池化形状传播
            var acIn = c.ShapeOf(input);
            if (acIn != null) c.SetShape(pooled, new long[] { acIn[0], acIn[1], acIn[2], acIn[3] });
            return Conv(c, (Convs.Conv)mod.get_field("cv1"), pooled, "ac1");
        }

        /// <summary>RepVGGDW (v12): act(conv(x)+conv1(x)) — two-branch sum / 双分支和</summary>
        public static string RepVGGDW(ExportContext c, Block.RepVGGDW mod, string input)
        {
            var conv = (Convs.Conv)mod.get_field("conv");
            var conv1 = (Convs.Conv)mod.get_field("conv1");
            var a = Conv(c, conv, input, "rv1");
            var b = Conv(c, conv1, input, "rv2");
            var sum = Ops.Add(c, a, b);
            var act = mod.get_field("act");
            if (act?.GetType().Name == "SiLU") return Ops.SiLU(c, sum);
            return sum;
        }

        /// <summary>RepConvN (v9): act(conv1(x)+conv2(x)) — 3x3 and 1x1 two-branch sum / 3x3与1x1双分支和</summary>
        public static string RepConvN(ExportContext c, YoloSharp.Models.Gelan.RepConvN mod, string input)
        {
            var c1m = (Convs.Conv)mod.get_field("conv1");
            var c2m = (Convs.Conv)mod.get_field("conv2");
            var a = Conv(c, c1m, input, "rcn1");
            var b = Conv(c, c2m, input, "rcn2");
            var sum = Ops.Add(c, a, b);
            var act = mod.get_field("act");
            if (act?.GetType().Name == "SiLU") return Ops.SiLU(c, sum);
            return sum;
        }

        /// <summary>RepNBottleneck (v9): RepConvN×2 + optional residual / 可选残差</summary>
        public static string RepNBottleneck(ExportContext c, YoloSharp.Models.Gelan.RepNBottleneck mod, string input)
        {
            var cv1 = (YoloSharp.Models.Gelan.RepConvN)mod.get_field("cv1");   // RepConvN
            var cv2 = (Convs.Conv)mod.get_field("cv2");                          // plain Conv / 普通Conv
            var addF = typeof(YoloSharp.Models.Gelan.RepNBottleneck).GetField("add", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var y = Conv(c, cv2, RepConvN(c, cv1, input), "rnb2");
            if (addF != null && (bool)addF.GetValue(mod)) return Ops.Add(c, input, y);
            return y;
        }

        /// <summary>RepConv (deploy state): training two-branch needs fusion — here exports the two-branch sum (numerically identical to training state) / 导出双分支和（与训练态数值一致）</summary>
        public static string RepConv(ExportContext c, Convs.RepConv mod, string input)
        {
            var conv1 = (Convs.Conv)mod.get_field("conv1");
            var conv2 = (Convs.Conv)mod.get_field("conv2");
            var a = Conv(c, conv1, input, "rc1");
            var b = Conv(c, conv2, input, "rc2");
            return Ops.Add(c, a, b);
        }

        /// <summary>Plain Conv2d (head last layer): direct output without BN / 纯Conv2d（head末层）：无BN直出</summary>
        public static string RawConv2d(ExportContext c, Conv2d conv, string input, string hint)
        {
            var w = conv.weight.detach().to_type(ScalarType.Float32);
            var b = conv.bias is null ? null : conv.bias.detach().to_type(ScalarType.Float32);
            return Ops.Conv(c, input, c.Temp(hint + "_w"), w, b,
                new long[] { (long)conv.kernel_size[0], (long)conv.kernel_size[1] },
                new long[] { (long)conv.stride[0], (long)conv.stride[1] },
                new long[] { (long)conv.padding[0], (long)conv.padding[1], (long)conv.padding[0], (long)conv.padding[1] },
                new long[] { 1, 1 }, (long)conv.groups, hint);
        }

        /// <summary>Classify head: conv→AdaptiveAvgPool(1)→flatten→linear→(bs,nc)</summary>
        public static void Classify(ExportContext c, Head.Classify mod, string input, Dictionary<string, string> final)
        {
            var conv = (Convs.Conv)mod.get_field("conv");
            var linear = (TorchSharp.Modules.Linear)mod.get_field("linear");
            var x = Conv(c, conv, input, "clsc");    // conv before pool / conv在pool前
            var pooled = c.Temp("clsp");
            c.Node("GlobalAveragePool", pooled, x);
            var flatSh = c.ShapeOf(pooled);
            var flat = Ops.Reshape(c, pooled, new long[] { flatSh != null ? flatSh[0] : 1, -1 });
            var outN = c.Temp("clso");
            var ln = c.Node("Gemm", outN, flat);
            ln.Inputs.Add(c.AddInitializer(c.Temp("clsw"), linear.weight.detach().to_type(ScalarType.Float32).transpose(0, 1).contiguous()));
            if (linear.bias is object)
                ln.Inputs.Add(c.AddInitializer(c.Temp("clsb"), linear.bias.detach().to_type(ScalarType.Float32)));
            final["cls"] = outN;
        }

    }

    /// <summary>Module access extensions (reflection read-only — no intrusion into original classes) / 模块访问扩展（反射只读——不侵入原类）</summary>
    internal static class ModuleAccess
    {
        public static torch.nn.Module get_field(this torch.nn.Module m, string name)
        {
            var f = m.GetType().GetField(name,
                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (f == null) return null;
            return f.GetValue(m) as torch.nn.Module;
        }


        public static System.Collections.Generic.IEnumerable<Sequential> GetHeads(this Head.Detect m, string key)
        {
            // Detect.heads: Dictionary<string, ModuleList<Sequential>> — used by forward_head / forward_head用
            var f = typeof(Head.Detect).GetField("heads",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
                ?? typeof(Head.Detect).GetField("heads", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
            var dict = f?.GetValue(m) as System.Collections.IDictionary;
            var list = dict?[key] as System.Collections.IEnumerable;
            if (list == null) throw new NotSupportedException($"Detect.heads[{key}] unreachable / 不可达");
            return list.Cast<Sequential>();
        }
    }
}
