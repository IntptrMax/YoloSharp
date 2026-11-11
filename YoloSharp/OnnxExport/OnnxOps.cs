// ONNX export engine: exports the YoloSharp network (ModuleList+Concat+Detect) as an ONNX graph
// Exports the YoloSharp network (ModuleList+Concat+Detect) as an ONNX graph / 把网络导出为ONNX计算图
// Architecture principles (zero coupling) / 架构原则（零耦合）:
//   1. Only depends on YoloSharp public module types (Conv/C2f/SPPF/...), reads params via reflection, never modifies them
//      depends only on public YoloSharp module types; reads parameters via reflection, never modifies them / 只依赖公开类型反射只读
//   2. BN fused into Conv weights (inference-equivalent), SiLU = Sigmoid*Mul expanded
//      BN fused into Conv weights (inference-equivalent); SiLU expanded as Sigmoid*Mul / BN融合SiLU展开
//   3. Detect head exports raw box/score branches (NMS post-process left to runtime C# — same boundary as ultralytics export)
//      Detect head exports raw box/score branches (NMS left to runtime C# — same boundary as ultralytics) / NMS留运行时
//   4. Single-file operator-level API (EmitConv/EmitSiLU...); module processors are thin compositions
//      single-file operator-level API (EmitConv/EmitSiLU...); module processors are thin compositions / 算子级API薄组合
using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;

using Convs = YoloSharp.Modules.Convs;
using Block = YoloSharp.Modules.Block;
using Head = YoloSharp.Modules.Head;

namespace YoloSharp.OnnxExport
{
    /// <summary>Export context: tensor names + graph + weight pool / 导出上下文：张量名录+图+权重池</summary>
    internal sealed class ExportContext
    {
        public readonly OnnxModelWriter Model = new OnnxModelWriter();
        public string Current;          // current dataflow tensor name / 当前数据流张量名
        int _tmp;
        public string Temp(string hint = "t") => $"{hint}_{_tmp++}";

        // Static shape propagation (NCHW, -1=unknown) — needed by shape-dependent ops such as Attention
        // Static shape propagation (NCHW, -1=unknown) — for operators that need shapes (Attention etc.) / 静态shape传播
        public readonly System.Collections.Generic.Dictionary<string, long[]> Shapes = new System.Collections.Generic.Dictionary<string, long[]>();
        public void SetShape(string name, long[] shape) => Shapes[name] = shape;
        public long[] ShapeOf(string name) => Shapes.TryGetValue(name, out var sh) ? sh : null;
        public string AddInitializer(string name, torch.Tensor w)
        {
            if (w.dtype == ScalarType.Int64)
            {
                var i64 = w.detach().cpu().contiguous();
                var idata = new long[i64.numel()];
                var iacc = i64.data<long>();
                for (int i = 0; i < idata.Length; i++) idata[i] = iacc[i];
                Model.AddInitializer(new OnnxTensor { Name = name, Dims = i64.shape, DataType = OnnxDataType.Int64, Int64Data = idata });
                return name;
            }
            var arr = w.detach().cpu().to_type(ScalarType.Float32).contiguous();
            var data = new float[arr.numel()];
            var acc = arr.data<float>();
            for (int i = 0; i < data.Length; i++) data[i] = acc[i];
            Model.AddInitializer(new OnnxTensor { Name = name, Dims = arr.shape, DataType = OnnxDataType.Float, FloatData = data });
            return name;
        }
        public OnnxNode Node(string op, string output, params string[] inputs)
        {
            var n = new OnnxNode(op, output);
            n.Inputs.AddRange(inputs);
            Model.AddNode(n);
            return n;
        }
    }

    /// <summary>Operator-level Emit API (processors compose here) / 算子级Emit API（处理器在此组合）</summary>
    internal static class Ops
    {
        public static string Conv(ExportContext c, string input, string wName, torch.Tensor weight, torch.Tensor bias,
            long[] kernel, long[] strides, long[] pads, long[] dilations, long group, string hint = "conv")
        {
            var outN = c.Temp(hint);
            var wI = c.AddInitializer(wName, weight);
            var n = c.Node("Conv", outN, input, wI);
            n.AttrIntLists.Add(("kernel_shape", kernel));
            n.AttrIntLists.Add(("strides", strides));
            n.AttrIntLists.Add(("pads", pads));
            n.AttrIntLists.Add(("dilations", dilations));
            n.AttrInts.Add(("group", group));
            if (bias is torch.Tensor bt)
            {
                var bI = c.AddInitializer(wName + "_b", bt);
                n.Inputs.Add(bI);
            }
            // shape propagation: out = (inH+2p-k)/s+1 / 形状传播
            var inSh = c.ShapeOf(input);
            if (inSh != null && weight.shape.Length == 4)
            {
                long kh = weight.shape[2], kw = weight.shape[3];
                long sh = strides[0], sw = strides[1];
                long ph0 = pads[0], ph1 = pads[2], pw0 = pads[1], pw1 = pads[3];
                long outC = weight.shape[0] * group;   // when groups>1 output=Cout / 分组时输出=Cout
                long outH = (inSh[2] + ph0 + ph1 - kh) / sh + 1;
                long outW = (inSh[3] + pw0 + pw1 - kw) / sw + 1;
                c.SetShape(outN, new long[] { inSh[0], weight.shape[0], outH, outW });
            }
            return outN;
        }

        public static string SiLU(ExportContext c, string input)
        {
            var sig = c.Temp("sig");
            c.Node("Sigmoid", sig, input);
            var outN = c.Temp("silu");
            c.Node("Mul", outN, input, sig);
            var sh = c.ShapeOf(input);
            if (sh != null) c.SetShape(outN, sh);
            return outN;
        }

        /// <summary>Element-wise sigmoid — used on exported cls scores to match libtorch inference
        /// (`scores.sigmoid()`); [Add 2026-11-02]</summary>
        public static string Sigmoid(ExportContext c, string input)
        {
            var outN = c.Temp("sig");
            c.Node("Sigmoid", outN, input);
            var sh = c.ShapeOf(input);
            if (sh != null) c.SetShape(outN, sh);
            return outN;
        }

        public static string Add(ExportContext c, string a, string b)
        {
            var outN = c.Temp("add");
            c.Node("Add", outN, a, b);
            var sh = c.ShapeOf(a) ?? c.ShapeOf(b);
            if (sh != null) c.SetShape(outN, sh);
            return outN;
        }

        public static string Concat(ExportContext c, string[] inputs, long axis = 1)
        {
            var outN = c.Temp("cat");
            var n = c.Node("Concat", outN, inputs);
            n.AttrInts.Add(("axis", axis));
            var sh0 = c.ShapeOf(inputs[0]);
            if (sh0 != null)
            {
                var outSh = (long[])sh0.Clone();
                for (int i = 1; i < inputs.Length; i++)
                {
                    var si = c.ShapeOf(inputs[i]);
                    if (si == null || si.Length != sh0.Length) { outSh = null; break; }
                    for (int d = 0; d < sh0.Length; d++) if (d != axis && si[d] != sh0[d]) outSh = null;
                    outSh[axis] += si[axis];
                }
                if (outSh != null) c.SetShape(outN, outSh);
            }
            return outN;
        }

        /// <summary>Split/chunk: split by given sizes (axis=1). opset13+: split is an input initializer, not an attr / opset13+：split是输入initializer不是attr</summary>
        /// <remarks>[Add 2026-11-11] useGather switch: the TRT dynamic-batch route once rewrote Split as Gather to
        /// bypass an nvonnxparser wild-index bug (ISliceLayer gather index 1630337677504) — but measurements
        /// showed: (1) TRT dynamic build was actually hitting the max-profile tactic blow-up (computeCosts),
        /// not that parser bug; (2) the Gather variant broke OV dynamic batch (partial_shape.cpp:266 crash —
        /// the Split variant works); (3) TRT b8 padding mis-sliced Gather-variant outputs. Default reverted to
        /// native Split (OV + b8 both green again); the Gather variant stays behind the switch for future use.
        /// / useGather开关：默认Split，Gather版留待后用</remarks>
        public static string[] Split(ExportContext c, string input, long[] splitSizes, long axis = 1, bool useGather = false)
        {
            if (useGather)
            {
                var outsG = Enumerable.Range(0, splitSizes.Length).Select(i => c.Temp("sp")).ToArray();
                var inShG = c.ShapeOf(input);
                long start = 0;
                for (int i = 0; i < splitSizes.Length; i++)
                {
                    long size = splitSizes[i];
                    var idx = new long[size];
                    for (long k = 0; k < size; k++) idx[k] = start + k;
                    var idxInit = new OnnxTensor
                    {
                        Name = c.Temp("spidx"), Dims = new long[] { size },
                        DataType = OnnxDataType.Int64, Int64Data = idx,
                    };
                    c.Model.AddInitializer(idxInit);
                    c.Node("Gather", outsG[i], input, idxInit.Name).AttrInts.Add(("axis", axis));
                    if (inShG != null)
                    {
                        var sh = (long[])inShG.Clone(); sh[axis] = size;
                        c.SetShape(outsG[i], sh);
                    }
                    start += size;
                }
                return outsG;
            }

            // Native Split implementation (default — OV / TRT static b8 both verified green) / 原生Split实现
            var outs = Enumerable.Range(0, splitSizes.Length).Select(i => c.Temp("sp")).ToArray();
            var splitInit = new OnnxTensor
            {
                Name = c.Temp("spsz"),
                Dims = new long[] { splitSizes.Length },
                DataType = OnnxDataType.Int64,
                Int64Data = splitSizes,
            };
            c.Model.AddInitializer(splitInit);
            var n = c.Node("Split", null, input, splitInit.Name);
            n.Outputs.Clear(); n.Outputs.AddRange(outs);
            n.AttrInts.Add(("axis", axis));
            var inSh = c.ShapeOf(input);
            if (inSh != null)
                for (int i = 0; i < outs.Length; i++)
                {
                    var sh = (long[])inSh.Clone(); sh[axis] = splitSizes[i];
                    c.SetShape(outs[i], sh);
                }
            return outs;
        }

        public static string Pad(ExportContext c, string input, long padR, long padB, long padL = 0, long padT = 0)
        {
            var outN = c.Temp("pad");
            var n = c.Node("Pad", outN, input);
            n.AttrIntLists.Add(("pads", new long[] { padT, padL, padB, padR }));   // NCHW: [h_t,w_t,h_b,w_b]
            n.AttrStrings.Add(("mode", "constant"));
            n.AttrFloats.Add(("value", 0f));
            return outN;
        }

        public static string MaxPool(ExportContext c, string input, long[] kernel, long[] strides, long[] pads)
        {
            var outN = c.Temp("mp");
            var n = c.Node("MaxPool", outN, input);
            n.AttrIntLists.Add(("kernel_shape", kernel));
            n.AttrIntLists.Add(("strides", strides));
            n.AttrIntLists.Add(("pads", pads));
            var inSh = c.ShapeOf(input);
            if (inSh != null)
            {
                long oh = (inSh[2] + pads[0] + pads[2] - kernel[0]) / strides[0] + 1;
                long ow = (inSh[3] + pads[1] + pads[3] - kernel[1]) / strides[1] + 1;
                c.SetShape(outN, new long[] { inSh[0], inSh[1], oh, ow });
            }
            return outN;
        }

        public static string Resize(ExportContext c, string input, double scaleH, double scaleW, string mode = "nearest")
        {
            var outN = c.Temp("up");
            var scalesName = c.Temp("ups");
            var scalesInit = new OnnxTensor
            {
                Name = scalesName,
                Dims = new long[] { 4 },
                DataType = OnnxDataType.Float,
                FloatData = new float[] { 1f, 1f, (float)scaleH, (float)scaleW },
            };
            c.Model.AddInitializer(scalesInit);
            // Resize: X, roi(empty), scales
            var emptyRoi = c.Temp("roi");
            c.Model.AddInitializer(new OnnxTensor { Name = emptyRoi, Dims = new long[] { 0 }, DataType = OnnxDataType.Float, FloatData = new float[0] });
            var n = c.Node("Resize", outN, input, emptyRoi, scalesName);
            n.AttrStrings.Add(("mode", mode));
            n.AttrStrings.Add(("nearest_mode", "floor"));
            n.AttrStrings.Add(("coordinate_transformation_mode", "asymmetric"));
            var inSh = c.ShapeOf(input);
            if (inSh != null) c.SetShape(outN, new long[] { inSh[0], inSh[1], (long)(inSh[2] * scaleH), (long)(inSh[3] * scaleW) });
            return outN;
        }

        public static string Reshape(ExportContext c, string input, long[] shape)
        {
            var outN = c.Temp("rs");
            var shapeI = c.AddInitializer(c.Temp("rsh"), torch.tensor(shape, ScalarType.Int64));
            c.Node("Reshape", outN, input, shapeI);
            c.SetShape(outN, shape);
            return outN;
        }

        public static string Transpose(ExportContext c, string input, long[] perm)
        {
            var outN = c.Temp("tr");
            var n = c.Node("Transpose", outN, input);
            n.AttrIntLists.Add(("perm", perm));
            var sh = c.ShapeOf(input);
            if (sh != null)
            {
                var o = new long[perm.Length];
                for (int i = 0; i < perm.Length; i++) o[i] = sh[perm[i]];
                c.SetShape(outN, o);
            }
            return outN;
        }
    }
}
