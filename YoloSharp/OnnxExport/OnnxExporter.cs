// Export main driver: Yolov8-family network (ModuleList+Concat+Detect interpreter structure) → ONNX
// Main export driver: Yolov8-family network (ModuleList+Concat+Detect interpreter structure) -> ONNX / 导出主驱动
// Walks the model list item by item (isomorphic to Yolo.forward); Concat replays skip connections via outputIndexs/concatIndex
// Iterates the model list drilling down per item (isomorphic to Yolo.forward); Concat replays concat relations via outputIndexs/concatIndex / 遍历model列表逐项下钻
using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;

using Convs = YoloSharp.Modules.Convs;
using Block = YoloSharp.Modules.Block;
using Yolo = YoloSharp.Models.Yolo;
using Head = YoloSharp.Modules.Head;

namespace YoloSharp.OnnxExport
{
    /// <summary>
    /// YOLO → ONNX exporter (NMS post-process left to runtime — same boundary as ultralytics export).
    /// YOLO -> ONNX exporter (NMS post-processing left to runtime — same boundary as ultralytics export) / NMS后处理留运行时
    /// Usage / 用法: OnnxExporter.Export(yoloModule, (1,3,H,W), "out.onnx")
    /// </summary>
    public static class OnnxExporter
    {
        /// <summary>Export entry. yolo = Yolo.Yolov8 instance (with model list); dummyShape=(1,3,H,W), multiple of 32 / 导出入口；dummyShape需32倍数</summary>
        public static void Export(torch.nn.Module yolo, long[] dummyShape, string outputPath)
        {
            Export(yolo, dummyShape, outputPath, dynamicBatch: false);
        }

        /// <summary>
        /// [Add 2026-11-10] dynamicBatch=true: input batch dim declared as -1 (symbolic) — TRT parses a dynamic
        /// input and the converter can configure an optimization profile (1~max dynamic batch in one execute);
        /// OV natively supports N frames too. Output shapes keep static inferred values (batch=1 domain); OV
        /// per-frame slicing is unaffected, and TRT dynamic engine outputs are set at runtime by
        /// context.setInputShape(N). / 输入batch维声明为-1
        /// </summary>
        public static void Export(torch.nn.Module yolo, long[] dummyShape, string outputPath, bool dynamicBatch)
        {
            var c = new ExportContext();
            string input = "images";
            // [Add 2026-11-10] Dynamic batch: graph input declares batch=-1 (shape propagation still follows dummyShape — declaration only) / 图输入声明batch=-1
            var declDims = new long[dummyShape.Length];
            for (int i = 0; i < dummyShape.Length; i++) declDims[i] = dynamicBatch && i == 0 ? -1 : dummyShape[i];
            c.Model.AddInput(new OnnxValueInfo { Name = input, Dims = declDims });
            c.SetShape(input, dummyShape);   // shape propagation starting point / shape传播起点

            // Walk isomorphic to Yolo.forward (model/outputIndexs/concatIndex are protected — via reflection)
            // Isomorphic iteration to Yolo.forward (model/outputIndexs/concatIndex are protected — fetched via reflection) / 与Yolo.forward同构的遍历
            var model = (TorchSharp.Modules.ModuleList<torch.nn.Module>)GetField(yolo, "model");
            var outputIndexs = (int[])GetField(yolo, "outputIndexs");
            var concatIndex = (int[])GetField(yolo, "concatIndex");

            var outputs = new List<string>();
            var feat = input;   // current dataflow / 当前流
            int catCount = 0;
            var finalOutputs = new Dictionary<string, string>();

            // Yolov9 specialized forward replay (isomorphic to Yolov9.cs): aux branch layer-22 rewires to layer-8 output + DetectV9 six branches
            // Yolov9-specialized forward replay (isomorphic to Yolov9.cs source): aux branch layer-22 rewires to layer-8 output + DetectV9 six branches / Yolov9特化forward重放
            bool isV9 = yolo.GetType().DeclaringType?.Name == "Yolov9" || yolo.GetType().Name == "Yolov9";
            int[] v9DetectSrc = { 10, 9, 8, 5, 6, 7 };
            bool hasAuxFork = isV9;

            for (int i = 0; i < model.Count; i++)
            {
                if (hasAuxFork && i == 22)
                {
                    feat = outputs[2];   // v9 aux branch: layer-22 (SPPELAN) input = layer-8 output / v9 aux支路：层22输入=层8输出
                }
                var mod = model[i];
                if (mod is Convs.Concat)
                {
                    var skip = outputs[concatIndex[catCount++]];
                    feat = Ops.Concat(c, new[] { feat, skip }, 1);
                }
                else if (mod is Head.DetectV9)
                {
                    // [Fix 2026-11-04] The training yaml (yolov9-s.yaml) head is the v8-style
                    // Detect with SIX groups — all participate in forward+loss (aux P3-P5 + main
                    // P3-P5, stride 8/16/32 ×2). Export all six branches at headIndexBase 0 so the
                    // deployed graph matches training (A=16800). The old main-3-only export broke
                    // the two edge objects (aux-group-only coverage) after training succeeded.
                    // The training yaml has all six head groups participating — all six branches exported (A=16800), consistent with training. / 六分支全导出与训练一致
                    var src = v9DetectSrc.Select(idx => outputs[idx]).ToArray();
                    Processors.Detect(c, (Head.Detect)mod, src, finalOutputs, headIndexBase: 0);
                }
                else if (mod is Head.Detect || mod is Head.Obb || mod is Head.Pose || mod is Head.Segment || mod is Head.Classify)
                {
                    EmitHead(c, mod, outputs, feat, finalOutputs);
                }
                else
                {
                    feat = EmitBlock(c, mod, feat);
                }

                if (outputIndexs.Contains(i)) outputs.Add(feat);
            }

            // Semantic output names: add Identity nodes to rename (avoid internal names like cat_302 — deployment backends bind tensors by name)
            // Semantic output names: Identity nodes rename them (avoids internal names like cat_302 — deployment backends bind tensors by name) / 语义输出名加Identity重命名
            // [Fix 2026-11-13 作者反馈] Output shape must be python-style mixed, not fully symbolic:
            // previously Rank=3 declared [bs, dim1, dim2] (all symbolic) — Netron/author expected the
            // torch.onnx.export convention: [batch, 84, 8400] (only batch symbolic, rest concrete).
            // The static shape is available from shape propagation (seeded by dummyShape) — read it and
            // emit dim_value for all dims except dim 0. / 输出声明改为python风格混合：仅batch符号维，其余具体值
            var parts = finalOutputs.Where(kv => kv.Key != "proto")
                                    .OrderBy(kv => kv.Key == "boxes" ? 0 : kv.Key == "scores" ? 1 : 2)
                                    .Select(kv => kv.Value).ToArray();
            string merged = parts.Length == 1 ? parts[0] : Ops.Concat(c, parts, 1);
            c.Node("Identity", "output0", merged);
            c.Model.AddOutput(MakePythonStyleOutput(c, "output0", merged));
            if (finalOutputs.ContainsKey("masks"))
            {
                // [Fix 2026-11-13] masks coefficients were computed but DROPPED from output0 (the
                // single-output refactor excluded them) — output0 was [64+nc] instead of [64+nc+32].
                // masks now rides in output0 channels after scores; output1 = proto only.
                // masks系数曾漏拼进output0——现拼入scores之后，output1仅proto。
                c.Node("Identity", "output1", finalOutputs["proto"]);
                c.Model.AddOutput(MakePythonStyleOutput(c, "output1", finalOutputs["proto"]));
            }
            c.Model.Save(outputPath, "yolosharp");
        }

        /// <summary>
        /// [Add 2026-11-13] Output declaration in torch.onnx.export style: concrete dims from static shape
        /// propagation, dynamic dims (<=0, e.g. the Detect-head bs=-1) symbolic "batch" — yielding
        /// [batch, 84, 8400], matching what the official python export produces. Falls back to
        /// fully-symbolic ranked shape only when propagation lost the shape entirely.
        /// / python风格输出声明：动态维(≤0)符号化，其余具体值；传播整体丢失才回退全符号
        /// </summary>
        static OnnxValueInfo MakePythonStyleOutput(ExportContext c, string name, string src)
        {
            var sh = c.ShapeOf(src);
            if (sh == null || sh.Length == 0)
                return new OnnxValueInfo { Name = name, Rank = 3 };   // fallback: fully symbolic / 回退全符号
            // dynamic dims (<=0 in propagation) become symbolic dim_params / 传播中≤0的维转符号维
            var sym = new List<int>();
            for (int i = 0; i < sh.Length; i++) if (sh[i] <= 0) sym.Add(i);
            // python semantic: dynamic dim is batch (dim 0); any other dynamic dim → fully symbolic fallback / 仅batch允许动态
            if (sym.Count > 0 && sym[0] != 0)
                return new OnnxValueInfo { Name = name, Rank = sh.Length };
            var dims = (long[])sh.Clone();
            for (int i = 0; i < dims.Length; i++) if (dims[i] <= 0) dims[i] = 1;   // concrete placeholder for symbolic dims / 符号维占位
            return new OnnxValueInfo { Name = name, Dims = dims, SymbolicDims = sym.Count > 0 ? sym.ToArray() : null };
        }

        /// <summary>Backbone module dispatch (everything except Detect heads) / 骨干模块分发（Detect头之外的一切）</summary>
        static string EmitBlock(ExportContext c, torch.nn.Module mod, string input)
        {
            switch (mod)
            {
                case Convs.Conv cv: return Processors.Conv(c, cv, input, "conv");
                case Block.C2f c2f: return Processors.C2f(c, c2f, input);
                case Block.SPPF sppf: return Processors.SPPF(c, sppf, input);
                case Block.C2 c2: return Processors.C2(c, c2, input);
                case Block.C3 c3: return Processors.C3(c, c3, input);
                case Block.Bottleneck bt: return Processors.Bottleneck(c, bt, input);
                case TorchSharp.Modules.Upsample up: return Processors.Upsample(c, up, input);
                case Block.SCDown scd: return Processors.SCDown(c, scd, input);
                case Block.HGStem hgs: return Processors.HGStem(c, hgs, input);
                case Block.HGBlock hgb: return Processors.HGBlock(c, hgb, input);
                case Block.A2C2f a2: return Processors.A2C2f(c, a2, input);
                case Block.C3k2 k2: return Processors.C2fLike(c, k2, input);
                case Block.C2fCIB cib: return Processors.C2fLike(c, cib, input);
                case Block.C2PSA psa: return Processors.C2PSA(c, psa, input);
                case YoloSharp.Models.Gelan.ELAN1 e1: return Processors.Elan(c, e1, input);
                case YoloSharp.Models.Gelan.RepNCSPELAN4 re: return Processors.Elan(c, re, input);
                case YoloSharp.Models.Gelan.SPPELAN spe: return Processors.SpPElan(c, spe, input);
                case YoloSharp.Models.Gelan.ADown ad: return Processors.ADown(c, ad, input);
                case YoloSharp.Models.Gelan.AConv ac: return Processors.AConv(c, ac, input);
                case Block.RepVGGDW rv: return Processors.RepVGGDW(c, rv, input);
                case Convs.RepConv rc: return Processors.RepConv(c, rc, input);
                default:
                    // Unrecognized module: try drilling down as a plain Sequential chain (covers v11 C3k2 internals, v12 attention combos, etc.)
                    // Unrecognized module: try drilling down as a plain Sequential chain (covers v11 C3k2 internals, v12 attention combos etc.) / 未识别模块作Sequential下钻
                    return Processors.SequentialDown(c, mod, input);
            }
        }

        static void EmitHead(ExportContext c, torch.nn.Module mod, List<string> outputs, string feat, Dictionary<string, string> final)
        {
            switch (mod)
            {
                case Head.Segment seg:   // Segment inherits Detect — must precede the Detect case / Segment继承Detect——必须放Detect前
                    {
                        var feats = new[] { outputs[outputs.Count - 3], outputs[outputs.Count - 2], outputs[outputs.Count - 1] };
                        Processors.Segment(c, seg, feats, final);
                        break;
                    }
                case Head.Pose pose:      // Pose inherits Detect / Pose继承Detect
                    {
                        var feats = new[] { outputs[outputs.Count - 3], outputs[outputs.Count - 2], outputs[outputs.Count - 1] };
                        Processors.Pose(c, pose, feats, final);
                        break;
                    }
                case Head.Obb obb:        // Obb inherits Detect / Obb继承Detect
                    {
                        var feats = new[] { outputs[outputs.Count - 3], outputs[outputs.Count - 2], outputs[outputs.Count - 1] };
                        Processors.Obb(c, obb, feats, final);
                        break;
                    }
                case Head.Classify cls:
                    Processors.Classify(c, cls, feat, final);
                    break;
                case Head.Detect det:   // DetectV9 inherits Detect — taking the last 3 feats already carries v9 semantics (main) / DetectV9继承Detect——feats取末3已含v9语义
                    {
                        var feats = new[] { outputs[outputs.Count - 3], outputs[outputs.Count - 2], outputs[outputs.Count - 1] };
                        Processors.Detect(c, det, feats, final);
                        break;
                    }
                default:
                    throw new NotSupportedException($"Head type {mod.GetType().Name} not supported for export yet / 头类型暂不支持导出");
            }
        }

        /// <summary>Get the internal yolo network module from a YoloTask (protected field via reflection — zero intrusion) / 从YoloTask取内部yolo网络模块（protected字段反射——零侵入）</summary>
        internal static Yolo.Yolov8 GetYoloModule(object yoloTask)
        {
            // YoloTask.yolo (YoloBaseTaskModel derived instance) → .yolo (the network itself)
            // YoloTask.yolo (YoloBaseTaskModel-derived instance) -> .yolo (the network body) / 取网络本体
            var taskModel = GetField(yoloTask, "yolo");
            return (Yolo.Yolov8)GetField(taskModel, "yolo");
        }

        static object GetField(object obj, string name)
        {
            var t = obj.GetType();
            while (t != null)
            {
                var f = t.GetField(name, System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public);
                if (f != null) return f.GetValue(obj);
                var p = t.GetProperty(name, System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public);
                if (p != null) return p.GetValue(obj);
                t = t.BaseType;
            }
            throw new MissingFieldException(obj.GetType().Name + "." + name);
        }
    }
}
