using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Data;
using OpenCvSharp;
using YoloSharp.Models;
using YoloSharp.Types;

namespace YoloSharpNet48Demo
{
    /// <summary>
    /// YoloSharp net48 兼容性验证：
    /// 与 YoloSharpDemo（net6.0）相同的功能逻辑，验证 .NET Framework 4.8 下
    /// TorchSharp + libtorch + OpenCvSharp + YoloSharp(netstandard2.0) 训练+推理链路。
    /// 统一入口（v5u/v8/v9/v11/v12 同一流程）：
    ///   YoloSharpNet48Demo [yoloType] [task] [predictImgsz] [predictConf] [epochs]
    /// ONNX导出对拍模式（与YoloSharpDemo同款）：
    ///   YoloSharpNet48Demo onnxexport [yoloType] [task] [imgsz] [--pt|--bin <path>]
    /// </summary>
    internal class Program
    {
        /// <summary>
        /// [Add 2026-11-10] ONNX export parity mode, ported from YoloSharpDemo (same functionality,
        /// net48 syntax: no nullable ref types, no target-typed new). Build network → export onnx →
        /// point-wise parity vs TorchSharp forward.
        /// ONNX导出对拍模式（与YoloSharpDemo同功能，net48语法适配）：
        /// 构建网络→导出onnx→与TorchSharp前向逐点对拍。
        /// </summary>
        static void RunOnnxExportTest(string yoloTypeName, string taskName, int imgsz, string[] args)
        {
            Console.WriteLine("=== OnnxExport parity: / 对拍: " + yoloTypeName + "+" + taskName + " imgsz=" + imgsz + " ===");
            var yoloType = (YoloType)Enum.Parse(typeof(YoloType), yoloTypeName, true);
            var task = (TaskType)Enum.Parse(typeof(TaskType), taskName, true);
            int nc = task == TaskType.Classification ? 1 : 15;

            var config = new Config
            {
                DeviceType = YoloSharp.Types.DeviceType.CPU,   // 对拍用CPU排除GPU数值差异
                ScalarType = ScalarType.Float32,
                YoloType = yoloType,
                YoloSize = yoloType == YoloType.Yolov9 ? YoloSize.s : YoloSize.n,
                TaskType = task,
                ImageSize = imgsz,
                NumberClass = nc,
            };
            if (WarnInvalidCombo(yoloType, task)) return;

            var yoloTask = new YoloTask(config);
            var yolo = YoloSharp.OnnxExport.OnnxExporter.GetYoloModule(yoloTask);
            if (args.Length > 6 && args[4] == "--pt" && File.Exists(args[5]))
            {
                yoloTask.LoadModel(args[5], skipNcNotEqualLayers: true);
                Console.WriteLine("[PtLoad] " + args[5]);
            }
            else if (args.Length > 6 && args[4] == "--bin" && File.Exists(args[5]))
            {
                yoloTask.LoadModel(args[5], skipNcNotEqualLayers: true);
                Console.WriteLine("[BinLoad] " + args[5]);
            }
            yolo.eval();

            var input = TorchSharp.torch.zeros(1, 3, imgsz, imgsz);
            TorchSharp.torch.Tensor refBoxes = null, refScores = null, refMasks = null, refProto = null, refCls = null;
            using (TorchSharp.torch.no_grad())
            {
                input = TorchSharp.torch.randn(1, 3, imgsz, imgsz);
                var fw = yolo.forward(input);
                var preds = (Dictionary<string, object>)fw.Value.Item2;
                var one2many = preds.ContainsKey("one2many") ? (Dictionary<string, object>)preds["one2many"] : preds;
                refBoxes = one2many.ContainsKey("boxes") ? ((TorchSharp.torch.Tensor)one2many["boxes"]).detach().cpu().to_type(TorchSharp.torch.ScalarType.Float32) : null;
                refScores = one2many.ContainsKey("scores") ? ((TorchSharp.torch.Tensor)one2many["scores"]).detach().cpu().to_type(TorchSharp.torch.ScalarType.Float32) : null;
                refMasks = one2many.ContainsKey("mask_coefficient") ? ((TorchSharp.torch.Tensor)one2many["mask_coefficient"]).detach().cpu().to_type(TorchSharp.torch.ScalarType.Float32) : null;
                refProto = one2many.ContainsKey("proto") ? ((TorchSharp.torch.Tensor)one2many["proto"]).detach().cpu().to_type(TorchSharp.torch.ScalarType.Float32) : null;
                object clsV;
                if (preds.TryGetValue("cls", out clsV)) refCls = clsV as TorchSharp.torch.Tensor;
            }
            Console.WriteLine("TorchSharp head outputs: boxes= / 头输出: boxes=" + (refBoxes is object ? string.Join("x", refBoxes.shape) : "-") + " scores=" + (refScores is object ? string.Join("x", refScores.shape) : "-"));

            string onnxPath = "export_" + yoloTypeName + "_" + taskName + "_" + imgsz + ".onnx";
            YoloSharp.OnnxExport.OnnxExporter.Export(yolo, new long[] { 1, 3, imgsz, imgsz }, onnxPath);
            Console.WriteLine("Export OK: / 导出OK: " + onnxPath + " (" + new FileInfo(onnxPath).Length / 1024 + "KB)");

            using (var session = new Microsoft.ML.OnnxRuntime.InferenceSession(onnxPath))
            {
                var floatArr = input.detach().cpu().to_type(TorchSharp.torch.ScalarType.Float32).data<float>().ToArray();
                var tensor = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(floatArr, new[] { 1, 3, imgsz, imgsz });
                var inputName = session.InputMetadata.First().Key;
                using (var results = session.Run(new[] { Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor(inputName, tensor) }))
                {
                    var outs = results.Select(r => r.AsTensor<float>().ToArray()).ToArray();
                    var rb = refBoxes is object ? refBoxes.data<float>().ToArray() : new float[0];
                    var rs = refScores is object ? refScores.data<float>().ToArray() : new float[0];
                    double maxDiff = 0; long nBig = 0;
                    var ob = outs[0]; var os_ = outs.Length > 1 ? outs[1] : new float[0];
                    for (int i = 0; i < Math.Min(rb.Length, ob.Length); i++)
                    {
                        var d = Math.Abs((double)rb[i] - ob[i]);
                        if (d > maxDiff) maxDiff = d; if (d > 1e-2) nBig++;
                    }
                    for (int i = 0; i < Math.Min(rs.Length, os_.Length); i++)
                    {
                        var d = Math.Abs((double)rs[i] - os_[i]);
                        if (d > maxDiff) maxDiff = d; if (d > 1e-2) nBig++;
                    }
                    if (refCls is object)
                    {
                        var rc = refCls.data<float>().ToArray();
                        var oc = outs[0];
                        for (int i = 0; i < Math.Min(rc.Length, oc.Length); i++)
                        {
                            var d = Math.Abs((double)rc[i] - oc[i]);
                            if (d > maxDiff) maxDiff = d; if (d > 1e-2) nBig++;
                        }
                    }
                    if (refMasks is object && outs.Length > 2)
                    {
                        var rm = refMasks.data<float>().ToArray();
                        var om = outs[2];
                        double mMax = 0;
                        for (int i = 0; i < Math.Min(rm.Length, om.Length); i++)
                        {
                            var d = Math.Abs((double)rm[i] - om[i]);
                            if (d > mMax) mMax = d;
                            if (d > maxDiff) maxDiff = d; if (d > 1e-2) nBig++;
                        }
                        Console.WriteLine("  masks path: ref= / masks路: ref=" + rm.Length + " onnx=" + om.Length + " MaxDiff=" + mMax.ToString("E2"));
                    }
                    if (refProto is object && outs.Length > 3)
                    {
                        var rp = refProto.data<float>().ToArray();
                        var op = outs[3];
                        double pMax = 0;
                        for (int i = 0; i < Math.Min(rp.Length, op.Length); i++)
                        {
                            var d = Math.Abs((double)rp[i] - op[i]);
                            if (d > pMax) pMax = d;
                            if (d > maxDiff) maxDiff = d; if (d > 1e-2) nBig++;
                        }
                        Console.WriteLine("  proto path: ref= / proto路: ref=" + rp.Length + " onnx=" + op.Length + " MaxDiff=" + pMax.ToString("E2"));
                    }
                    bool isCls = refCls is object;
                    bool ok = maxDiff < 1e-2 && rs.Length == os_.Length
                        && (isCls ? (outs.Length > 0 && refCls.numel() == outs[0].Length)
                                  : (rb.Length == ob.Length));
                    Console.WriteLine("Parity: boxes ref= / 对拍: boxes ref=" + rb.Length + "/onnx=" + ob.Length + ", scores ref=" + rs.Length + "/onnx=" + os_.Length
                        + (refMasks is object ? ", masks=" + refMasks.numel() : "") + (refProto is object ? ", proto=" + refProto.numel() : "")
                        + ", MaxDiff=" + maxDiff.ToString("E2") + " over-1e-2= / 超1e-2=" + nBig);
                    Console.WriteLine(ok ? "=== onnxexport PASS ===" : "=== onnxexport FAIL ===");
                }
            }
        }

        static void Main(string[] args)
        {
            // [Add 2026-11-10] ONNX导出对拍模式（与YoloSharpDemo同功能）：
            //   onnxexport <yoloType> <task> [imgsz] [--pt|--bin <path>]
            if (args.Length >= 1 && args[0] == "onnxexport")
            {
                RunOnnxExportTest(args.Length > 1 ? args[1] : "Yolov8",
                                  args.Length > 2 ? args[2] : "Detection",
                                  args.Length > 3 ? int.Parse(args[3]) : 640, args);
                return;
            }

            // 统一入口（与YoloSharpDemo一致）：
            //   yoloType:     Yolov5u/Yolov8/Yolov9/Yolov11/Yolov12（默认Yolov8）
            //   task:         Detection/Obb/Segmentation/Pose/Classification（默认Obb——沿用示例数据集dotav1）
            //   predictImgsz: 推理输入尺寸（letterbox等比缩放目标，需32倍数，默认640）
            //                 ——应与训练尺寸一致；训练1280的模型推理传1280，否则精度下降
            //   predictConf:  置信度阈值（默认0.45）
            //   epochs:       训练轮数（默认10；net48下训练较慢建议1-2轮验证链路）
            YoloType yoloType = args.Length > 0 ? (YoloType)Enum.Parse(typeof(YoloType), args[0], true) : YoloType.Yolov8;
            TaskType taskType = args.Length > 1 ? (TaskType)Enum.Parse(typeof(TaskType), args[1], true) : TaskType.Obb;
            int predictImgsz = args.Length > 2 ? int.Parse(args[2]) : 0;   // 0=用config.ImageSize
            float predictConf = args.Length > 3 ? float.Parse(args[3]) : 0.45f;
            int epochs = args.Length > 4 ? int.Parse(args[4]) : 10;

            // Assets shared with YoloSharpDemo (relative path: net48→Debug→bin→project→repo root)
            string assetRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"..\..\..\..\YoloSharpDemo\Assets"));
            string preTrainedModelPath = Path.Combine(assetRoot, @"PreTrainedModels\yolov8n-obb.pt");
            string predictImagePath = Path.Combine(assetRoot, @"TestImage\trucks.jpg");
            string dataRootPath = Path.Combine(assetRoot, @"DataSets\DOTAv1");
            string trainDataPath = "train.txt";
            string valDataPath = "val.txt";

            Console.WriteLine("=== YoloSharp .NET Framework 4.8 verification / 验证 ===");
            Console.WriteLine("CLR: " + Environment.Version);
            Console.WriteLine("Combo: / 组合: " + yoloType + "+" + taskType + " epochs=" + epochs);

            Mat predictImage = Cv2.ImRead(predictImagePath);
            if (predictImage.Empty())
            {
                Console.WriteLine("FAIL: test image read failed / 测试图像读取失败 " + predictImagePath);
                return;
            }
            Console.WriteLine("OpenCvSharp OK, image size: / 图像尺寸: " + predictImage.Width + "x" + predictImage.Height);

            // Create a Yolo config（与YoloSharpDemo同规则：nc跟数据集dotav1=15类；Classification按1类）
            int nc = taskType == TaskType.Classification ? 1 : 15;
            Config config = new Config
            {
                DeviceType = DeviceType.CUDA,       // 与net6.0版一致走CUDA（net48验证含GPU链路）
                ScalarType = ScalarType.BFloat16,
                RootPath = dataRootPath,
                TrainDataPath = trainDataPath,
                ValDataPath = valDataPath,
                YoloType = yoloType,
                YoloSize = yoloType == YoloType.Yolov9 ? YoloSize.s : YoloSize.n,
                TaskType = taskType,
                ImageSize = 640,
                BatchSize = 8,
                NumberClass = nc,
                Workers = 4,
                Epochs = epochs,
                LearningRate = 2e-4f,
                KeyPoint_Num = 17,
                KeyPoint_Dim = 3,
                CloseMosaic = 10,
            };

            // Create a yolo task. (invalid task×arch combos are shown a warning here)
            if (WarnInvalidCombo(config.YoloType, config.TaskType)) return;
            YoloTask yoloTask = new YoloTask(config);

            // Load pre-trained model (skip for architectures without one in Assets — trains from scratch).
            if (File.Exists(preTrainedModelPath))
                yoloTask.LoadModel(preTrainedModelPath, skipNcNotEqualLayers: true);

            // Train model
            yoloTask.Train();

            // Predict image, if the model is not trained or loaded, it will use random weight to predict.
            // [predict带参] imgsz显式传入时覆盖config（训练1280的模型推理传1280）；conf可调阈值
            if (predictImgsz >= 32) config.ImageSize = predictImgsz - predictImgsz % 32;
            Console.WriteLine("Predict " + yoloType + "+" + taskType + " imgsz=" + config.ImageSize + " conf=" + predictConf);

            System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
            List<YoloResult> predictResult = yoloTask.ImagePredict(predictImage, predictConf, 0.5f);
            sw.Stop();

            Console.WriteLine();
            Console.WriteLine("Inference result: / 推理结果: " + predictResult.Count + " targets, took / 个目标, 耗时 " + sw.ElapsedMilliseconds + " ms");

            // 输出结果（不画图，聚焦链路验证；画图部分与net6.0版相同逻辑可自行补）
            foreach (YoloResult result in predictResult)
            {
                Console.WriteLine(string.Format("LabelID:{0}, Score:{1:F1}%, CX:{2:F0}, CY:{3:F0}, W:{4:F0}, H:{5:F0}, R:{6:F3}",
                    result.ClassID, result.Score * 100, result.CenterX, result.CenterY,
                    result.Width, result.Height, result.Radian));
            }

            Console.WriteLine();
            Console.WriteLine("=== PASS: net48 + TorchSharp(netstandard2.0) + YoloSharp train+inference pipeline OK / 训练+推理链路可用 ===");
        }

        /// <summary>非法 架构×任务 组合拦截（与YoloSharpDemo同款）</summary>
        private static bool WarnInvalidCombo(YoloType yolo, TaskType task)
        {
            string why = null;
            if (yolo == YoloType.Yolov9 && task == TaskType.Pose)
                why = "官方 YOLOv9 (GELAN) 从未发布 Pose（姿态估计）变体。\n→ 请改用 Yolov8 / Yolov11 训练 Pose，或把 TaskType 改为 Detection。";
            else if (yolo == YoloType.Yolov9 && task == TaskType.Classification)
                why = "官方 YOLOv9 从未发布 Classification 变体。\n→ 请改用 Yolov8 / Yolov11，或把 TaskType 改为 Detection/Segmentation。";
            if (why == null) return false;

            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("========================================");
            Console.WriteLine("ERROR 不支持的Combo: / 组合: " + yolo + " + " + task);
            Console.WriteLine(why);
            Console.WriteLine("========================================");
            Console.ResetColor();
            return true;
        }
    }
}
