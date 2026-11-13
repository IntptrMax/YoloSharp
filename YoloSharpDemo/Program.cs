using Data;
using OpenCvSharp;
using YoloSharp.Models;
using YoloSharp.Types;

namespace YoloSharpDemo
{
    internal class Program
    {
        /// <summary>无效任务×架构组合→弹窗说明原因（Windows），返回true=无效已提示</summary>
        static void RunOnnxExportTest(string yoloTypeName, string taskName, int imgsz, string[] args = null)
        {
            Console.WriteLine($"=== OnnxExport parity: / 对拍: {yoloTypeName}+{taskName} imgsz={imgsz} ===");
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

            // 建网络并取内部yolo模块
            var yoloTask = new YoloTask(config);
            var yolo = YoloSharp.OnnxExport.OnnxExporter.GetYoloModule(yoloTask);
            if (args.Length > 5 && args[4] == "--pt" && System.IO.File.Exists(args[5]))
            {
                yoloTask.LoadModel(args[5], skipNcNotEqualLayers: true);   // load real weights for COCO parity / 加载真权重对拍
                Console.WriteLine($"[PtLoad] {args[5]}");
            }
            else if (args.Length > 5 && args[4] == "--bin" && System.IO.File.Exists(args[5]))
            {
                yoloTask.LoadModel(args[5], skipNcNotEqualLayers: true);   // load bin weights / 加载bin权重
                Console.WriteLine($"[BinLoad] {args[5]}");
            }
            yolo.eval();

            // TorchSharp基准输出（boxes分支 = 推理输出）
            var input = TorchSharp.torch.zeros(1, 3, imgsz, imgsz);
            TorchSharp.torch.Tensor refBoxes, refScores, refMasks = null, refProto = null, refCls = null;
            using (TorchSharp.torch.no_grad())
            {
                input = TorchSharp.torch.randn(1, 3, imgsz, imgsz);
                var fw = yolo.forward(input);
                var preds = (System.Collections.Generic.Dictionary<string, object>)fw?.Item2;
                var one2many = preds.ContainsKey("one2many") ? (System.Collections.Generic.Dictionary<string, object>)preds["one2many"] : preds;
                refBoxes = one2many.ContainsKey("boxes") ? ((TorchSharp.torch.Tensor)one2many["boxes"]).detach().cpu().to_type(TorchSharp.torch.ScalarType.Float32) : (TorchSharp.torch.Tensor)null;
                refScores = one2many.ContainsKey("scores") ? ((TorchSharp.torch.Tensor)one2many["scores"]).detach().cpu().to_type(TorchSharp.torch.ScalarType.Float32) : (TorchSharp.torch.Tensor)null;
                refMasks = one2many.ContainsKey("mask_coefficient") ? ((TorchSharp.torch.Tensor)one2many["mask_coefficient"]).detach().cpu().to_type(TorchSharp.torch.ScalarType.Float32) : (TorchSharp.torch.Tensor)null;
                refProto = one2many.ContainsKey("proto") ? ((TorchSharp.torch.Tensor)one2many["proto"]).detach().cpu().to_type(TorchSharp.torch.ScalarType.Float32) : (TorchSharp.torch.Tensor)null;
                if (preds.TryGetValue("cls", out var clsV)) refCls = clsV as TorchSharp.torch.Tensor;
            }
            Console.WriteLine($"TorchSharp head outputs: boxes= / 头输出: boxes={(refBoxes is object ? string.Join("x", refBoxes.shape) : "-")} scores={(refScores is object ? string.Join("x", refScores.shape) : "-")}");

            // 导出
            string onnxPath = $"export_{yoloTypeName}_{taskName}_{imgsz}.onnx";
            // [Fix 2026-11-13] dynamicBatch:true — the parity demo must exercise the SAME deployment
            // export as training's best.onnx (single dynamic file). The old static call left batch=1
            // and mismatched what users deploy. / 对拍demo与训练产物同款动态导出
            YoloSharp.OnnxExport.OnnxExporter.Export(yolo, new long[] { 1, 3, imgsz, imgsz }, onnxPath, dynamicBatch: true);
            Console.WriteLine($"Export OK: / 导出OK: {onnxPath} ({new System.IO.FileInfo(onnxPath).Length / 1024}KB)");

            // ONNX Runtime跑（单输出output0 = boxes‖scores‖extras 通道拼接；seg另有output1=proto）
            using (var session = new Microsoft.ML.OnnxRuntime.InferenceSession(onnxPath))
            {
                var floatArr = input.detach().cpu().to_type(TorchSharp.torch.ScalarType.Float32).data<float>().ToArray();
                // TorchSharp NCHW→ONNX同名输入
                var tensor = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(floatArr, new[] { 1, 3, imgsz, imgsz });
                var inputName = session.InputMetadata.First().Key;
                using var results = session.Run(new[] { Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor(inputName, tensor) });

                // [Fix 2026-11-13] Single-output parity: output0 = [1, 64+nc(+extras), A] with channels
                // 0..63 = boxes DFL, 64.. = scores (sigmoid applied). Slice onnx output0 by channel
                // and compare against the TorchSharp head tensors. (Old code compared outs[0]/outs[1]
                // as separate boxes/scores outputs — the multi-head era layout.)
                // 单输出对拍：按通道切片后与TorchSharp头输出逐点比对
                var outs = results.Select(r => r.AsTensor<float>().ToArray()).ToArray();
                var rb = refBoxes is object ? refBoxes.data<float>().ToArray() : new float[0];
                var rs = refScores is object ? refScores.data<float>().ToArray() : new float[0];
                double maxDiff = 0; long nBig = 0;
                var o0 = outs[0];   // single output0 / 单输出
                long A = refBoxes is object ? refBoxes.shape[refBoxes.shape.Length - 1] : 0;
                long nBoxCh = refBoxes is object ? refBoxes.shape[1] : 0;    // 64
                long nClsCh = refScores is object ? refScores.shape[1] : 0;  // nc
                // boxes: channels 0..nBoxCh — both tensors are [1, C, A] contiguous: (0,ch,a) → ch*A+a
                // boxes段：两侧都是[1,C,A]连续布局，展平索引=ch*A+a
                for (int ch = 0; ch < nBoxCh; ch++)
                    for (int a = 0; a < A; a++)
                    {
                        var d = Math.Abs((double)rb[(int)(ch * A + a)] - o0[(int)(ch * A + a)]);
                        if (d > maxDiff) maxDiff = d; if (d > 1e-2) nBig++;
                    }
                // scores: channels nBoxCh..nBoxCh+nClsCh — reference is raw logits; the exported
                // graph applies sigmoid (deployment convention), so sigmoid the reference too.
                // scores段：参考侧是logits，导出图已过sigmoid（部署约定）——参考侧补sigmoid再比
                for (int ch = 0; ch < nClsCh; ch++)
                    for (int a = 0; a < A; a++)
                    {
                        float refV = 1f / (1f + (float)Math.Exp(-rs[(int)(ch * A + a)]));   // sigmoid
                        var d = Math.Abs((double)refV - o0[(int)((nBoxCh + ch) * A + a)]);
                        if (d > maxDiff) maxDiff = d; if (d > 1e-2) nBig++;
                    }
                // Classification: 输出0=cls（rank2，直接逐点）
                if (refCls is object)
                {
                    var rc = refCls.data<float>().ToArray();
                    for (int i = 0; i < Math.Min(rc.Length, o0.Length); i++)
                    {
                        var d = Math.Abs((double)rc[i] - o0[i]);
                        if (d > maxDiff) maxDiff = d; if (d > 1e-2) nBig++;
                    }
                }
                // seg: masks拼在output0通道尾部(64+nc..)，proto=outs[1] — [1,C,A]布局索引=ch*A+a
                if (refMasks is object && nClsCh > 0)
                {
                    var rm = refMasks.data<float>().ToArray();
                    long nMaskCh = refMasks.shape[1];
                    double mMax = 0;
                    for (int ch = 0; ch < nMaskCh; ch++)
                        for (int a = 0; a < A; a++)
                        {
                            var d = Math.Abs((double)rm[(int)(ch * A + a)] - o0[(int)((nBoxCh + nClsCh + ch) * A + a)]);
                            if (d > mMax) mMax = d;
                            if (d > maxDiff) maxDiff = d; if (d > 1e-2) nBig++;
                        }
                    Console.WriteLine($"  masks path: ref={rm.Length} MaxDiff={mMax:E2}");
                }
                if (refProto is object && outs.Length > 1)
                {
                    var rp = refProto.data<float>().ToArray();
                    var op = outs[1];   // output1 = proto / proto走第二输出
                    double pMax = 0;
                    for (int i = 0; i < Math.Min(rp.Length, op.Length); i++)
                    {
                        var d = Math.Abs((double)rp[i] - op[i]);
                        if (d > pMax) pMax = d;
                        if (d > maxDiff) maxDiff = d; if (d > 1e-2) nBig++;
                    }
                    Console.WriteLine($"  proto path: ref={rp.Length} onnx={op.Length} MaxDiff={pMax:E2}");
                }
                bool isCls = refCls is object;
                // extras: obb angle(1) / pose kpts(nk*nd) / seg masks(nm) ride after cls channels
                long extraCh = (o0.Length / Math.Max(1, A)) - nBoxCh - nClsCh;   // from actual output
                bool ok = maxDiff < 1e-2
                    && (isCls ? (outs.Length > 0 && refCls.numel() == outs[0].Length)
                              : (o0.Length == (nBoxCh + nClsCh + extraCh) * A && extraCh >= 0));
                Console.WriteLine($"Parity: output0=[1,{o0.Length / Math.Max(1, A)},A={A}] boxesCh={nBoxCh} clsCh={nClsCh} extraCh={extraCh}, MaxDiff={maxDiff:E2} over-1e-2={nBig}");
                Console.WriteLine(ok ? "=== onnxexport PASS ===" : "=== onnxexport FAIL ===");
            }
        }

        static bool WarnInvalidCombo(YoloType yolo, TaskType task)
        {
            string? why = (yolo, task) switch
            {
                (YoloType.Yolov9, TaskType.Obb) =>
                    "Official YOLOv9 (GELAN) never shipped an OBB (rotated box) variant. / 官方从未发布OBB变体。\n" +
                    "→ 请改用 Yolov8 / Yolov11 训练 OBB，或把 TaskType 改为 Detection。",
                (YoloType.Yolov9, TaskType.Pose) =>
                    "Official YOLOv9 (GELAN) never shipped a Pose variant. / 官方从未发布Pose变体。\n" +
                    "→ 请改用 Yolov8 / Yolov11 训练 Pose，或把 TaskType 改为 Detection。",
                (YoloType.Yolov9, TaskType.Classification) =>
                    "YOLOv9 has a classification variant (yolov9-cls), not yet ported to C#. / 分类变体尚未实现。\n" +
                    "→ 请暂用 Yolov8 / Yolov11 训练分类；v9-cls 头欢迎社区补充。",
                _ => null,
            };
            if (why == null) return false;

            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("========================================");
            Console.WriteLine($"ERROR 不支持的Combo: / 组合: {yolo} + {task}");
            Console.WriteLine(why);
            Console.WriteLine("========================================");
            Console.ResetColor();
            return true;
        }

        static void Main(string[] args)
        {
            // ONNX export parity mode: onnxexport <yoloType> <task> [imgsz] [--pt|--bin <path>] — build network → export onnx → point-wise parity vs TorchSharp forward
            // ONNX导出对拍模式——构建网络→导出onnx→与TorchSharp前向逐点对拍
            if (args.Length >= 1 && args[0] == "onnxexport")
            {
                RunOnnxExportTest(args.Length > 1 ? args[1] : "Yolov8",
                                  args.Length > 2 ? args[2] : "Detection",
                                  args.Length > 3 ? int.Parse(args[3]) : 640, args);
                return;
            }

            // 统一入口（v8/v9/v11/v12同一流程，仅Config不同）：
            //   YoloSharpDemo [yoloType] [task] [predictImgsz] [predictConf] [epochs]
            //   yoloType:     Yolov5u/Yolov8/Yolov9/Yolov11/Yolov12（默认Yolov8）
            //   task:         Detection/Obb/Segmentation/Pose/Classification（默认Obb——沿用示例数据集dotav1）
            //   predictImgsz: 推理输入尺寸（letterbox等比缩放目标，需32倍数，默认640）
            //                 ——应与训练尺寸一致；训练1280的模型推理传1280，否则精度下降
            //   predictConf:  置信度阈值（默认0.45）
            //   epochs:       训练轮数（默认10；dotav1-obb/v9示例均适用）
            // 例：
            //   YoloSharpDemo                              → v8+Obb 默认（训练+推理）
            //   YoloSharpDemo Yolov9 Detection             → v9 从头训练dotav1+推理
            //   YoloSharpDemo Yolov8 Obb 1280 0.3          → v8 训练后按1280推理、conf=0.3
            //   YoloSharpDemo Yolov9 Segmentation          → v9-seg（需自备权重路径见下）
            YoloType yoloType = args.Length > 0 ? (YoloType)Enum.Parse(typeof(YoloType), args[0], true) : YoloType.Yolov8;
            TaskType taskType = args.Length > 1 ? (TaskType)Enum.Parse(typeof(TaskType), args[1], true) : TaskType.Obb;
            int predictImgsz = args.Length > 2 ? int.Parse(args[2]) : 0;   // 0=用config.ImageSize
            float predictConf = args.Length > 3 ? float.Parse(args[3]) : 0.45f;
            int epochs = args.Length > 4 ? int.Parse(args[4]) : 10;

            string preTrainedModelPath = @"..\..\..\Assets\PreTrainedModels\yolov8n-obb.pt"; // Pretrained model path.
            string predictImagePath = @"..\..\..\Assets\TestImage\trucks.jpg";

            string dataRootPath = @"..\..\..\Assets\DataSets\dotav1";

            string trainDataPath = @"train.txt";
            string valDataPath = @"val.txt";

            Mat predictImage = Cv2.ImRead(predictImagePath);

            // Create a Yolo config（任务相关参数按task自动适配；v9等无预训练权重的架构从头训练）
            // nc: 示例数据集dotav1为15类（train.txt/val.txt标签0..14）；Classification任务无框标签按1类
            int nc = taskType == TaskType.Classification ? 1 : 15;
            Config config = new Config
            {
                DeviceType = DeviceType.CUDA,
                ScalarType = ScalarType.BFloat16,
                RootPath = dataRootPath,
                TrainDataPath = trainDataPath,
                ValDataPath = valDataPath,
                YoloType = yoloType,
                YoloSize = yoloType == YoloType.Yolov9 ? YoloSize.s : YoloSize.n,   // v9官方最小为s
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

            // Create a yolo task. (invalid task×arch combos are shown a warning popup here)
            if (WarnInvalidCombo(config.YoloType, config.TaskType)) return;
            YoloTask yoloTask = new YoloTask(config);

            // Load pre-trained model (skip for architectures without one in Assets — trains from scratch).
            if (System.IO.File.Exists(preTrainedModelPath))
                yoloTask.LoadModel(preTrainedModelPath, skipNcNotEqualLayers: true);

            // Train model
            yoloTask.Train();

            // Predict image, if the model is not trained or loaded, it will use random weight to predict.
            // [predict带参] imgsz显式传入时覆盖config（训练1280的模型推理传1280）；conf可调阈值
            if (predictImgsz >= 32) config.ImageSize = predictImgsz - predictImgsz % 32;
            Console.WriteLine($"Predict {yoloType}+{taskType} imgsz={config.ImageSize} conf={predictConf}");
            List<YoloResult> predictResult = yoloTask.ImagePredict(predictImage, predictConf, 0.5f);

            // Rand for mask color.
            Random rand = new Random(1024);

            // Draw results
            foreach (YoloResult result in predictResult)
            {
                float[] cxcywhr = new float[] { result.CenterX, result.CenterY, result.Width, result.Height, result.Radian };
                float[] points = cxcywhr2xyxyxyxy(cxcywhr);

                OpenCvSharp.Point[] pts = new OpenCvSharp.Point[4]
                {
                    new OpenCvSharp.Point(points[0], points[1]),
                    new OpenCvSharp.Point(points[2], points[3]),
                    new OpenCvSharp.Point(points[4], points[5]),
                    new OpenCvSharp.Point(points[6], points[7]),
                };
                Cv2.Polylines(predictImage, new OpenCvSharp.Point[][] { pts }, true, OpenCvSharp.Scalar.Red, 2);
                string label = string.Format("{0}:{1:F1}%", result.ClassID, result.Score * 100);

                OpenCvSharp.Size textSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, 0.5, 1, out int baseline);
                Cv2.Rectangle(predictImage, new Rect(new OpenCvSharp.Point(result.X, result.Y - textSize.Height - baseline), new OpenCvSharp.Size(textSize.Width, textSize.Height + baseline)), OpenCvSharp.Scalar.White, Cv2.FILLED);
                Cv2.PutText(predictImage, label, new OpenCvSharp.Point(result.X, result.Y - baseline), HersheyFonts.HersheySimplex, 0.5, OpenCvSharp.Scalar.Black, 1);
                Console.WriteLine(string.Format("LabelID:{0}, Score:{1:F1}%, CenterX:{2}, CenterY:{3}, Width:{4}, Height:{5}, R:{6:F3} rnd", result.ClassID, result.Score * 100, result.CenterX, result.CenterY, result.Width, result.Height, result.Radian));

                // Draw mask
                if (result.Mask is not null)
                {
                    Mat maskMat = Mat.FromArray<byte>(result.Mask);
                    maskMat = maskMat * 255;
                    maskMat = maskMat.Transpose();

                    // Create random color
                    int R = rand.Next(0, 255);
                    int G = rand.Next(0, 255);
                    int B = rand.Next(0, 255);
                    OpenCvSharp.Scalar color = new OpenCvSharp.Scalar(R, G, B, 200);
                    Mat backColor = new Mat(maskMat.Rows, maskMat.Cols, MatType.CV_8UC3, color);
                    Cv2.Add(predictImage, backColor, predictImage, maskMat);
                }

                if (result.KeyPoints is not null)
                {
                    foreach (YoloSharp.Types.KeyPoint keyPoint in result.KeyPoints)
                    {
                        Cv2.Circle(predictImage, (int)keyPoint.X, (int)keyPoint.Y, 3, OpenCvSharp.Scalar.Blue);
                    }
                }
            }
            predictImage.SaveImage("result.jpg");

            Console.WriteLine();
            Console.WriteLine("Image Predict done");
        }

        /// <summary>
        /// CenterX, CenterY, Width, Height, r → P0(x0, y0), P1(x1, y1), P2(x2, y2), P3(x3, y3)
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        private static float[] cxcywhr2xyxyxyxy(float[] x)
        {
            float cx = x[0];
            float cy = x[1];
            float w = x[2];
            float h = x[3];
            float r = x[4];
            float cosR = (float)Math.Cos(r);
            float sinR = (float)Math.Sin(r);
            float wHalf = w / 2;
            float hHalf = h / 2;
            return new float[]
            {
                cx - wHalf * cosR + hHalf * sinR,
                cy - wHalf * sinR - hHalf * cosR,
                cx + wHalf * cosR + hHalf * sinR,
                cy + wHalf * sinR - hHalf * cosR,
                cx + wHalf * cosR - hHalf * sinR,
                cy + wHalf * sinR + hHalf * cosR,
                cx - wHalf * cosR - hHalf * sinR,
                cy - wHalf * sinR + hHalf * cosR,
            };
        }

        /// <summary>YOLOv9训练验证：GELAN骨架从头训练（Detection）dotav1→推理trucks.jpg</summary>
    }
}

