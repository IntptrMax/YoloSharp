using Data;
using ScottPlot;
using System.Data;
using System.Diagnostics;
using System.Text;
using TorchSharp;
using TorchSharp.Modules;
using Utils;
using YoloSharp.Modules;
using YoloSharp.Types;
using YoloSharp.Utils;

namespace YoloSharp.Models
{
    internal abstract class YoloBaseTaskModel
    {
        private bool _needHeadInit = true;

        protected Config config = new Config();
        protected Yolo.Yolov8 yolo;
        protected torch.nn.Module<Dictionary<string, object>, Dictionary<string, torch.Tensor>, (torch.Tensor loss, torch.Tensor loss_detach)> loss;
        protected ModelEMA ema;

        internal YoloBaseTaskModel()
        {
            torchvision.io.DefaultImager = new torchvision.io.SkiaImager();
        }

        internal virtual void LoadModel(string path, bool skipNcNotEqualLayers = false)
        {
            Console.WriteLine("Loading model...");

            Dictionary<string, torch.Tensor> sourceStateDict;

            if (path.EndsWith(".pt", StringComparison.OrdinalIgnoreCase))
            {
                var ptLoader = new YoloSharp.ModelLoader.PtReader.YoloPtLoader();
                var loadResult = ptLoader.LoadFromFile(path);
                sourceStateDict = YoloSharp.ModelLoader.PtReader.TensorBuilder.BuildStateDict(loadResult);
            }
            else
            {
                sourceStateDict = Lib.LoadModel(path, skipNcNotEqualLayers);
            }

            var modelShapes = new Dictionary<string, long[]>();
            foreach (var kv in yolo.state_dict())
            {
                modelShapes[kv.Key] = kv.Value.shape;
            }

            var matchedDict = new Dictionary<string, torch.Tensor>();
            int skippedMissing = 0;
            int skippedShape = 0;
            var skippedShapeSamples = new List<string>();

            foreach (var kv in sourceStateDict)
            {
                string name = kv.Key;
                torch.Tensor srcTensor = kv.Value;

                if (!modelShapes.TryGetValue(name, out var expectedShape))
                {
                    skippedMissing++;
                    continue;
                }

                if (!srcTensor.shape.SequenceEqual(expectedShape))
                {
                    skippedShape++;
                    if (skippedShapeSamples.Count < 10)
                    {
                        skippedShapeSamples.Add($"{name}: src [{string.Join(",", srcTensor.shape)}] " + $"vs model [{string.Join(",", expectedShape)}]");
                    }
                    continue;
                }

                matchedDict[name] = srcTensor;
            }

            Console.WriteLine($"Transferred {matchedDict.Count}/{sourceStateDict.Count} items");
            if (skippedShape > 0)
            {
                Console.WriteLine($"  Shape mismatch: {skippedShape} (will be re-initialized)");
                foreach (var s in skippedShapeSamples)
                {
                    //Console.WriteLine($"    {s}");
                }
            }
            if (skippedMissing > 0)
            {
                Console.WriteLine($"  Not in model: {skippedMissing}");
            }
            if (matchedDict.Count > 0)
            {
                yolo.load_state_dict(matchedDict, strict: false);
            }
            yolo.to(config.Dtype);

            _needHeadInit = matchedDict.Count == 0 || skippedShape > 0 || skippedMissing > 0;

            if (_needHeadInit)
            {
                Console.WriteLine("  Head bias will be re-initialized before training.");
            }

            Console.WriteLine("Model loaded.");
        }

        internal virtual void Train()
        {
            if (config.End2End)
            {
                One2one_Init();
            }
            if (_needHeadInit)
            {
                Console.WriteLine("Initializing head bias...");
                Bias_Init();
            }

            float best_fitness = float.NegativeInfinity;

            Console.WriteLine(config.ToString());
            WriteConfig();

            BaseDataset trainDataSet = this.config.TaskType == TaskType.Classification ? new ClassificationDataset(config) : new YoloDataset(config);
            if (trainDataSet.Count == 0)
            {
                throw new FileNotFoundException("No data found in the path: " + config.RootPath);
            }

            YoloDataLoader trainDataLoader = new YoloDataLoader(trainDataSet, config.BatchSize, num_worker: config.Workers, shuffle: true, device: config.Device);

            config.ValDataPath = string.IsNullOrEmpty(config.ValDataPath) ? config.TrainDataPath : config.ValDataPath;

            BaseDataset valDataSet = this.config.TaskType == TaskType.Classification ? new ClassificationDataset(this.config, true) : new YoloDataset(this.config, true);
            if (valDataSet.Count == 0)
            {
                throw new FileNotFoundException("No data found in the path: " + config.RootPath);
            }

            YoloDataLoader valDataLoader = new YoloDataLoader(valDataSet, config.BatchSize, num_worker: config.Workers, shuffle: false, device: config.Device);

            foreach (var (name, param) in yolo.named_parameters())
            {
                if (name.Contains(".dfl."))
                {
                    Console.WriteLine($"Freezing layer '{name}'");
                    param.requires_grad = false;
                }
            }

            double lr_fit = config.LearningRate;

            float weightDecay = 0.0005f;

            var allParams = yolo.named_parameters().Where(p => p.parameter.requires_grad).ToList();

            var bnWeightNames = allParams.Where(p => (p.name.Contains("bn.") || p.name.Contains("Norm") || p.name.Contains("norm")) && p.name.Contains("weight")).ToList();
            var biasNames = allParams.Where(p => p.name.Contains("bias")).ToList();

            var bnBiasNames = new HashSet<string>(bnWeightNames.Select(p => p.name));
            var biasNamesSet = new HashSet<string>(biasNames.Select(p => p.name));

            var weightNames = allParams.Where(p => !bnBiasNames.Contains(p.name) && !biasNamesSet.Contains(p.name)).ToList();

            var weightGroup = new AdamW.ParamGroup
            {
                Parameters = weightNames.Select(p => p.parameter),
                Options = new AdamW.Options() { weight_decay = weightDecay, LearningRate = lr_fit }
            };
            var bnWeightGroup = new AdamW.ParamGroup
            {
                Parameters = bnWeightNames.Select(p => p.parameter),
                Options = new AdamW.Options() { weight_decay = 0.0, LearningRate = lr_fit },
            };
            var biasGroup = new AdamW.ParamGroup
            {
                Parameters = biasNames.Select(p => p.parameter),
                Options = new AdamW.Options() { weight_decay = 0.0, LearningRate = lr_fit },
            };

            torch.optim.Optimizer optimizer = new AdamW(new AdamW.ParamGroup[] { weightGroup, bnWeightGroup, biasGroup });

            long nb_estimate = trainDataLoader.Count;
            float tau = Math.Max(100f, nb_estimate * 10f);
            ema = new ModelEMA(yolo, decay: 0.999f, tau: tau);

            Func<int, double> lrLambda = config.UseCosLR ?
                OneCycle(1.0, config.Lrf, config.Epochs) :
                LrLambda(1.0, config.Lrf, config.Epochs);

            torch.optim.lr_scheduler.LRScheduler lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lrLambda);
            Console.WriteLine();

            AMPWrapper amp = new AMPWrapper(yolo, optimizer, precision: config.Dtype);
            EarlyStopping stopper = new EarlyStopping(config.Patience);
            string weightsPath = Path.Combine(config.OutputPath, "weights");

            long nb = trainDataLoader.Count;
            // long nw = Math.Max(config.WarmUpEpoches * nb, 100);
            long nw = config.WarmUpEpoches > 0 ? Math.Max((long)Math.Round((double)config.WarmUpEpoches * nb), nb) : -1;

            long valBatches = valDataLoader.Count;

            for (int epoch = 1; epoch <= config.Epochs; epoch++)
            {
                if (!Directory.Exists(weightsPath))
                {
                    Directory.CreateDirectory(weightsPath);
                }

                Console.WriteLine(GetTrainDescription());
                Stopwatch stopwatch = Stopwatch.StartNew();

                bool closeMosaic = config.CloseMosaic > 0 && epoch > config.Epochs - config.CloseMosaic;
                trainDataSet.CloseMosaic(closeMosaic);

                float[] trainLoss_items = TrainEpoch(trainDataLoader, amp, epoch, nb, nw);
                lr_scheduler.step();

                var trainingBackup = SaveModelState(yolo);
                ema.ApplyTo(yolo);
                (float[] valLoss_items, float[] metrics) = Val(valDataLoader, amp, epoch);
                RestoreModelState(yolo, trainingBackup);

                float fitness = GetFitness(metrics);

                if (fitness > best_fitness)
                {
                    best_fitness = fitness;
                    SaveWeight(Path.Combine(weightsPath, "best.bin"), ema.GetStateDict(yolo));
                }

                bool shouldStop = stopper.ShouldStop(fitness, epoch);

                if (shouldStop)
                {
                    break;
                }

                SaveWeight(Path.Combine(weightsPath, "last.bin"));
                Console.WriteLine();
                stopwatch.Stop();

                WriteLog(epoch, stopwatch.ElapsedMilliseconds / 1000f,
                         trainLoss_items, valLoss_items, metrics, nb, valBatches);
            }
            DrawCurve();

            Console.WriteLine("Train Done.");

            void WriteLog(int epoch, float time, float[] trainLoss_Items, float[] valLoss_Items,
                          float[] metrics, long trainBatches, long valBatches)
            {
                if (!Directory.Exists(config.OutputPath))
                {
                    Directory.CreateDirectory(config.OutputPath);
                }
                string fileName = Path.Combine(config.OutputPath, "log.csv");
                StringBuilder stringBuilder = new StringBuilder();
                if (!File.Exists(fileName))
                {
                    stringBuilder.AppendLine(GetSeperatLogHeaders());
                }

                stringBuilder.Append($"{epoch}, {time}, ");
                foreach (float item in trainLoss_Items)
                {
                    stringBuilder.Append($"{item / trainBatches}, ");
                }
                foreach (float item in valLoss_Items)
                {
                    stringBuilder.Append($"{item / valBatches}, ");
                }
                foreach (float item in metrics)
                {
                    stringBuilder.Append($"{item}, ");
                }
                stringBuilder.AppendLine($"{trainLoss_Items.Sum() / trainBatches}, {valLoss_Items.Sum() / valBatches}");
                File.AppendAllText(fileName, stringBuilder.ToString());
            }

            void WriteConfig()
            {
                if (!Directory.Exists(config.OutputPath))
                {
                    Directory.CreateDirectory(config.OutputPath);
                }
                string fileName = Path.Combine(config.OutputPath, "config.txt");
                StringBuilder stringBuilder = new StringBuilder();
                stringBuilder.AppendLine("Training Settings:");
                stringBuilder.AppendLine($"Date Time: {DateTime.Now}");
                stringBuilder.AppendLine(config.ToString());
                File.AppendAllText(fileName, stringBuilder.ToString());
            }

            void DrawCurve()
            {
                string fileName = Path.Combine(config.OutputPath, "log.csv");
                Dictionary<string, List<float>> dictionary = Tools.LoadCSV(fileName);
                Multiplot multiplot = new Multiplot();

                int plotCount = dictionary.Keys.Count - 4;
                if (string.IsNullOrEmpty(dictionary.Keys.Last()))
                {
                    plotCount = plotCount - 1;
                }

                multiplot.AddPlots(plotCount);
                multiplot.Layout = new ScottPlot.MultiplotLayouts.Grid(2, plotCount / 2);

                int max_count = Math.Min(plotCount, 10);
                for (int i = 0; i < max_count; i++)
                {
                    string name = dictionary.Keys.ToArray()[i + 2];
                    List<float> x = dictionary["Epoch"];
                    List<float> y = dictionary[name];

                    Plot plot = multiplot.Subplots.GetPlot(i);
                    plot.Title(name);
                    plot.Add.Scatter(x, y);
                }

                string plotImagePath = Path.Combine(config.OutputPath, "results.png");
                multiplot.SavePng(plotImagePath, 200 * plotCount, 1200);
            }
        }

        internal virtual float[] TrainEpoch(YoloDataLoader trainDataLoader, AMPWrapper amp, int epoch, long nb, long nw)
        {
            Func<int, double> lrLambda = config.UseCosLR ?
                    OneCycle(1.0, config.Lrf, config.Epochs) :
                    LrLambda(1.0, config.Lrf, config.Epochs);

            using (Tqdm<Dictionary<string, torch.Tensor>> pbar = new Tqdm<Dictionary<string, torch.Tensor>>(trainDataLoader, total: (int)trainDataLoader.Count, barStyle: Tqdm.BarStyle.Classic, barColor: Tqdm.BarColor.White, barWidth: 10, showPartialChar: true))
            {
                yolo.train();
                torch.Tensor loss_items = torch.empty(0);
                int i = 0;
                foreach (Dictionary<string, torch.Tensor> data in pbar)
                {
                    using (torch.NewDisposeScope())
                    {
                        // Warm Up
                        long ni = i + nb * (epoch - 1);
                        if (ni <= nw)
                        {
                            double[] xi = new double[] { 0, nw };

                            int paraId = 0;
                            foreach (var x in amp.Optimizer.ParamGroups)
                            {
                                var d = x.InitialLearningRate * lrLambda(epoch);
                                x.LearningRate = Interp(ni, xi, new double[] { paraId == 0 ? config.WarmUpBiasLr : 0, d });
                                paraId++;
                            }
                        }

                        if (data["batch_idx"].NumberOfElements < 1)
                        {
                            continue;
                        }
                        yolo.train();

                        (torch.Tensor ls, torch.Tensor loss_detach) = amp.TrainStep(data["images"].to(config.Dtype), data, loss);

                        if (ema != null)
                        {
                            ema.Update(yolo);
                        }

                        if (loss_items.NumberOfElements < 1)
                        {
                            loss_items = torch.zeros_like(loss_detach);
                        }
                        loss_items = loss_items + loss_detach.to(loss_items.dtype, loss_items.device);
                        loss_items = loss_items.MoveToOuterDisposeScope();

                        float[] ls_items = (loss_detach).data<float>().ToArray();
                        StringBuilder stringBuilder = new StringBuilder();
                        stringBuilder.AppendFormat("{0,10}", epoch + "/" + config.Epochs);
                        foreach (float Items in ls_items)
                        {
                            stringBuilder.AppendFormat("{0,10:f3}", Items);
                        }
                        stringBuilder.AppendFormat("{0,10}", data["batch_idx"].NumberOfElements);
                        stringBuilder.AppendFormat("{0,10}", data["images"].shape[2]);
                        pbar.SetDescription(stringBuilder.ToString());
                    }
                    i++;
                }

                if (loss is Loss.E2EOBBLoss)
                {
                    ((Loss.E2EOBBLoss)loss).update();
                }
                return loss_items.@float().data<float>().ToArray();
            }
        }

        internal List<YoloResult> ImagePredict(torch.Tensor orgImage)
        {
            return ImagePredict(orgImage, config.PredictThreshold, config.IouThreshold);
        }

        internal abstract List<YoloResult> ImagePredict(torch.Tensor orgImage, float predictThreshold, float iouThreshold);

        internal abstract (float[] loss, float[] metrics) Val(YoloDataLoader valDataLoader, AMPWrapper amp, int epoch);

        internal abstract float GetFitness(float[] metrics);


        internal abstract string GetSeperatLogHeaders();

        /// <summary>
        /// Match predictions to ground truth objects using IoU.
        /// </summary>
        /// <param name="pred_classes">Predicted class indices of shape (N,).</param>
        /// <param name="true_classes">Target class indices of shape (M,).</param>
        /// <param name="iou">An NxM tensor containing the pairwise IoU values for predictions and ground truth.</param>
        /// <param name="use_scipy">Whether to use scipy for matching (more precise).</param>
        /// <returns>Correct tensor of shape (N, 10) for 10 IoU thresholds.</returns>
        internal torch.Tensor match_predictions(torch.Tensor pred_classes, torch.Tensor true_classes, torch.Tensor iou, bool use_scipy = false)
        {
            using (torch.NewDisposeScope())
            using (torch.no_grad())
            {
                torch.Tensor iouv = torch.linspace(0.5f, 0.95f, 10, dtype: torch.ScalarType.Float32);

                // Dx10 matrix, where D - detections, 10 - IoU thresholds
                torch.Tensor correct = torch.zeros(new long[] { pred_classes.shape[0], iouv.shape[0] }, dtype: torch.ScalarType.Bool);

                // LxD matrix where L - labels (rows), D - detections (columns)
                torch.Tensor correct_class = true_classes[torch.TensorIndex.Ellipsis, torch.TensorIndex.None] == pred_classes;
                iou = iou * correct_class;  // zero out the wrong classes
                for (int i = 0; i < iouv.NumberOfElements; i++)
                {
                    float threshold = iouv[i].ToSingle();
                    torch.Tensor matches = torch.nonzero(iou >= threshold);  // IoU > threshold and classes match
                    if (matches.shape[0] > 0)
                    {
                        if (matches.shape[0] > 1)
                        {
                            matches = matches[iou[matches[torch.TensorIndex.Ellipsis, 0], matches[torch.TensorIndex.Ellipsis, 1]].argsort(descending: true)];
                            matches = GetUniqueMatches(matches);
                        }

                        correct[matches[torch.TensorIndex.Ellipsis, 1], i] = true;
                    }
                }
                return correct.to(pred_classes.device).MoveToOuterDisposeScope();
            }

            torch.Tensor GetUniqueMatches(torch.Tensor matches)
            {
                using (torch.no_grad())
                {
                    if (matches.ndim != 2 || matches.shape[1] != 2)
                    {
                        throw new ArgumentException("matches shape must be [n, 2]");
                    }
                    matches = GetUniqueByColumn(matches, columnIndex: 1);
                    matches = GetUniqueByColumn(matches, columnIndex: 0);
                    return matches;
                }
            }

            torch.Tensor GetUniqueByColumn(torch.Tensor matches, int columnIndex)
            {
                using (torch.NewDisposeScope())
                using (torch.no_grad())
                {
                    torch.Tensor columnValues = matches[torch.TensorIndex.Ellipsis, columnIndex];
                    (torch.Tensor uniqueValues, torch.Tensor inverseIndices, _) = columnValues.unique(return_inverse: true);

                    long n = columnValues.shape[0];
                    var firstOccurrence = torch.full(new long[] { uniqueValues.shape[0] }, -1L, torch.ScalarType.Int64, device: matches.device);

                    for (long i = 0; i < n; i++)
                    {
                        long inverseIdx = inverseIndices[i].item<long>();
                        if (firstOccurrence[inverseIdx].item<long>() == -1)
                        {
                            firstOccurrence[inverseIdx] = i;
                        }
                    }

                    return (matches.index_select(0, firstOccurrence)).MoveToOuterDisposeScope();
                }
            }

        }

        internal abstract string GetValDescription();

        internal abstract string GetTrainDescription();

        internal void One2one_Init()
        {
            if (config.TaskType != TaskType.Classification)
            {
                Head.Detect head = (Head.Detect)(yolo.named_children().ToArray()[0].module.named_children().Last().module);
                head.one2one_init();
            }
        }

        internal void Bias_Init()
        {
            if (config.TaskType != TaskType.Classification)
            {
                Head.Detect head = (Head.Detect)(yolo.named_children().ToArray()[0].module.named_children().Last().module);
                head.bias_init();
            }
        }

        internal void SaveWeight(string path, Dictionary<string, torch.Tensor> stateDict = null)
        {
            var source = stateDict ?? yolo.state_dict();
            Dictionary<string, torch.Tensor> sd = new Dictionary<string, torch.Tensor>();
            foreach (var li in source)
            {
                if (!li.Key.Contains("one2one"))
                {
                    sd.Add(li.Key, li.Value);
                }
            }

            using FileStream output = File.Create(path);
            using BinaryWriter writer = new BinaryWriter(output);
            Encode(writer, sd.Count);
            foreach (KeyValuePair<string, torch.Tensor> item2 in sd)
            {
                writer.Write(item2.Key);
                item2.Value.Save(writer);
            }
        }

        private Dictionary<string, torch.Tensor> SaveModelState(torch.nn.Module model)
        {
            var backup = new Dictionary<string, torch.Tensor>();
            using (torch.no_grad())
            {
                foreach (var kv in model.state_dict())
                {
                    backup[kv.Key] = kv.Value.clone();
                }
            }
            return backup;
        }

        private void RestoreModelState(torch.nn.Module model, Dictionary<string, torch.Tensor> backup)
        {
            using (torch.no_grad())
            {
                var current = model.state_dict();
                foreach (var kv in backup)
                {
                    if (current.TryGetValue(kv.Key, out var p))
                    {
                        p.copy_(kv.Value);
                    }
                }
            }
        }

        private static Func<int, double> OneCycle(double y1, double y2, int steps)
        {
            return (int x) =>
            {
                double progress = x * Math.PI / steps;
                double cosVal = Math.Cos(progress);
                double factor = (1 - cosVal) / 2;
                factor = Math.Max(factor, 0);
                return factor * (y2 - y1) + y1;
            };
        }

        private static Func<int, double> LrLambda(double y1, double y2, int steps)
        {
            return (int epoch) =>
              {
                  double x = (double)epoch / steps;
                  double factor = Math.Max(1 - x, 0) * (y1 - y2) + y2;
                  return factor;
              };
        }

        public static double Interp(double x, double[] xp, double[] fp)
        {
            if (xp.Length != fp.Length)
                throw new ArgumentException("xp and fp must have the same length.");
            if (xp.Length == 0)
                throw new ArgumentException("xp and fp must not be empty.");

            if (x <= xp[0]) return fp[0];
            if (x >= xp[xp.Length - 1]) return fp[fp.Length - 1];

            int index = Array.BinarySearch(xp, x);
            if (index >= 0)
                return fp[index];

            index = ~index;
            double x0 = xp[index - 1];
            double x1 = xp[index];
            double y0 = fp[index - 1];
            double y1 = fp[index];

            double t = (x - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }

        private static void Encode(BinaryWriter writer, long value)
        {
            if (value < 0)
            {
                throw new NotImplementedException("LEB128 encoding of negative numbers");
            }

            long num;
            while (true)
            {
                num = value & 0x7F;
                value >>= 7;
                if (value == 0L)
                {
                    break;
                }

                writer.Write((byte)(num | 0x80));
            }

            writer.Write((byte)num);
        }

    }

}
