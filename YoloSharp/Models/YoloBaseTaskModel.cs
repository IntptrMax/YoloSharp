using Data;
using ScottPlot;
using System.Data;
using System.Diagnostics;
using System.Text;
using System.Text.RegularExpressions;
using TorchSharp;
using TorchSharp.Modules;
using Utils;
using YoloSharp.Types;
using YoloSharp.Utils;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;

namespace YoloSharp.Models
{
    internal abstract class YoloBaseTaskModel
    {
        protected Module<Tensor, Tensor[]> yolo;
        protected Module<Tensor[], Dictionary<string, Tensor>, (Tensor loss, Tensor loss_items)> loss;

        protected Config config = new Config();

        internal YoloBaseTaskModel()
        {
            torchvision.io.DefaultImager = new torchvision.io.SkiaImager();
        }

        internal virtual void LoadModel(string path, bool skipNcNotEqualLayers = false)
        {
            Console.WriteLine("Loading model...");
            Dictionary<string, Tensor> state_dict = Lib.LoadModel(path, skipNcNotEqualLayers);
            if (state_dict.Count != yolo.state_dict().Count)
            {
                Console.WriteLine("Mismatched tensors count while loading. Model will run with random weight.");
            }
            else
            {
                torch.ScalarType modelType = state_dict.Values.First().dtype;
                yolo.to(modelType);

                List<string> skipList = new List<string>();
                if (skipNcNotEqualLayers)
                {
                    nn.Module mod = yolo.children().First().named_children().Last().module;
                    int modelCount = yolo.children().First().named_children().Count();

                    switch (mod)
                    {
                        case Modules.Modules.Classify:
                            {
                                string layerPattern = @"model\." + (modelCount - 1) + @"\.linear";
                                string key = state_dict.Keys.Where(x => Regex.IsMatch(x, layerPattern + @".+bias")).Last();
                                long nc = state_dict[key].shape[0];

                                if (nc != config.NumberClass)
                                {
                                    skipList = state_dict.Keys.Where(x => Regex.IsMatch(x, layerPattern)).ToList();
                                }
                                break;
                            }
                        case Modules.Modules.Pose:
                            {
                                string ncLayerPattern = @"model\." + (modelCount - 1) + @"\.cv3";
                                string ncKey = state_dict.Keys.Where(x => Regex.IsMatch(x, ncLayerPattern + @".+bias")).Last();
                                long nc = state_dict[ncKey].shape[0];

                                string kptLayerPattern = @"model\." + (modelCount - 1) + @"\.cv4";
                                string kptKey = state_dict.Keys.Where(x => Regex.IsMatch(x, kptLayerPattern + @".+bias")).Last();
                                long kpt = state_dict[kptKey].shape[0];

                                if (nc != config.NumberClass)
                                {
                                    skipList = state_dict.Keys.Where(x => Regex.IsMatch(x, ncLayerPattern)).ToList();
                                }
                                if (kpt != config.KeyPoint_Num * config.KeyPoint_Dim)
                                {
                                    skipList = state_dict.Keys.Where(x => Regex.IsMatch(x, kptLayerPattern)).ToList();
                                }
                                break;
                            }
                        case Modules.Modules.OBB:
                        case Modules.Modules.Segment:
                        case Modules.Modules.Yolov8Detect:
                            {
                                string layerPattern = @"model\." + (modelCount - 1) + @"\.cv3";
                                string key = state_dict.Keys.Where(x => Regex.IsMatch(x, layerPattern + @".+bias")).Last();
                                long nc = state_dict[key].shape[0];
                                if (nc != config.NumberClass)
                                {
                                    skipList = state_dict.Keys.Where(x => Regex.IsMatch(x, layerPattern)).ToList();
                                }
                                break;
                            }
                        default:
                            {
                                break;
                            }
                    }
                }

                var (miss, err) = yolo.load_state_dict(state_dict, skip: skipList);
                if (skipList.Count > 0)
                {
                    Console.WriteLine("Warning! Skipping number classes or pose reference layers. This may cause incorrect predictions when not trained again.");
                }
                yolo.to(config.Dtype);
                Console.WriteLine("Model loaded.");
            }
        }

        internal virtual void Train()
        {
            float best_fitness = float.MinValue;
            Console.WriteLine("Start Training:");
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

            //Optimizer optimizer = new SGD(yolo.parameters(), lr: config.LearningRate, momentum: 0.937f, weight_decay: 5e-4);

            double lr_fit = Math.Round(0.002 * 5 / (4 + config.NumberClass), 6);  // lr0 fit equation to 6 decimal places

            AdamW.ParamGroup biasGroup = new AdamW.ParamGroup();
            biasGroup.Parameters = yolo.named_parameters().Where(a => a.name.Contains("bias")).Select(a => a.parameter);

            AdamW.ParamGroup weightGroup = new AdamW.ParamGroup();
            weightGroup.Parameters = yolo.named_parameters().Where(a => a.name.Contains("weight")).Select(a => a.parameter);

            AdamW.ParamGroup bnGroup = new AdamW.ParamGroup();
            bnGroup.Parameters = yolo.named_parameters().Where(a => a.name.Contains("bn")).Select(a => a.parameter);

            //Optimizer optimizer = new AdamW(yolo.parameters(), lr: lr_fit, weight_decay: 5e-4f);
            Optimizer optimizer = new AdamW(new AdamW.ParamGroup[] { biasGroup, weightGroup, bnGroup }, lr: lr_fit, weight_decay: 5e-4f);

            Func<int, double> lrLambda = config.UseCosLR ?
                OneCycle(1.0, config.Lrf, config.Epochs) :
                (int epoch) =>
                    {
                        double x = (double)epoch / config.Epochs;
                        double factor = Math.Max(1 - x, 0) * (1.0 - config.Lrf) + config.Lrf;
                        return factor;
                    };

            lr_scheduler.LRScheduler lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lrLambda);
            Console.WriteLine();

            AMPWrapper amp = new AMPWrapper(yolo, optimizer, precision: config.Dtype);
            EarlyStopping stopper = new EarlyStopping(config.Patience);
            yolo.train();
            string weightsPath = Path.Combine(config.OutputPath, "weights");

            long nb = trainDataLoader.Count;
            long nw = Math.Max(config.WarmUpEpoches * nb, 100);


            for (int epoch = 1; epoch <= config.Epochs; epoch++)
            {
                if (!Directory.Exists(weightsPath))
                {
                    Directory.CreateDirectory(weightsPath);
                }

                Console.WriteLine(GetTrainDescription());
                Stopwatch stopwatch = Stopwatch.StartNew();

                trainDataSet.CloseMosaic(epoch <= config.CloseMosaic);

                float[] trainLoss_items = TrainEpoch(trainDataLoader, amp, epoch, nb, nw);
                lr_scheduler.step();
                (float[] valLoss_items, float[] metrics) = Val(valDataLoader, amp, epoch);

                float fitness = -valLoss_items.Sum();

                if (fitness > best_fitness)
                {
                    best_fitness = fitness;
                    Console.WriteLine("Get a better result, will be save to best.bin");
                    yolo.save(Path.Combine(weightsPath, "best.bin"));
                }

                bool shouldStop = stopper.ShouldStop(fitness, epoch);

                if (shouldStop)
                {
                    //Console.WriteLine($"Early stop at epoch {epoch + 1} with val loss {valLoss_items.Sum() / valDataSet.Count}");
                    break;
                }

                yolo.save(Path.Combine(weightsPath, "last.bin"));

                Console.WriteLine();
                stopwatch.Stop();
                WriteLog(epoch, stopwatch.ElapsedMilliseconds / 1000f, trainLoss_items, valLoss_items, metrics);
            }
            DrawCurve();

            Console.WriteLine("Train Done.");

            void WriteLog(int epoch, float time, float[] trainLoss_Items, float[] valLoss_Items, float[] metrics)
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
                //stringBuilder.AppendLine($"{epoch}, {time}, {trainLoss}, {valLoss}");
                stringBuilder.Append($"{epoch}, {time}, ");
                foreach (float item in trainLoss_Items)
                {
                    stringBuilder.Append($"{item / trainDataSet.Count}, ");
                }
                foreach (float item in valLoss_Items)
                {
                    stringBuilder.Append($"{item / valDataSet.Count}, ");
                }
                foreach (float item in metrics)
                {
                    stringBuilder.Append($"{item}, ");
                }
                stringBuilder.AppendLine($"{trainLoss_Items.Sum() / trainDataSet.Count}, {valLoss_Items.Sum() / valDataSet.Count}");
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
                File.WriteAllText(fileName, stringBuilder.ToString());
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
                    (int epoch) =>
                    {
                        double x = (double)epoch / config.Epochs;
                        double factor = Math.Max(1 - x, 0) * (1.0 - config.Lrf) + config.Lrf;
                        return factor;
                    };

            using (Tqdm<Dictionary<string, Tensor>> pbar = new Tqdm<Dictionary<string, Tensor>>(trainDataLoader, total: (int)trainDataLoader.Count, barStyle: Tqdm.BarStyle.Classic, barColor: Tqdm.BarColor.White, barWidth: 10, showPartialChar: true))
            {
                yolo.train();
                Tensor loss_items = torch.empty(0);
                int i = 0;
                foreach (Dictionary<string, Tensor> data in pbar)
                {
                    using (NewDisposeScope())
                    {
                        // Warm Up
                        long ni = i + nb * epoch;
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
                        Tensor[] list = amp.Forward(data["images"]);

                        (Tensor ls, Tensor ls_item) = loss.forward(list.ToArray(), data);
                        if (loss_items.NumberOfElements < 1)
                        {
                            loss_items = torch.zeros_like(ls_item);
                        }
                        loss_items = loss_items + ls_item.to(loss_items.dtype, loss_items.device);
                        loss_items = loss_items.MoveToOuterDisposeScope();
                        amp.Step(ls);
                        float[] ls_items = (ls_item).data<float>().ToArray();
                        StringBuilder stringBuilder = new StringBuilder();
                        stringBuilder.AppendFormat("{0,10}", epoch + "/" + config.Epochs);
                        foreach (float Items in ls_items)
                        {
                            stringBuilder.AppendFormat("{0,10:f3}", Items / data["images"].shape[0]);
                        }
                        stringBuilder.AppendFormat("{0,10}", data["batch_idx"].NumberOfElements);
                        stringBuilder.AppendFormat("{0,10}", data["images"].shape[2]);
                        pbar.SetDescription(stringBuilder.ToString());
                    }
                    i++;
                }
                return loss_items.@float().data<float>().ToArray();
            }
        }

        internal List<YoloResult> ImagePredict(Tensor orgImage)
        {
            return ImagePredict(orgImage, config.PredictThreshold, config.IouThreshold);
        }

        internal abstract List<YoloResult> ImagePredict(Tensor orgImage, float predictThreshold, float iouThreshold);

        internal abstract (float[] loss, float[] metrics) Val(YoloDataLoader valDataLoader, AMPWrapper amp, int epoch);

        internal abstract string GetSeperatLogHeaders();

        /// <summary>
        /// Match predictions to ground truth objects using IoU.
        /// </summary>
        /// <param name="pred_classes">Predicted class indices of shape (N,).</param>
        /// <param name="true_classes">Target class indices of shape (M,).</param>
        /// <param name="iou">An NxM tensor containing the pairwise IoU values for predictions and ground truth.</param>
        /// <param name="use_scipy">Whether to use scipy for matching (more precise).</param>
        /// <returns>Correct tensor of shape (N, 10) for 10 IoU thresholds.</returns>
        internal Tensor match_predictions(torch.Tensor pred_classes, torch.Tensor true_classes, torch.Tensor iou, bool use_scipy = false)
        {
            using (NewDisposeScope())
            using (no_grad())
            {
                Tensor iouv = torch.linspace(0.5f, 0.95f, 10, dtype: torch.ScalarType.Float32);

                // Dx10 matrix, where D - detections, 10 - IoU thresholds
                Tensor correct = torch.zeros(new long[] { pred_classes.shape[0], iouv.shape[0] }, dtype: torch.ScalarType.Bool);

                // LxD matrix where L - labels (rows), D - detections (columns)
                Tensor correct_class = true_classes[.., TensorIndex.None] == pred_classes;
                iou = iou * correct_class;  // zero out the wrong classes
                for (int i = 0; i < iouv.NumberOfElements; i++)
                {
                    float threshold = iouv[i].ToSingle();
                    Tensor matches = torch.nonzero(iou >= threshold);  // IoU > threshold and classes match
                    if (matches.shape[0] > 0)
                    {
                        if (matches.shape[0] > 1)
                        {
                            matches = matches[iou[matches[.., 0], matches[.., 1]].argsort(descending: true)];
                            matches = GetUniqueMatches(matches);
                        }

                        correct[matches[.., 1], i] = true;
                    }
                }
                return correct.to(pred_classes.device).MoveToOuterDisposeScope();
            }

            Tensor GetUniqueMatches(Tensor matches)
            {
                using (no_grad())
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

            Tensor GetUniqueByColumn(Tensor matches, int columnIndex)
            {
                using (NewDisposeScope())
                using (no_grad())
                {
                    Tensor columnValues = matches[.., columnIndex];
                    (Tensor uniqueValues, Tensor inverseIndices, _) = columnValues.unique(return_inverse: true);

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

        private static Func<int, double> OneCycle(double y1, double y2, int steps)
        {
            return (int x) =>
            {
                double progress = x * Math.PI / steps;
                double cosVal = Math.Cos(progress);
                double factor = (1 - cosVal) / 2;
                factor = Math.Max(factor, 0);          // 避免负数（实际不会）
                return factor * (y2 - y1) + y1;
            };
        }


        public static double Interp(double x, double[] xp, double[] fp)
        {
            if (xp.Length != fp.Length)
                throw new ArgumentException("xp and fp must have the same length.");
            if (xp.Length == 0)
                throw new ArgumentException("xp and fp must not be empty.");

            if (x <= xp[0]) return fp[0];
            if (x >= xp[^1]) return fp[^1];

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

    }

}
