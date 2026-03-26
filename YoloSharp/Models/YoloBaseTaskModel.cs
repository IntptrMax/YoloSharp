using Data;
using ScottPlot;
using System.Data;
using System.Diagnostics;
using System.Text;
using System.Text.RegularExpressions;
using TorchSharp;
using TorchSharp.Modules;
using Utils;
using YoloSharp.Data;
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

        private int counter = 0;

        protected Config config = new Config();

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
                                if (kpt != config.KeyPointShape[0] * config.KeyPointShape[1])
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

            YoloDataset trainDataSet = new YoloDataset(config.RootPath, config.TrainDataPath, config.ImageSize, config.TaskType, config.ImageProcessType, brightness: config.Brightness, contrast: config.Contrast, saturation: config.Saturation, hue: config.Hue);
            if (trainDataSet.Count == 0)
            {
                throw new FileNotFoundException("No data found in the path: " + config.RootPath);
            }

            YoloDataLoader trainDataLoader = new YoloDataLoader(trainDataSet, config.BatchSize, num_worker: config.Workers, shuffle: true, device: config.Device);
            config.ValDataPath = string.IsNullOrEmpty(config.ValDataPath) ? config.TrainDataPath : config.ValDataPath;

            YoloDataset valDataSet = new YoloDataset(config.RootPath, config.ValDataPath, config.ImageSize, config.TaskType, ImageProcessType.Letterbox, brightness: 0f, contrast: 0f, saturation: 0f, hue: 0f);
            if (valDataSet.Count == 0)
            {
                throw new FileNotFoundException("No data found in the path: " + config.RootPath);
            }

            YoloDataLoader valDataLoader = new YoloDataLoader(valDataSet, config.BatchSize, num_worker: config.Workers, shuffle: false, device: config.Device);

            Optimizer optimizer = new SGD(yolo.parameters(), lr: config.LearningRate, momentum: 0.937f, weight_decay: 5e-4);
            lr_scheduler.LRScheduler lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max: config.Epochs);
            Console.WriteLine();

            AMPWrapper amp = new AMPWrapper(yolo, optimizer, precision: config.Dtype);
            EarlyStopping stopper = new EarlyStopping(config.Patience);
            yolo.train();
            string weightsPath = Path.Combine(config.OutputPath, "weights");
            for (int epoch = 1; epoch <= config.Epochs; epoch++)
            {
                if (!Directory.Exists(weightsPath))
                {
                    Directory.CreateDirectory(weightsPath);
                }

                Console.WriteLine(GetTrainDescription());
                Stopwatch stopwatch = Stopwatch.StartNew();
                float[] trainLoss_items = TrainEpoch(trainDataLoader, amp, epoch);
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
                for (int i = 0; i < 10; i++)
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

        internal virtual float[] TrainEpoch(YoloDataLoader trainDataLoader, AMPWrapper amp, int epoch)
        {
            using (Tqdm<Dictionary<string, Tensor>> pbar = new Tqdm<Dictionary<string, Tensor>>(trainDataLoader, total: (int)trainDataLoader.Count, barStyle: Tqdm.BarStyle.Classic, barColor: Tqdm.BarColor.White, barWidth: 10, showPartialChar: true))
            {
                yolo.train();
                Tensor loss_items = torch.empty(0);
                foreach (Dictionary<string, Tensor> data in pbar)
                {
                    using (NewDisposeScope())
                    {
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

    }

}
