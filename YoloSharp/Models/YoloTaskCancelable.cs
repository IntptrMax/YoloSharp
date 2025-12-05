using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using TorchSharp;
using TorchSharp.Modules;
using YoloSharp.Data;
using YoloSharp.Types;
using YoloSharp.Utils;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;
using DeviceType = YoloSharp.Types.DeviceType;
using ScalarType = YoloSharp.Types.ScalarType;

namespace YoloSharp.Models
{
    public enum TrainingPhase
    {
        EpochStart,
        Batch,
        Validation,
        Saving,
        Completed
    }

    public sealed class TrainingProgressInfo
    {
        public int Epoch { get; set; }
        public int Epochs { get; set; }
        public int Step { get; set; }
        public int Steps { get; set; }
        public float? ValLoss { get; set; }
        public TrainingPhase Phase { get; set; }
    }

    /// <summary>
    /// Yolo 训练器，增加取消和进度回调，不修改原有 YoloTask。
    /// </summary>
    public class YoloTaskCancelable
    {
        private readonly YoloBaseTaskModel yolo;

        public YoloTaskCancelable(TaskType taskType, int numberClasses, YoloType yoloType, YoloSize yoloSize, DeviceType deviceType = DeviceType.CUDA, ScalarType dtype = ScalarType.Float32, int[]? keyPointShape = null)
        {
            yolo = taskType switch
            {
                TaskType.Detection => new Detector(numberClasses, yoloType, yoloSize, deviceType, dtype),
                TaskType.Segmentation => new Segmenter(numberClasses, yoloType, yoloSize, deviceType, dtype),
                TaskType.Obb => new Obber(numberClasses, yoloType, yoloSize, deviceType, dtype),
                TaskType.Pose => new PoseDetector(numberClasses, keyPointShape!, yoloType, yoloSize, deviceType, dtype),
                TaskType.Classification => new Classifier(numberClasses, yoloType, yoloSize, deviceType, dtype),
                _ => throw new NotImplementedException("Task type not support now.")
            };
        }

        public void LoadModel(string path, bool skipNcNotEqualLayers = false)
        {
            yolo.LoadModel(path, skipNcNotEqualLayers);
        }

        public void Train(
            string rootPath,
            string trainDataPath = "",
            string valDataPath = "",
            string outputPath = "output",
            int imageSize = 640,
            int epochs = 100,
            float lr = 0.0001f,
            int batchSize = 8,
            int numWorkers = 0,
            ImageProcessType imageProcessType = ImageProcessType.Letterbox,
            CancellationToken cancellationToken = default,
            IProgress<TrainingProgressInfo>? progress = null)
        {
            Console.WriteLine("Start Training:");
            Console.WriteLine($"Yolo task type is: {yolo.TaskType}");
            Console.WriteLine($"Yolo type is: {yolo.YoloType}");
            Console.WriteLine($"Number Classes is: {yolo.SortCount}");
            Console.WriteLine("Model will be write to: " + outputPath);

            YoloDataset trainDataSet = new YoloDataset(rootPath, trainDataPath, imageSize, yolo.TaskType, imageProcessType);
            if (trainDataSet.Count == 0)
            {
                throw new FileNotFoundException("No data found in the path: " + rootPath);
            }

            DataLoader trainDataLoader = new DataLoader(trainDataSet, batchSize, num_worker: numWorkers, shuffle: true, device: yolo.Device);

            valDataPath ??= string.Empty;
            if (string.IsNullOrEmpty(valDataPath))
            {
                valDataPath = trainDataPath;
            }

            YoloDataset valDataSet = new YoloDataset(rootPath, valDataPath, imageSize, yolo.TaskType, imageProcessType);
            if (valDataSet.Count == 0)
            {
                throw new FileNotFoundException("No data found in the path: " + rootPath);
            }

            DataLoader valDataLoader = new DataLoader(valDataSet, 4, num_worker: 0, shuffle: true, device: yolo.Device);

            Optimizer optimizer = new SGD(yolo.Model.parameters(), lr: lr, momentum: 0.9, weight_decay: 5e-4);
            var lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max: 200);
            float tempLoss = float.MaxValue;
            var trainSteps = (int)trainDataLoader.Count;

            Console.WriteLine();
            yolo.Model.train(true);
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                progress?.Report(new TrainingProgressInfo
                {
                    Epoch = epoch + 1,
                    Epochs = epochs,
                    Step = 0,
                    Steps = trainSteps,
                    Phase = TrainingPhase.EpochStart
                });

                yolo.Model.train();
                int step = 0;
                foreach (var data in trainDataLoader)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    using (NewDisposeScope())
                    {
                        step++;
                        long[] indexs = data["index"].data<long>().ToArray();

                        Dictionary<string, Tensor> targets = yolo.GetTargets(indexs, trainDataSet);
                        if (targets["batch_idx"].NumberOfElements == 0)
                        {
                            continue;
                        }

                        Tensor[] list = yolo.Model.forward(targets["images"]);
                        var result = yolo.Loss.forward(list, targets);
                        Tensor ls = result.loss;
                        Tensor ls_item = result.loss_items;

                        optimizer.zero_grad();
                        ls.backward();
                        optimizer.step();

                        progress?.Report(new TrainingProgressInfo
                        {
                            Epoch = epoch + 1,
                            Epochs = epochs,
                            Step = step,
                            Steps = trainSteps,
                            Phase = TrainingPhase.Batch
                        });
                    }
                }
                lr_scheduler.step();

                cancellationToken.ThrowIfCancellationRequested();
                progress?.Report(new TrainingProgressInfo
                {
                    Epoch = epoch + 1,
                    Epochs = epochs,
                    Step = trainSteps,
                    Steps = trainSteps,
                    Phase = TrainingPhase.Validation
                });

                float valLoss = Val(valDataSet, valDataLoader, cancellationToken);

                Console.WriteLine($"Epoch {epoch + 1}, Val Loss: {valLoss}");
                progress?.Report(new TrainingProgressInfo
                {
                    Epoch = epoch + 1,
                    Epochs = epochs,
                    Step = trainSteps,
                    Steps = trainSteps,
                    ValLoss = valLoss,
                    Phase = TrainingPhase.Validation
                });

                cancellationToken.ThrowIfCancellationRequested();
                if (!Directory.Exists(outputPath))
                {
                    Directory.CreateDirectory(outputPath);
                }

                yolo.Model.save(Path.Combine(outputPath, "last.bin"));
                if (tempLoss > valLoss)
                {
                    yolo.Model.save(Path.Combine(outputPath, "best.bin"));
                    tempLoss = valLoss;
                }
                progress?.Report(new TrainingProgressInfo
                {
                    Epoch = epoch + 1,
                    Epochs = epochs,
                    Step = trainSteps,
                    Steps = trainSteps,
                    Phase = TrainingPhase.Saving
                });
                Console.WriteLine();
            }
            Console.WriteLine("Train Done.");
            progress?.Report(new TrainingProgressInfo
            {
                Epoch = epochs,
                Epochs = epochs,
                Step = trainSteps,
                Steps = trainSteps,
                Phase = TrainingPhase.Completed
            });
        }

        private float Val(YoloDataset valDataset, DataLoader valDataLoader, CancellationToken cancellationToken)
        {
            float lossValue = float.MaxValue;
            foreach (var data in valDataLoader)
            {
                cancellationToken.ThrowIfCancellationRequested();
                using (NewDisposeScope())
                using (no_grad())
                {
                    long[] indexs = data["index"].data<long>().ToArray();

                    Dictionary<string, Tensor> targets = yolo.GetTargets(indexs, valDataset);
                    if (targets["batch_idx"].NumberOfElements == 0)
                    {
                        continue;
                    }

                    Tensor[] list = yolo.Model.forward(targets["images"]);
                    var result = yolo.Loss.forward(list.ToArray(), targets);
                    Tensor ls = result.loss;
                    if (lossValue == float.MaxValue)
                    {
                        lossValue = ls.ToSingle();
                    }
                    else
                    {
                        lossValue = lossValue + ls.ToSingle();
                    }
                }
            }
            lossValue = lossValue / valDataset.Count;
            return lossValue;
        }
    }
}
