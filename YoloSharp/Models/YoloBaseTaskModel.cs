using System.Reflection;
using System.Text.RegularExpressions;
using TorchSharp;
using TorchSharp.Modules;
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
		protected Device device;
		protected torch.ScalarType dtype;
		protected int sortCount;
		protected YoloType yoloType;
		protected Module<Tensor, Tensor[]> yolo;
		protected Module<Tensor[], Dictionary<string, Tensor>, (Tensor loss, Tensor loss_items)> loss;
		protected TaskType taskType;
		protected int[] keyPointsShape;

        // 提供只读访问，方便外部封装器（如带取消的训练器）查询状态。
        internal Device Device => device;
        internal torch.ScalarType TorchDType => dtype;
        internal int SortCount => sortCount;
        internal YoloType YoloType => yoloType;
        internal TaskType TaskType => taskType;
        internal Module<Tensor, Tensor[]> Model => yolo;
        internal Module<Tensor[], Dictionary<string, Tensor>, (Tensor loss, Tensor loss_items)> Loss => loss;
        internal int[] KeyPointsShape => keyPointsShape;

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

				List<string> skipList = new();
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

								if (nc != sortCount)
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

								if (nc != sortCount)
								{
									skipList = state_dict.Keys.Where(x => Regex.IsMatch(x, ncLayerPattern)).ToList();
								}
								if (kpt != keyPointsShape[0] * keyPointsShape[1])
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
								if (nc != sortCount)
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
				yolo.to(dtype);
			}
		}

		internal virtual void Train(string rootPath, string trainDataPath = "", string valDataPath = "", string outputPath = "output", int imageSize = 640, int epochs = 100, float lr = 0.0001f, int batchSize = 8, int numWorkers = 0, ImageProcessType imageProcessType = ImageProcessType.Letterbox)
		{
			Console.WriteLine("Start Training:");
			Console.WriteLine($"Yolo task type is: {taskType}");
			Console.WriteLine($"Yolo type is: {yoloType}");
			Console.WriteLine($"Number Classes is: {sortCount}");
			Console.WriteLine("Model will be write to: " + outputPath);

			YoloDataset trainDataSet = new YoloDataset(rootPath, trainDataPath, imageSize, this.taskType, imageProcessType);
			if (trainDataSet.Count == 0)
			{
				throw new FileNotFoundException("No data found in the path: " + rootPath);
			}

			DataLoader trainDataLoader = new DataLoader(trainDataSet, batchSize, num_worker: numWorkers, shuffle: true, device: device);

			valDataPath = string.IsNullOrEmpty(valDataPath) ? trainDataPath : valDataPath;

			YoloDataset valDataSet = new YoloDataset(rootPath, valDataPath, imageSize, this.taskType, imageProcessType);
			if (valDataSet.Count == 0)
			{
				throw new FileNotFoundException("No data found in the path: " + rootPath);
			}

			DataLoader valDataLoader = new DataLoader(valDataSet, 4, num_worker: 0, shuffle: true, device: device);

			Optimizer optimizer = new SGD(yolo.parameters(), lr: lr, momentum: 0.9, weight_decay: 5e-4);
			var lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max: 200);
			float tempLoss = float.MaxValue;

			Console.WriteLine();
			yolo.train(true);
			for (int epoch = 0; epoch < epochs; epoch++)
			{
				yolo.train();
				int step = 0;
				foreach (var data in trainDataLoader)
				{
					using (NewDisposeScope())
					{
						step++;
						long[] indexs = data["index"].data<long>().ToArray();

						Dictionary<string, Tensor> targets = GetTargets(indexs, trainDataSet);
						if (targets["batch_idx"].NumberOfElements == 0)
						{
							continue;
						}

						Tensor[] list = yolo.forward(targets["images"]);
						(Tensor ls, Tensor ls_item) = loss.forward(list, targets);

						optimizer.zero_grad();
						ls.backward();
						optimizer.step();
						var cursorPosition = Console.GetCursorPosition();
						Console.SetCursorPosition(0, cursorPosition.Top);
						Console.Write($"Process: Epoch {epoch}, Step/Total Step:{step}/{trainDataLoader.Count}");
					}
				}
				lr_scheduler.step();
				Console.WriteLine();
				Console.Write("Do val now... ");
				float valLoss = Val(valDataSet, valDataLoader);
				Console.WriteLine($"Epoch {epoch}, Val Loss: {valLoss}");
				if (!Directory.Exists(outputPath))
				{
					Directory.CreateDirectory(outputPath);
				}
				yolo.save(Path.Combine(outputPath, "last.bin"));
				if (tempLoss > valLoss)
				{
					yolo.save(Path.Combine(outputPath, "best.bin"));
					tempLoss = valLoss;
				}
				Console.WriteLine();
			}
			Console.WriteLine("Train Done.");
		}

		internal virtual float Val(YoloDataset valDataset, DataLoader valDataLoader)
		{
			float lossValue = float.MaxValue;
			foreach (var data in valDataLoader)
			{
				using (NewDisposeScope())
				using (no_grad())
				{
					long[] indexs = data["index"].data<long>().ToArray();

					Dictionary<string, Tensor> targets = GetTargets(indexs, valDataset);
					if (targets["batch_idx"].NumberOfElements == 0)
					{
						continue;
					}

					Tensor[] list = yolo.forward(targets["images"]);
					var (ls, ls_item) = loss.forward(list.ToArray(), targets);
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

		internal abstract Dictionary<string, Tensor> GetTargets(long[] indexs, YoloDataset dataset);

		internal abstract List<YoloResult> ImagePredict(Tensor orgImage, float PredictThreshold = 0.25f, float IouThreshold = 0.5f);



	}
}
