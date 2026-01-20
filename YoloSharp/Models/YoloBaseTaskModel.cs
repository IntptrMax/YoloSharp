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
		protected Device device;
		protected torch.ScalarType dtype;
		protected int sortCount;
		protected YoloType yoloType;
		protected Module<Tensor, Tensor[]> yolo;
		protected Module<Tensor[], Dictionary<string, Tensor>, (Tensor loss, Tensor loss_items)> loss;
		protected TaskType taskType;
		protected int[] keyPointsShape;

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

		internal virtual void Train(string rootPath, string trainDataPath = "", string valDataPath = "", string outputPath = "output", int imageSize = 640, int epochs = 100, float lr = 1e-4f, int batchSize = 8, int numWorkers = 0, ImageProcessType imageProcessType = ImageProcessType.Letterbox)
		{
			Console.WriteLine("Start Training:");
			Console.WriteLine($"Yolo task type is: {taskType}");
			Console.WriteLine($"Yolo type is: {yoloType}");
			Console.WriteLine($"Device type is: {device}");
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

			DataLoader valDataLoader = new DataLoader(valDataSet, batchSize, num_worker: 0, shuffle: true, device: device);

			Optimizer optimizer = new SGD(yolo.parameters(), lr: lr, momentum: 0.937f, weight_decay: 5e-4);
			lr_scheduler.LRScheduler lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max: 200);
			float tempLoss = float.MaxValue;

			Console.WriteLine();
			yolo.train(true);

			AMPWrapper amp = new AMPWrapper(yolo, optimizer, precision: dtype);

			for (int epoch = 1; epoch <= epochs; epoch++)
			{
				yolo.train();
				int step = 0;
				float trainLoss = 0;
				Console.WriteLine();
				using (Tqdm<Dictionary<string, Tensor>> pbar = new Tqdm<Dictionary<string, Tensor>>(trainDataLoader, total: (int)trainDataLoader.Count, barStyle: Tqdm.BarStyle.Classic, barColor: Tqdm.BarColor.White, barWidth: 30, showPartialChar: true))
				{
					foreach (Dictionary<string, Tensor> data in pbar)
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
							long c = targets["images"].shape[0];

							//Tensor[] list = yolo.forward(targets["images"].to(dtype));
							//(Tensor ls, Tensor ls_item) = loss.forward(list, targets);

							//optimizer.zero_grad();
							//ls.backward();
							//optimizer.step();
							//trainLoss += ls.ToSingle();

							Tensor[] list = amp.Forward(targets["images"].to(dtype));
							(Tensor ls, Tensor ls_item) = loss.forward(list, targets);

							amp.Step(ls);

							pbar.SetDescription($"Train Epoch:{epoch,4}/{epochs,4}, Loss:{ls.ToSingle() / c,6:F6} ");
						}
					}
				}
				trainLoss = trainLoss / trainDataSet.Count;
				lr_scheduler.step();
				Console.Write("Do val now... ");
				float valLoss = Val(valDataSet, valDataLoader, amp);
				Console.WriteLine($"Epoch {epoch}/{epochs}, Train Loss:{trainLoss}, Val Loss: {valLoss}");
				if (!Directory.Exists(outputPath))
				{
					Directory.CreateDirectory(outputPath);
				}
				yolo.save(Path.Combine(outputPath, "last.bin"));
				if (tempLoss > valLoss)
				{
					Console.WriteLine("Get a better result, will be save to best.bin");
					yolo.save(Path.Combine(outputPath, "best.bin"));
					tempLoss = valLoss;
				}
				Console.WriteLine();
			}
			Console.WriteLine("Train Done.");
		}

		internal virtual float Val(YoloDataset valDataset, DataLoader valDataLoader, AMPWrapper amp)
		{
			float lossValue = float.MaxValue;
			int step = 0;
			using (Tqdm<Dictionary<string, Tensor>> pbar = new Tqdm<Dictionary<string, Tensor>>(valDataLoader, total: (int)valDataLoader.Count, barStyle: Tqdm.BarStyle.Classic, barColor: Tqdm.BarColor.White, barWidth: 30, showPartialChar: true))
			{
				foreach (Dictionary<string, Tensor> data in pbar)
				{
					step++;
					using (NewDisposeScope())
					using (no_grad())
					{
						long[] indexs = data["index"].data<long>().ToArray();

						Dictionary<string, Tensor> targets = GetTargets(indexs, valDataset);
						if (targets["batch_idx"].NumberOfElements == 0)
						{
							continue;
						}
						Tensor[] list = amp.Evaluate(targets["images"].to(dtype));
						//Tensor[] list = yolo.forward(targets["images"].to(dtype));
						var (ls, ls_item) = loss.forward(list.ToArray(), targets);
						if (lossValue == float.MaxValue)
						{
							lossValue = ls.ToSingle();
						}
						else
						{
							lossValue = lossValue + ls.ToSingle();
						}
						pbar.SetDescription($"Val Loss:{ls.ToSingle() / list[0].shape[0],6:F6} ");
					}
				}
				lossValue = lossValue / valDataset.Count;
				return lossValue;

			}
		}

		internal abstract Dictionary<string, Tensor> GetTargets(long[] indexs, YoloDataset dataset);

		internal abstract List<YoloResult> ImagePredict(Tensor orgImage, float PredictThreshold = 0.25f, float IouThreshold = 0.5f);



	}
}
