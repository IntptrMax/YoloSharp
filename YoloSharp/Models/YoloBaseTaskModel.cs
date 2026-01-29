using Data;
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
		protected Device device;
		protected torch.ScalarType dtype;
		protected int sortCount;
		protected YoloType yoloType;
		protected Module<Tensor, Tensor[]> yolo;
		protected Module<Tensor[], Dictionary<string, Tensor>, (Tensor loss, Tensor loss_items)> loss;
		protected TaskType taskType;
		protected int[] keyPointsShape;
		protected YoloSize yoloSize;

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

			WriteSettings();

			YoloDataset trainDataSet = new YoloDataset(rootPath, trainDataPath, imageSize, this.taskType, imageProcessType);
			if (trainDataSet.Count == 0)
			{
				throw new FileNotFoundException("No data found in the path: " + rootPath);
			}

			YoloDataLoader trainDataLoader = new YoloDataLoader(trainDataSet, batchSize, num_worker: numWorkers, shuffle: true, device: device);
			valDataPath = string.IsNullOrEmpty(valDataPath) ? trainDataPath : valDataPath;

			YoloDataset valDataSet = new YoloDataset(rootPath, valDataPath, imageSize, this.taskType, imageProcessType);
			if (valDataSet.Count == 0)
			{
				throw new FileNotFoundException("No data found in the path: " + rootPath);
			}

			YoloDataLoader valDataLoader = new YoloDataLoader(valDataSet, batchSize, num_worker: numWorkers, shuffle: false, device: device);

			Optimizer optimizer = new SGD(yolo.parameters(), lr: lr, momentum: 0.937f, weight_decay: 5e-4);
			lr_scheduler.LRScheduler lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max: 200);
			float tempLoss = float.MaxValue;
			Console.WriteLine();

			AMPWrapper amp = new AMPWrapper(yolo, optimizer, precision: dtype);
			yolo.train();
			string weightsPath = Path.Combine(outputPath, "weights");
			for (int epoch = 1; epoch <= epochs; epoch++)
			{
				Console.WriteLine();
				switch (taskType)
				{
					case TaskType.Detection:
					case TaskType.Obb:
						Console.WriteLine("{0,10}{1,10}{2,10}{3,10}{4,11}{5,6}", "Epoch", "box_loss", "cls_loss", "dfl_loss", "Instances", "Size");
						break;
					case TaskType.Segmentation:
						Console.WriteLine("{0,10}{1,10}{2,10}{3,10}{4,10}{5,11}{6,6}", "Epoch", "box_loss", "seg_loss", "cls_loss", "dfl_loss", "Instances", "Size");
						break;
					case TaskType.Pose:
						Console.WriteLine("{0,10}{1,10}{2,10}{3,10}{4,10}{5,10}{6,11}{7,6}", "Epoch", "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss", "Instances", "Size");
						break;
					case TaskType.Classification:
						Console.WriteLine("{0,10}{1,10}{2,11}{3,6}", "Epoch", "loss", "Instances", "Size");
						break;
					default:
						throw new NotImplementedException("Not implemented task type: " + taskType);
				}
				Stopwatch stopwatch = Stopwatch.StartNew();
				float[] trainLoss_items = TrainEpoch(trainDataLoader, amp, epoch);
				lr_scheduler.step();
				float[] valLoss_items = Val(valDataLoader, amp, epoch);
				Console.WriteLine($"Epoch {epoch}/{epochs}, Train Loss:{trainLoss_items.Sum() / trainDataSet.Count}, Val Loss: {valLoss_items.Sum() / valDataSet.Count}");

				if (!Directory.Exists(weightsPath))
				{
					Directory.CreateDirectory(weightsPath);
				}

				yolo.save(Path.Combine(weightsPath, "last.bin"));
				if (tempLoss > valLoss_items.Sum())
				{
					Console.WriteLine("Get a better result, will be save to best.bin");
					yolo.save(Path.Combine(weightsPath, "best.bin"));
					tempLoss = valLoss_items.Sum();
				}
				Console.WriteLine();
				stopwatch.Stop();
				WriteLog(epoch, stopwatch.ElapsedMilliseconds / 1000f, trainLoss_items, valLoss_items);
			}
			Console.WriteLine("Train Done.");

			void WriteLog(int epoch, float time, float[] trainLoss_Items, float[] valLoss_Items)
			{
				if (!Directory.Exists(outputPath))
				{
					Directory.CreateDirectory(outputPath);
				}
				string fileName = Path.Combine(outputPath, "log.csv");
				StringBuilder stringBuilder = new StringBuilder();
				if (!File.Exists(fileName))
				{
					stringBuilder.Append("Epoch, Time, ");
					stringBuilder.Append(taskType switch
					{
						TaskType.Detection or TaskType.Obb => "TrainBoxLoss, TrainClsLoss, TrainDflLoss, ValBoxLoss, ValClsLoss, ValDflLoss,",
						TaskType.Segmentation => "TrainBoxLoss, TrainSegLoss, TrainClsLoss, TrainDflLoss, ValBoxLoss, ValSegLoss, ValClsLoss, ValDflLoss,",
						TaskType.Pose => "TrainBoxLoss, TrainPoseLoss, TrainKobjLoss, TrainClsLoss, TrainDflLoss, ValBoxLoss, ValPoseLoss, ValKobjLoss, ValClsLoss, ValDflLoss,",
						TaskType.Classification => "TrainLoss, ValLoss,",
						_ => throw new NotImplementedException("Not implemented task type: " + taskType),
					});
					stringBuilder.AppendLine("TrainLoss, ValLoss");
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
				stringBuilder.AppendLine($"{trainLoss_Items.Sum() / trainDataSet.Count}, {valLoss_Items.Sum() / valDataSet.Count}");
				File.AppendAllText(fileName, stringBuilder.ToString());
			}

			void WriteSettings()
			{
				if (!Directory.Exists(outputPath))
				{
					Directory.CreateDirectory(outputPath);
				}
				string fileName = Path.Combine(outputPath, "settings.txt");
				StringBuilder stringBuilder = new StringBuilder();
				stringBuilder.AppendLine("Training Settings:");
				stringBuilder.AppendLine($"Date Time: {DateTime.Now}");
				stringBuilder.AppendLine($"Yolo task type: {taskType}");
				stringBuilder.AppendLine($"Yolo type: {yoloType}");
				stringBuilder.AppendLine($"Yolo size: {yoloSize}");
				stringBuilder.AppendLine($"Image Process Type: {imageProcessType}");
				stringBuilder.AppendLine($"Precision type: {dtype}");
				stringBuilder.AppendLine($"Device type: {device}");
				stringBuilder.AppendLine($"Number Classes: {sortCount}");
				stringBuilder.AppendLine($"Image Size: {imageSize}");
				stringBuilder.AppendLine($"Epochs: {epochs}");
				stringBuilder.AppendLine($"Learning Rate: {lr}");
				stringBuilder.AppendLine($"Batch Size: {batchSize}");
				stringBuilder.AppendLine($"Num Workers: {numWorkers}");
				stringBuilder.AppendLine($"Root Path: {Path.GetFullPath(rootPath)}");
				stringBuilder.AppendLine($"Train Data Path: {trainDataPath}");
				stringBuilder.AppendLine($"Val Data Path: {valDataPath}");
				File.WriteAllText(fileName, stringBuilder.ToString());
			}
		}

		internal virtual float[] TrainEpoch(YoloDataLoader trainDataLoader, AMPWrapper amp, int epoch)
		{
			using (Tqdm<Dictionary<string, Tensor>> pbar = new Tqdm<Dictionary<string, Tensor>>(trainDataLoader, total: (int)trainDataLoader.Count, barStyle: Tqdm.BarStyle.Classic, barColor: Tqdm.BarColor.White, barWidth: 10, showPartialChar: true))
			{
				Tensor loss_items = null;
				foreach (Dictionary<string, Tensor> data in pbar)
				{
					using (NewDisposeScope())
					{
						if (data["batch_idx"].NumberOfElements == 0)
						{
							continue;
						}
						Tensor[] list = amp.Forward(data["images"].to(dtype));
						(Tensor ls, Tensor ls_item) = loss.forward(list.ToArray(), data);
						if (loss_items is null)
						{
							loss_items = torch.zeros_like(ls_item);
						}
						loss_items = loss_items + ls_item.to(loss_items.dtype, loss_items.device);
						loss_items = loss_items.MoveToOuterDisposeScope();
						amp.Step(ls);
						float[] ls_items = (ls_item).data<float>().ToArray();
						StringBuilder stringBuilder = new StringBuilder();
						stringBuilder.AppendFormat("{0,10}", epoch);
						foreach (float Items in ls_items)
						{
							stringBuilder.AppendFormat("{0,10:f3}", Items / data["images"].shape[0]);
						}
						stringBuilder.AppendFormat("{0,11}", data["batch_idx"].NumberOfElements);
						stringBuilder.AppendFormat("{0,6}", data["images"].shape[2]);
						pbar.SetDescription(stringBuilder.ToString());
					}
				}
				return loss_items.@float().data<float>().ToArray();

			}
		}

		internal virtual float[] Val(YoloDataLoader valDataLoader, AMPWrapper amp, int epoch)
		{
			using (Tqdm<Dictionary<string, Tensor>> pbar = new Tqdm<Dictionary<string, Tensor>>(valDataLoader, desc: $"Epoch {epoch,3}", total: (int)valDataLoader.Count, barStyle: Tqdm.BarStyle.Classic, barColor: Tqdm.BarColor.White, barWidth: 10, showPartialChar: true))
			{
				Tensor loss_items = null;
				foreach (Dictionary<string, Tensor> data in pbar)
				{
					using (NewDisposeScope())
					using (no_grad())
					{
						if (data["batch_idx"].NumberOfElements == 0)
						{
							continue;
						}
						Tensor[] list = amp.Evaluate(data["images"].to(dtype));
						var (ls, ls_item) = loss.forward(list.ToArray(), data);
						if (loss_items is null)
						{
							loss_items = torch.zeros_like(ls_item);
						}
						loss_items = loss_items + ls_item.to(loss_items.dtype, loss_items.device);
						loss_items = loss_items.MoveToOuterDisposeScope();
						pbar.SetPostfix(new (string key, object value)[]
								{
									("Val Loss", $"{(ls.ToSingle()):f3}"),
								});
					}
				}
				return loss_items.@float().data<float>().ToArray();

			}
		}

		internal abstract List<YoloResult> ImagePredict(Tensor orgImage, float PredictThreshold = 0.25f, float IouThreshold = 0.5f);



	}
}
