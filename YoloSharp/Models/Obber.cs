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
	public class Obber : YoloBaseTaskModel
	{
		private readonly Device device;
		private readonly torch.ScalarType dtype;
		private readonly int sortCount;
		private readonly YoloType yoloType;
		private Module<Tensor, Tensor[]> yolo;
		private Module<Tensor[], Dictionary<string, Tensor>, (Tensor loss, Tensor loss_items)> loss;

		public Obber(int numberClasses = 15, YoloType yoloType = YoloType.Yolov8, YoloSize yoloSize = YoloSize.n, Types.DeviceType deviceType = Types.DeviceType.CUDA, Types.ScalarType dtype = Types.ScalarType.Float32)
		{
			torchvision.io.DefaultImager = new torchvision.io.SkiaImager();
			if (yoloType == YoloType.Yolov5 || yoloType == YoloType.Yolov5u || yoloType == YoloType.Yolov12)
			{
				throw new ArgumentException("Obb not support yolov5, yolov5u or yolov12. Please use yolov8 or yolov11 instead.");
			}

			device = new Device((TorchSharp.DeviceType)deviceType);
			this.dtype = (torch.ScalarType)dtype;
			this.sortCount = numberClasses;
			this.yoloType = yoloType;
			yolo = yoloType switch
			{
				YoloType.Yolov8 => new Yolo.Yolov8Obb(numberClasses, yoloSize, device, this.dtype),
				YoloType.Yolov11 => new Yolo.Yolov11Obb(numberClasses, yoloSize, device, this.dtype),
				_ => throw new NotImplementedException("Yolo type not supported."),
			};
			loss = yoloType switch
			{
				YoloType.Yolov8 => new Loss.V8OBBLoss(this.sortCount),
				YoloType.Yolov11 => new Loss.V8OBBLoss(this.sortCount),
				_ => throw new NotImplementedException("Yolo type not supported."),
			};

			//Tools.TransModelFromSafetensors(yolo, @".\yolov8n-obb.safetensors", @".\PreTrainedModels\yolov8n-obb.bin");
		}

		public void LoadModel(string path, bool skipNcNotEqualLayers = false)
		{
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
					string layerPattern = yoloType switch
					{
						YoloType.Yolov8 => @"model\.22\.cv3",
						YoloType.Yolov11 => @"model\.23\.cv3",
						_ => string.Empty
					};

					if (!string.IsNullOrEmpty(layerPattern))
					{
						skipList = state_dict.Keys.Where(x => Regex.IsMatch(x, layerPattern)).ToList();
						if (state_dict[skipList.LastOrDefault(a => a.EndsWith(".bias"))!].shape[0] == sortCount)
						{
							skipList.Clear();
						}
					}
				}

				var (miss, err) = yolo.load_state_dict(state_dict, skip: skipList);
				if (skipList.Count > 0)
				{
					Console.WriteLine("Warning! Skipping nc reference layers. This may cause incorrect predictions.");
				}
				yolo.to(dtype);
			}
		}

		public List<YoloResult> ImagePredict(Tensor orgImage, float PredictThreshold = 0.25f, float IouThreshold = 0.5f)
		{
			using (no_grad())
			{
				yolo.eval();
				// Change RGB → BGR
				orgImage = orgImage.to(dtype, device).unsqueeze(0);

				int w = (int)orgImage.shape[3];
				int h = (int)orgImage.shape[2];
				int padHeight = 32 - (int)(orgImage.shape[2] % 32);
				int padWidth = 32 - (int)(orgImage.shape[3] % 32);

				padHeight = padHeight == 32 ? 0 : padHeight;
				padWidth = padWidth == 32 ? 0 : padWidth;

				Tensor input = functional.pad(orgImage, new long[] { 0, padWidth, 0, padHeight }, PaddingModes.Zeros, 114) / 255.0f;
				Tensor[] tensors = yolo.forward(input);
				(List<Tensor> nms_result, var _) = Ops.non_max_suppression(tensors[0], nc: sortCount, iou_thres: IouThreshold, rotated: true);
				List<YoloResult> results = new List<YoloResult>();
				if (nms_result.Count > 0)
				{
					if (nms_result[0] is not null)
					{
						for (int i = 0; i < nms_result[0].shape[0]; i++)
						{
							YoloResult result = new YoloResult();
							result.CenterX = nms_result[0][i][0].ToInt32();
							result.CenterY = nms_result[0][i][1].ToInt32();
							result.Width = nms_result[0][i][2].ToInt32();
							result.Height = nms_result[0][i][3].ToInt32();
							result.Radian = nms_result[0][i][6].ToSingle();
							result.ClassID = nms_result[0][i][5].ToInt32();
							result.Score = nms_result[0][i][4].ToSingle();
							results.Add(result);
						}
					}
				}
				return results;
			}
		}

		public void Train(string rootPath, string trainDataPath = "", string valDataPath = "", string outputPath = "output", int imageSize = 640, int epochs = 100, float lr = 0.0001f, int batchSize = 8, int numWorkers = 0, ImageProcessType imageProcessType = ImageProcessType.Letterbox)
		{
			Console.WriteLine("Model will be write to: " + outputPath);
			Console.WriteLine("Load model...");

			YoloDataset trainDataSet = new YoloDataset(rootPath, trainDataPath, imageSize, TaskType.Obb, imageProcessType);
			if (trainDataSet.Count == 0)
			{
				throw new FileNotFoundException("No data found in the path: " + trainDataPath);
			}

			DataLoader trainDataLoader = new DataLoader(trainDataSet, batchSize, num_worker: numWorkers, shuffle: true, device: device);

			valDataPath = string.IsNullOrEmpty(valDataPath) ? trainDataPath : valDataPath;

			YoloDataset valDataSet = new YoloDataset(rootPath, valDataPath, imageSize, TaskType.Obb, imageProcessType);
			if (valDataSet.Count == 0)
			{
				throw new FileNotFoundException("No data found in the path: " + trainDataPath);
			}

			DataLoader valDataLoader = new DataLoader(valDataSet, 4, num_worker: 0, shuffle: true, device: device);
			Optimizer optimizer = new SGD(yolo.parameters(), lr: lr, momentum: 0.9, weight_decay: 5e-4);
			float tempLoss = float.MaxValue;
			Console.WriteLine("Start Training...");
			yolo.train(true);
			for (int epoch = 0; epoch < epochs; epoch++)
			{
				int step = 0;
				foreach (var data in trainDataLoader)
				{
					step++;
					long[] indexs = data["index"].data<long>().ToArray();
					Tensor[] images = new Tensor[indexs.Length];
					List<float> batch_idx = new List<float>();
					List<float> cls = new List<float>();
					List<Tensor> bboxes = new List<Tensor>();
					for (int i = 0; i < indexs.Length; i++)
					{
						ImageData imageData = trainDataSet.GetImageAndLabelData(indexs[i]);
						images[i] = Lib.GetTensorFromImage(imageData.ResizedImage).to(device).unsqueeze(0) / 255.0f;
						if (imageData.ResizedLabels is not null)
						{
							batch_idx.AddRange(Enumerable.Repeat((float)i, imageData.ResizedLabels.Count));
							cls.AddRange(imageData.ResizedLabels.Select(x => (float)x.LabelID));
							bboxes.AddRange(imageData.ResizedLabels.Select(x => tensor(new float[] { x.CenterX / trainDataSet.ImageSize, x.CenterY / trainDataSet.ImageSize, x.Width / trainDataSet.ImageSize, x.Height / trainDataSet.ImageSize, x.Radian })));
						}
					}

					Tensor batch_idx_tensor = tensor(batch_idx, dtype: dtype, device: device).view(-1, 1);
					Tensor cls_tensor = tensor(cls, dtype: dtype, device: device).view(-1, 1);
					Tensor bboxes_tensor = bboxes.Count == 0 ? zeros(new long[] { 0, 5 }) : stack(bboxes).to(dtype, device);
					Tensor imageTensor = concat(images);
					if (batch_idx.Count < 1)
					{
						continue;
					}

					Dictionary<string, Tensor> targets = new Dictionary<string, Tensor>()
					{
						{ "batch_idx", batch_idx_tensor },
						{ "cls", cls_tensor },
						{ "bboxes", bboxes_tensor }
					};

					Tensor[] list = yolo.forward(imageTensor);
					(Tensor ls, Tensor ls_item) = loss.forward(list, targets);
					optimizer.zero_grad();
					ls.backward();
					optimizer.step();
					Console.WriteLine($"Process: Epoch {epoch}, Step/Total Step  {step}/{trainDataLoader.Count}");
				}
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
			}
			Console.WriteLine("Train Done.");
		}

		private float Val(YoloDataset valDataset, DataLoader valDataLoader)
		{
			using (no_grad())
			{
				float lossValue = float.MaxValue;
				foreach (var data in valDataLoader)
				{
					long[] indexs = data["index"].data<long>().ToArray();
					Tensor[] images = new Tensor[indexs.Length];
					List<float> batch_idx = new List<float>();
					List<float> cls = new List<float>();
					List<Tensor> bboxes = new List<Tensor>();
					for (int i = 0; i < indexs.Length; i++)
					{
						ImageData imageData = valDataset.GetImageAndLabelData(indexs[i]);
						images[i] = Lib.GetTensorFromImage(imageData.ResizedImage).to(device).unsqueeze(0) / 255.0f;
						if (imageData.ResizedLabels is not null)
						{
							batch_idx.AddRange(Enumerable.Repeat((float)i, imageData.ResizedLabels.Count));
							cls.AddRange(imageData.ResizedLabels.Select(x => (float)x.LabelID));
							bboxes.AddRange(imageData.ResizedLabels.Select(x => tensor(new float[] { x.CenterX / valDataset.ImageSize, x.CenterY / valDataset.ImageSize, x.Width / valDataset.ImageSize, x.Height / valDataset.ImageSize, x.Radian })));
						}
					}
					Tensor batch_idx_tensor = tensor(batch_idx, dtype: dtype, device: device).view(-1, 1);
					Tensor cls_tensor = tensor(cls, dtype: dtype, device: device).view(-1, 1);
					Tensor bboxes_tensor = stack(bboxes).to(dtype, device);

					Tensor imageTensor = concat(images);
					if (batch_idx.Count < 1)
					{
						continue;
					}

					Dictionary<string, Tensor> targets = new Dictionary<string, Tensor>()
					{
						{ "batch_idx", batch_idx_tensor },
						{ "cls", cls_tensor },
						{ "bboxes", bboxes_tensor }
					};

					Tensor[] list = yolo.forward(imageTensor);
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
				lossValue = lossValue / valDataset.Count;
				return lossValue;
			}
		}



	}
}
