using Data;
using OpenCvSharp;
using SkiaSharp;
using System.Text.RegularExpressions;
using TorchSharp;
using TorchSharp.Modules;
using Utils;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;
using static YoloSharp.Yolo;

namespace YoloSharp
{
	public class Predictor
	{
		private Module<Tensor, Tensor[]> yolo;
		private Module<Tensor[], Dictionary<string, Tensor>, (Tensor loss, Tensor loss_items)> loss;
		private Module<Tensor, float, float, Tensor> predict;
		private torch.Device device;
		private torch.ScalarType dtype;
		private int socrCount;
		private YoloType yoloType;

		public Predictor(int socrCount = 80, YoloType yoloType = YoloType.Yolov8, YoloSize yoloSize = YoloSize.n, DeviceType deviceType = DeviceType.CUDA, ScalarType dtype = ScalarType.Float32)
		{
			torchvision.io.DefaultImager = new torchvision.io.SkiaImager();

			this.device = new torch.Device((TorchSharp.DeviceType)deviceType);
			this.dtype = (torch.ScalarType)dtype;
			this.socrCount = socrCount;
			this.yoloType = yoloType;

			yolo = yoloType switch
			{
				YoloType.Yolov5 => new Yolov5(socrCount, yoloSize, device, this.dtype),
				YoloType.Yolov5u => new Yolov5u(socrCount, yoloSize, device, this.dtype),
				YoloType.Yolov8 => new Yolov8(socrCount, yoloSize, device, this.dtype),
				YoloType.Yolov11 => new Yolov11(socrCount, yoloSize, device, this.dtype),
				YoloType.Yolov12 => new Yolov12(socrCount, yoloSize, device, this.dtype),
				_ => throw new NotImplementedException(),
			};

			loss = yoloType switch
			{
				YoloType.Yolov5 => new Utils.Loss.V5DetectionLoss(this.socrCount),
				YoloType.Yolov5u => new Utils.Loss.V8DetectionLoss(this.socrCount),
				YoloType.Yolov8 => new Utils.Loss.V8DetectionLoss(this.socrCount),
				YoloType.Yolov11 => new Utils.Loss.V8DetectionLoss(this.socrCount),
				YoloType.Yolov12 => new Utils.Loss.V8DetectionLoss(this.socrCount),
				_ => throw new NotImplementedException(),
			};
			predict = yoloType switch
			{
				YoloType.Yolov5 => new Yolov5Predict(),
				YoloType.Yolov5u => new YoloPredict(),
				YoloType.Yolov8 => new YoloPredict(),
				YoloType.Yolov11 => new YoloPredict(),
				YoloType.Yolov12 => new YoloPredict(),
				_ => throw new NotImplementedException(),
			};

			//Tools.TransModelFromSafetensors(yolo, @".\yolov11n.safetensors", @".\yolov11n.bin");
		}

		public void Train(string rootPath, string trainDataPath = "", string valDataPath = "", string outputPath = "output", int imageSize = 640, int epochs = 100, float lr = 0.0001f, int batchSize = 8, int numWorkers = 0, ImageProcessType imageProcessType = ImageProcessType.Letterbox)
		{
			Console.WriteLine("Model will be write to: " + outputPath);
			Console.WriteLine("Load model...");

			YoloDataset trainDataSet = new YoloDataset(rootPath, trainDataPath, imageSize, TaskType.Detection, imageProcessType);
			if (trainDataSet.Count == 0)
			{
				throw new FileNotFoundException("No data found in the path: " + rootPath);
			}

			DataLoader trainDataLoader = new DataLoader(trainDataSet, batchSize, num_worker: numWorkers, shuffle: true, device: device);

			valDataPath = string.IsNullOrEmpty(valDataPath) ? trainDataPath : valDataPath;

			YoloDataset valDataSet = new YoloDataset(rootPath, valDataPath, imageSize, TaskType.Detection, imageProcessType);
			if (valDataSet.Count == 0)
			{
				throw new FileNotFoundException("No data found in the path: " + rootPath);
			}

			DataLoader valDataLoader = new DataLoader(valDataSet, 4, num_worker: 0, shuffle: true, device: device);

			Optimizer optimizer = new SGD(yolo.parameters(), lr: lr, momentum: 0.9, weight_decay: 5e-4);

			float tempLoss = float.MaxValue;
			Console.WriteLine("Start Training...");
			yolo.train(true);
			for (int epoch = 0; epoch < epochs; epoch++)
			{
				yolo.train();
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
							bboxes.AddRange(imageData.ResizedLabels.Select(x => torch.tensor(new float[] { x.CenterX, x.CenterY, x.Width, x.Height })));
						}
					}

					Tensor batch_idx_tensor = torch.tensor(batch_idx, dtype: dtype, device: device).view(-1, 1);
					Tensor cls_tensor = torch.tensor(cls, dtype: dtype, device: device).view(-1, 1);
					Tensor bboxes_tensor = bboxes.Count == 0 ? torch.zeros(new long[] { 0, 4 }) : torch.stack(bboxes).to(dtype, device) / trainDataSet.ImageSize;
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
						ImageData imageData = valDataset.GetImageAndLabelDataWithLetterBox(indexs[i]);
						images[i] = Lib.GetTensorFromImage(imageData.ResizedImage).to(device).unsqueeze(0) / 255.0f;
						if (imageData.ResizedLabels is not null)
						{
							batch_idx.AddRange(Enumerable.Repeat((float)i, imageData.ResizedLabels.Count));
							cls.AddRange(imageData.ResizedLabels.Select(x => (float)x.LabelID));
							bboxes.AddRange(imageData.ResizedLabels.Select(x => torch.tensor(new float[] { x.CenterX, x.CenterY, x.Width, x.Height })));
						}
					}
					Tensor batch_idx_tensor = torch.tensor(batch_idx, dtype: dtype, device: device).view(-1, 1);
					Tensor cls_tensor = torch.tensor(cls, dtype: dtype, device: device).view(-1, 1);
					Tensor bboxes_tensor = torch.stack(bboxes).to(dtype, device) / valDataset.ImageSize;

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

		public List<YoloResult> ImagePredict(SKBitmap image, float PredictThreshold = 0.25f, float IouThreshold = 0.5f)
		{
			Tensor orgImage = Lib.GetTensorFromImage(image).to(dtype, device);
			return ImagePredict(orgImage, PredictThreshold, IouThreshold);
		}

		public List<YoloResult> ImagePredict(string imagePath, float PredictThreshold = 0.25f, float IouThreshold = 0.5f)
		{
			Tensor orgImage = Lib.GetTensorFromImage(imagePath).to(dtype, device);
			return ImagePredict(orgImage, PredictThreshold, IouThreshold);
		}

		public List<YoloResult> ImagePredict(Mat mat, float PredictThreshold = 0.25f, float IouThreshold = 0.5f)
		{
			Tensor orgImage = Lib.GetTensorFromImage(mat).to(dtype, device);
			return ImagePredict(orgImage, PredictThreshold, IouThreshold);
		}

		public List<YoloResult> ImagePredict(Tensor orgImage, float PredictThreshold = 0.25f, float IouThreshold = 0.5f)
		{
			using (no_grad())
			{
				// Change RGB → BGR
				orgImage = orgImage.index_select(0, torch.tensor(new long[] { 2, 1, 0 }, device: device)).unsqueeze(0) / 255.0f;

				int w = (int)orgImage.shape[3];
				int h = (int)orgImage.shape[2];
				int padHeight = 32 - (int)(orgImage.shape[2] % 32);
				int padWidth = 32 - (int)(orgImage.shape[3] % 32);

				padHeight = padHeight == 32 ? 0 : padHeight;
				padWidth = padWidth == 32 ? 0 : padWidth;

				Tensor input = torch.nn.functional.pad(orgImage, new long[] { 0, padWidth, 0, padHeight }, PaddingModes.Zeros);
				yolo.eval();

				Tensor[] tensors = yolo.forward(input);
				Tensor results = predict.forward(tensors[0], PredictThreshold, IouThreshold);

				List<YoloResult> predResults = new List<YoloResult>();
				for (int i = 0; i < results.shape[0]; i++)
				{
					int x = results[i, 0].ToInt32();
					int y = results[i, 1].ToInt32();
					int rw = (results[i, 2].ToInt32() - x);
					int rh = (results[i, 3].ToInt32() - y);

					float score = results[i, 4].ToSingle();
					int sort = results[i, 5].ToInt32();

					predResults.Add(new YoloResult()
					{
						ClassID = sort,
						Score = score,
						CenterX = x + rw / 2,
						CenterY = y + rh / 2,
						Width = rw,
						Height = rh
					});
				}
				return predResults;
			}
		}


		/// <summary>
		/// Load model from path.
		/// </summary>
		/// <param name="path">The Model path</param>
		/// <param name="skipNcNotEqualLayers">If nc not equals the label count in model, please set it true otherwise set it false.</param>
		public void LoadModel(string path, bool skipNcNotEqualLayers = false)
		{
			Dictionary<string, Tensor> state_dict = Lib.LoadModel(path, skipNcNotEqualLayers);

			if (state_dict.Count != yolo.state_dict().Count)
			{
				Console.WriteLine("Mismatched tensor count while loading. Make sure that the model you are loading into is exactly the same as the origin.");
				Console.WriteLine("Model will run with random weight.");
			}
			else
			{
				torch.ScalarType modelType = state_dict.Values.First().dtype;
				List<string> skipList = new List<string>();
				long nc = 0;

				if (skipNcNotEqualLayers)
				{
					string layerPattern = yoloType switch
					{
						YoloType.Yolov5 => @"model\.24\.m",
						YoloType.Yolov5u => @"model\.24\.cv3",
						YoloType.Yolov8 => @"model\.22\.cv3",
						YoloType.Yolov11 => @"model\.23\.cv3",
						YoloType.Yolov12 => @"model\.21\.cv3",
						_ => string.Empty,
					};

					if (!string.IsNullOrEmpty(layerPattern))
					{
						skipList = state_dict.Keys.Where(x => Regex.IsMatch(x, layerPattern)).ToList();
						nc = yoloType switch
						{
							YoloType.Yolov5 => state_dict[skipList[0]].shape[0] / 3 - 5,
							_ => state_dict[skipList.LastOrDefault(a => a.EndsWith(".bias"))!].shape[0]
						};
					}

					if (nc == socrCount)
					{
						skipList.Clear();
					}
				}

				var (miss, err) = yolo.load_state_dict(state_dict, skip: skipList);
				if (skipList.Count > 0)
				{
					Console.WriteLine("Waring! You are skipping nc reference layers.");
					Console.WriteLine("This will get wrong result in Predict, sort count loaded in weight is " + nc);
				}
			}
		}

		private class Yolov5Predict : Module<Tensor, float, float, Tensor>
		{
			internal Yolov5Predict() : base(nameof(Yolov5Predict))
			{

			}

			public override Tensor forward(Tensor tensor, float PredictThreshold = 0.25f, float IouThreshold = 0.5f)
			{
				List<Tensor> re = NonMaxSuppression(tensor, PredictThreshold, IouThreshold);

				if (!Equals(re[0], null))
				{
					return re[0];
				}
				else
				{
					return torch.tensor(new float[0, 6]);
				}
			}

			private List<Tensor> NonMaxSuppression(Tensor prediction, float confThreshold = 0.25f, float iouThreshold = 0.45f, bool agnostic = false, int max_det = 300, int nm = 0)
			{
				// Checks
				if (confThreshold < 0 || confThreshold > 1)
				{
					throw new ArgumentException($"Invalid Confidence threshold {confThreshold}, valid values are between 0.0 and 1.0");
				}
				if (iouThreshold < 0 || iouThreshold > 1)
				{
					throw new ArgumentException($"Invalid IoU {iouThreshold}, valid values are between 0.0 and 1.0");
				}

				var device = prediction.device;
				var scalType = prediction.dtype;

				var bs = prediction.shape[0]; // batch size
				var nc = prediction.shape[2] - nm - 5; // number of classes
				var xc = prediction[TensorIndex.Ellipsis, 4] > confThreshold; // candidates

				// Settings
				var max_wh = 7680; // maximum box width and height
				var max_nms = 30000; // maximum number of boxes into torchvision.ops.nms()
				var time_limit = 0.5f + 0.05f * bs; // seconds to quit after

				var t = DateTime.Now;
				var mi = 5 + nc; // mask start index
				var output = new List<Tensor>(new Tensor[bs]);
				for (int xi = 0; xi < bs; xi++)
				{
					var x = prediction[xi];
					x = x[xc[xi]]; // confidence

					// Compute conf
					x[TensorIndex.Ellipsis, TensorIndex.Slice(5, mi)] *= x[TensorIndex.Ellipsis, 4].unsqueeze(-1); // conf = obj_conf * cls_conf

					// Box/Mask
					var box = torchvision.ops.box_convert(x[TensorIndex.Ellipsis, TensorIndex.Slice(0, 4)], torchvision.ops.BoxFormats.cxcywh, torchvision.ops.BoxFormats.xyxy); // center_x, center_y, width, height) to (x1, y1, x2, y2)

					// Detections matrix nx6 (xyxy, conf, cls)

					var conf = x[TensorIndex.Colon, TensorIndex.Slice(5, mi)].max(1, true);
					var j = conf.indexes;
					x = torch.cat(new Tensor[] { box, conf.values, j.to_type(scalType) }, 1)[conf.values.view(-1) > confThreshold];

					var n = x.shape[0]; // number of boxes
					if (n == 0)
					{
						continue;
					}

					x = x[x[TensorIndex.Ellipsis, 4].argsort(descending: true)][TensorIndex.Slice(0, max_nms)]; // sort by confidence and remove excess boxes

					// Batched NMS
					var c = x[TensorIndex.Ellipsis, 5].unsqueeze(-1) * (agnostic ? 0 : max_wh); // classes
					var boxes = x[TensorIndex.Ellipsis, TensorIndex.Slice(0, 4)] + c;
					var scores = x[TensorIndex.Ellipsis, 4];
					var i = torchvision.ops.nms(boxes, scores, iouThreshold); // NMS
					i = i[TensorIndex.Slice(0, max_det)]; // limit detections

					output[xi] = x[i];
					if ((DateTime.Now - t).TotalSeconds > time_limit)
					{
						Console.WriteLine($"WARNING ⚠️ NMS time limit {time_limit:F3}s exceeded");
						break; // time limit exceeded
					}
				}

				return output;

			}
		}

		private class YoloPredict : Module<Tensor, float, float, Tensor>
		{
			internal YoloPredict() : base(nameof(YoloPredict))
			{

			}

			public override Tensor forward(Tensor tensor, float PredictThreshold = 0.25f, float IouThreshold = 0.5f)
			{
				var (output, keepi) = Ops.non_max_suppression(tensor, PredictThreshold, IouThreshold);

				if (output[0] is not null)
				{
					return output[0];
				}
				else
				{
					return torch.tensor(new float[0, 6]);
				}
			}
		}

	}
}
