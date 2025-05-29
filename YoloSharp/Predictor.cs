using System.Diagnostics;
using System.Text.RegularExpressions;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;
using static YoloSharp.Yolo;

namespace YoloSharp
{
	public class Predictor
	{
		public class PredictResult
		{
			public int ClassID;
			public float Score;
			public int X;
			public int Y;
			public int W;
			public int H;
		}

		private Module<Tensor, Tensor[]> yolo;
		private Module<Tensor[], Tensor, (Tensor, Tensor)> loss;
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
				YoloType.Yolov5 => new Loss.Yolov5DetectionLoss(this.socrCount),
				YoloType.Yolov5u => new Loss.YoloDetectionLoss(this.socrCount),
				YoloType.Yolov8 => new Loss.YoloDetectionLoss(this.socrCount),
				YoloType.Yolov11 => new Loss.YoloDetectionLoss(this.socrCount),
				YoloType.Yolov12 => new Loss.YoloDetectionLoss(this.socrCount),
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

		public void Train(string trainDataPath, string valDataPath = "", string outputPath = "output", int imageSize = 640, int epochs = 100, float lr = 0.0001f, int batchSize = 8, int numWorkers = 0, bool useMosaic = true)
		{
			Console.WriteLine("Model will be write to: " + outputPath);
			Console.WriteLine("Load model...");

			YoloDataset trainDataSet = new YoloDataset(trainDataPath, imageSize, deviceType: this.device.type, useMosaic: useMosaic);
			if (trainDataSet.Count == 0)
			{
				throw new FileNotFoundException("No data found in the path: " + trainDataPath);
			}
			DataLoader trainDataLoader = new DataLoader(trainDataSet, batchSize, num_worker: numWorkers, shuffle: true, device: device);
			valDataPath = string.IsNullOrEmpty(valDataPath) ? trainDataPath : valDataPath;
			Optimizer optimizer = new SGD(yolo.parameters(), lr: lr);


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
					Tensor[] labels = new Tensor[indexs.Length];
					for (int i = 0; i < indexs.Length; i++)
					{
						var (img, lb) = trainDataSet.GetDataTensor(indexs[i]);
						images[i] = img.to(dtype, device);
						labels[i] = full(new long[] { lb.shape[0], lb.shape[1] + 1 }, i, dtype: dtype, device: lb.device);
						labels[i].slice(1, 1, lb.shape[1] + 1, 1).copy_(lb);
					}
					Tensor imageTensor = concat(images);
					Tensor labelTensor = concat(labels);
					if (labelTensor.shape[0] == 0)
					{
						continue;
					}
					Tensor[] list = yolo.forward(imageTensor);
					var (ls, ls_item) = loss.forward(list, labelTensor);
					optimizer.zero_grad();
					ls.backward();
					optimizer.step();
					Console.WriteLine($"Process: Epoch {epoch}, Step/Total Step  {step}/{trainDataLoader.Count}");
				}
				Console.Write("Do val now... ");
				float valLoss = Val(valDataPath, imageSize);
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

		private float Val(string valDataPath, int imageSize = 640)
		{
			YoloDataset yoloDataset = new YoloDataset(valDataPath, imageSize, deviceType: this.device.type, useMosaic: false);
			DataLoader loader = new DataLoader(yoloDataset, 4, num_worker: 0, shuffle: true, device: device);

			float lossValue = float.MaxValue;
			foreach (var data in loader)
			{
				long[] indexs = data["index"].data<long>().ToArray();
				Tensor[] images = new Tensor[indexs.Length];
				Tensor[] labels = new Tensor[indexs.Length];
				for (int i = 0; i < indexs.Length; i++)
				{
					var (img, lb) = yoloDataset.GetDataTensor(indexs[i]);
					images[i] = img.to(this.dtype, device);
					labels[i] = full(new long[] { lb.shape[0], lb.shape[1] + 1 }, i, dtype: dtype, device: lb.device);
					labels[i].slice(1, 1, lb.shape[1] + 1, 1).copy_(lb);
				}
				Tensor imageTensor = concat(images);
				Tensor labelTensor = concat(labels);

				if (labelTensor.shape[0] == 0)
				{
					continue;
				}

				Tensor[] list = yolo.forward(imageTensor);
				var (ls, ls_item) = loss.forward(list.ToArray(), labelTensor);
				if (lossValue == float.MaxValue)
				{
					lossValue = ls.ToSingle();
				}
				else
				{
					lossValue = lossValue + ls.ToSingle();
				}
			}
			lossValue = lossValue / yoloDataset.Count;
			return lossValue;
		}

		public List<PredictResult> ImagePredict(ImageMagick.MagickImage image, float PredictThreshold = 0.25f, float IouThreshold = 0.5f)
		{
			Tensor orgImage = Lib.GetTensorFromImage(image).to(dtype, device);
			orgImage = torch.stack(new Tensor[] { orgImage[2], orgImage[1], orgImage[0] }, dim: 0).unsqueeze(0) / 255.0f;

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

			List<PredictResult> predResults = new List<PredictResult>();
			for (int i = 0; i < results.shape[0]; i++)
			{
				int x = results[i, 0].ToInt32();
				int y = results[i, 1].ToInt32();
				int rw = (results[i, 2].ToInt32() - x);
				int rh = (results[i, 3].ToInt32() - y);

				float score = results[i, 4].ToSingle();
				int sort = results[i, 5].ToInt32();

				predResults.Add(new PredictResult()
				{
					ClassID = sort,
					Score = score,
					X = x,
					Y = y,
					W = rw,
					H = rh
				});
			}
			return predResults;
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
					string? layerPattern = yoloType switch
					{
						YoloType.Yolov5 => @"model\.24\.m",
						YoloType.Yolov5u => @"model\.24\.cv3",
						YoloType.Yolov8 => @"model\.22\.cv3",
						YoloType.Yolov11 => @"model\.23\.cv3",
						YoloType.Yolov12 => @"model\.21\.cv3",
						_ => null,
					};

					if (layerPattern != null)
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
				var re = NonMaxSuppression(tensor, PredictThreshold, IouThreshold);

				if (re[0] is not null)
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

				long bs = prediction.shape[0]; // batch size
				long nc = prediction.shape[1] - nm - 4; // number of classes
				long mi = 4 + nc; // mask start index
				Tensor xc = prediction[.., 4..(int)mi].amax(1) > confThreshold; // candidates

				List<Tensor> xindsList = new List<Tensor>();
				for (int idx = 0; idx < xc.shape[0]; idx++)
				{
					Tensor mask = xc[idx];
					xindsList.Add(torch.arange(mask.NumberOfElements, device: prediction.device));
				}
				Tensor xinds = torch.stack(xindsList, 0).unsqueeze(-1); // [batch, N, 1]

				prediction = prediction.transpose(-1, -2);
				// Settings
				int max_wh = 7680; // maximum box width and height
				int max_nms = 30000; // maximum number of boxes into torchvision.ops.nms()
				float time_limit = 0.5f + 0.05f * bs; // seconds to quit after

				prediction[TensorIndex.Ellipsis, ..4] = torchvision.ops.box_convert(prediction[TensorIndex.Ellipsis, ..4], torchvision.ops.BoxFormats.cxcywh, torchvision.ops.BoxFormats.xyxy);
				DateTime t = DateTime.Now;

				List<Tensor> output = new List<Tensor>(new Tensor[bs]);
				List<Tensor> keepi = new List<Tensor>(new Tensor[bs]);
				for (int xi = 0; xi < bs; xi++)
				{
					Tensor x = prediction[xi];
					Tensor xk = xinds[xi];
					Tensor filt = xc[xi];
					x = x[filt]; // confidence
					xk = xk[filt];

					Tensor[] box_cls_mask = x.split(new long[] { 4, nc, nm }, 1);
					Tensor box = box_cls_mask[0];
					Tensor cls = box_cls_mask[1];
					Tensor mask = box_cls_mask[2];

					(Tensor conf, Tensor j) = cls.max(1, keepdim: true);
					filt = conf.view(-1) > confThreshold;

					x = torch.cat(new Tensor[] { box, conf, j.@float(), mask }, 1)[filt];
					xk = xk[filt];

					// Check shape
					long n = x.shape[0];  // number of boxes
					if (n is 0)
					{
						continue;
					}

					if (n > max_nms)//  # excess boxes
					{
						filt = x[.., 4].argsort(descending: true)[..max_nms];  // sort by confidence and remove excess boxes
						(x, xk) = (x[filt], xk[filt]);
					}

					// Batched NMS
					Tensor c = x[.., 5..6] * max_wh;  // classes
					Tensor scores = x[.., 4];  // scores
					Tensor boxes = x[.., ..4] + c;  // boxes (offset by class)
					Tensor i = torchvision.ops.nms(boxes, scores, iouThreshold);  // NMS
					i = i[..max_det]; // limit detections
					(output[xi], keepi[xi]) = (x[i], xk[i].reshape(-1));
					if ((DateTime.Now - t).TotalSeconds > time_limit)
					{
						// time limit exceeded
						Console.WriteLine($"NMS time limit {time_limit}s exceeded");
					}
				}

				return output;

			}

		}

	}
}
