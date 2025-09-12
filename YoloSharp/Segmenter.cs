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
	public class Segmenter
	{
		private Module<Tensor, Tensor[]> yolo;
		private Module<Tensor[], Tensor, Tensor, (Tensor, Tensor)> loss;
		private torch.Device device;
		private torch.ScalarType dtype;
		private int sortCount;
		private YoloType yoloType;

		public Segmenter(int sortCount = 80, YoloType yoloType = YoloType.Yolov8, YoloSize yoloSize = YoloSize.n, DeviceType deviceType = DeviceType.CUDA, ScalarType dtype = ScalarType.Float32)
		{
			torchvision.io.DefaultImager = new torchvision.io.SkiaImager();
			if (yoloType == YoloType.Yolov5 || yoloType == YoloType.Yolov5u || yoloType == YoloType.Yolov12)
			{
				throw new ArgumentException("Segmenter not support yolov5, yolov5u or yolov12. Please use yolov8 or yolov11 instead.");
			}

			this.device = new torch.Device((TorchSharp.DeviceType)deviceType);
			this.dtype = (torch.ScalarType)dtype;
			this.sortCount = sortCount;
			this.yoloType = yoloType;
			yolo = yoloType switch
			{
				YoloType.Yolov8 => new Yolov8Segment(sortCount, yoloSize, device, this.dtype),
				YoloType.Yolov11 => new Yolov11Segment(sortCount, yoloSize, device, this.dtype),
				_ => throw new NotImplementedException("Yolo type not supported."),
			};
			loss = yoloType switch
			{
				YoloType.Yolov8 => new Utils.Loss.V8SegmentationLoss(this.sortCount),
				YoloType.Yolov11 => new Utils.Loss.V8SegmentationLoss(this.sortCount),
				_ => throw new NotImplementedException("Yolo type not supported."),
			};
			//Tools.TransModelFromSafetensors(yolo, @".\yolov8n-seg.safetensors", @".\PreTrainedModels\yolov11x-seg.bin");
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
					Tensor[] masks = new Tensor[indexs.Length];
					for (int i = 0; i < indexs.Length; i++)
					{
						(Tensor img, Tensor lb, Tensor mask) = trainDataSet.GetSegmentDataTensor(indexs[i]);
						images[i] = img.to(dtype, device).unsqueeze(0) / 255.0f;
						labels[i] = full(new long[] { lb.shape[0], lb.shape[1] + 1 }, i, dtype: dtype, device: lb.device);
						labels[i].slice(1, 1, lb.shape[1] + 1, 1).copy_(lb);
						masks[i] = mask.to(dtype, device).unsqueeze(0);
					}
					Tensor imageTensor = concat(images);
					Tensor labelTensor = concat(labels);
					Tensor maskTensor = concat(masks);
					if (labelTensor.shape[0] == 0)
					{
						continue;
					}

					Tensor[] list = yolo.forward(imageTensor);
					var (ls, ls_item) = loss.forward(list, labelTensor, maskTensor);
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
				Tensor[] masks = new Tensor[indexs.Length];
				for (int i = 0; i < indexs.Length; i++)
				{
					var (img, lb, mask) = yoloDataset.GetSegmentDataTensor(indexs[i]);
					images[i] = img.to(dtype, device).unsqueeze(0) / 255.0f;
					labels[i] = full(new long[] { lb.shape[0], lb.shape[1] + 1 }, i, dtype: dtype, device: lb.device);
					labels[i].slice(1, 1, lb.shape[1] + 1, 1).copy_(lb);
					masks[i] = mask.to(dtype, device).unsqueeze(0);
				}
				Tensor imageTensor = concat(images);
				Tensor labelTensor = concat(labels);
				Tensor maskTensor = concat(masks);
				if (labelTensor.shape[0] == 0)
				{
					continue;
				}
				Tensor[] list = yolo.forward(imageTensor);
				var (ls, ls_item) = loss.forward(list, labelTensor, maskTensor);
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

		public (List<YoloResult>, SKBitmap) ImagePredict(SKBitmap image, float PredictThreshold = 0.25f, float IouThreshold = 0.5f, int imgSize = 640)
		{
			yolo.eval();
			Tensor orgImage = Lib.GetTensorFromImage(image).to(dtype, device);
			orgImage = orgImage.unsqueeze(0) / 255.0f;

			float gain = Math.Max((float)orgImage.shape[2] / imgSize, (float)orgImage.shape[3] / imgSize);
			int new_w = (int)(orgImage.shape[3] / gain);
			int new_h = (int)(orgImage.shape[2] / gain);
			Tensor tensor = torch.nn.functional.interpolate(orgImage, new long[] { new_h, new_w }, mode: InterpolationMode.Bilinear, align_corners: false);
			int padHeight = imgSize - new_h;
			int padWidth = imgSize - new_w;

			tensor = torch.nn.functional.pad(tensor, new long[] { 0, padWidth, 0, padHeight }, PaddingModes.Zeros);

			Tensor[] outputs = yolo.forward(tensor);

			(List<Tensor> preds, var _) = Ops.non_max_suppression(outputs[0], nc: sortCount, conf_thres: PredictThreshold, iou_thres: IouThreshold);
			Tensor proto = outputs[4];

			List<YoloResult> results = new List<YoloResult>();
			SKBitmap maskBitmap = new SKBitmap();
			if (proto.shape[0] > 0)
			{
				if (!Equals(preds[0], null))
				{
					int i = 0;
					Tensor masks = Utils.Ops.process_mask(proto[i], preds[i][.., 6..], preds[i][.., 0..4], new long[] { tensor.shape[2], tensor.shape[3] }, upsample: true);
					preds[i][.., ..4] = preds[i][.., ..4] * gain;
					preds[i][.., ..4] = Utils.Ops.clip_boxes(preds[i][.., ..4], new float[] { orgImage.shape[2], orgImage.shape[3] });
					Tensor orgImg = (tensor[0] * 255).@byte();
					masks = torchvision.transforms.functional.crop(masks, 0, 0, new_h, new_w);
					masks = torchvision.transforms.functional.resize(masks, (int)orgImage.shape[2], (int)orgImage.shape[3]);

					Random rand = new Random(42);
					for (int j = 0; j < masks.shape[0]; j++)
					{
						bool[,] mask = new bool[masks.shape[2], masks.shape[1]];
						Buffer.BlockCopy(masks[j].transpose(0, 1).@bool().data<bool>().ToArray(), 0, mask, 0, mask.Length);

						int x = (preds[i][j, 0]).ToInt32();
						int y = (preds[i][j, 1]).ToInt32();

						int w = (preds[i][j, 2]).ToInt32() - x;
						int h = (preds[i][j, 3]).ToInt32() - y;

						results.Add(new YoloResult()
						{
							ClassID = preds[i][j, 5].ToInt32(),
							Score = preds[i][j, 4].ToSingle(),
							CenterX = x + w / 2,
							CenterY = y + h / 2,
							Width = w,
							Height = h,
							Mask = mask
						});
						orgImage[0, 0, masks[j].@bool()] += rand.NextSingle();
						orgImage[0, 1, masks[j].@bool()] += rand.NextSingle();
						orgImage[0, 2, masks[j].@bool()] += rand.NextSingle();
					}
					orgImage = (orgImage.clip(0, 1) * 255).@byte().squeeze(0);
					maskBitmap = Lib.GetImageFromTensor(orgImage);
				}
			}
			return (results, maskBitmap);
		}

		public void LoadModel(string path, bool skipNcNotEqualLayers = false)
		{
			Dictionary<string, Tensor> state_dict = Lib.LoadModel(path, skipNcNotEqualLayers);
			if (state_dict.Count != yolo.state_dict().Count)
			{
				Console.WriteLine("Mismatched tensor count while loading. Model will run with random weight.");
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

	}
}
