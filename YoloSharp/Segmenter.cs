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
	public class Segmenter
	{
		private Module<Tensor, Tensor[]> yolo;
		private Module<Tensor[], Dictionary<string, Tensor>, (Tensor loss, Tensor loss_items)> loss;
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

		public void Train(string trainDataPath, string valDataPath = "", string outputPath = "output", int imageSize = 640, int epochs = 100, float lr = 0.0001f, int batchSize = 8, int numWorkers = 0, ImageProcessType imageProcessType = ImageProcessType.Letterbox, int maskSize = 160)
		{
			Console.WriteLine("Model will be write to: " + outputPath);
			Console.WriteLine("Load model...");

			YoloDataClass trainDataSet = new YoloDataClass(trainDataPath, imageSize, TaskType.Segmentation, imageProcessType);
			if (trainDataSet.Count == 0)
			{
				throw new FileNotFoundException("No data found in the path: " + trainDataPath);
			}
			DataLoader trainDataLoader = new DataLoader(trainDataSet, batchSize, num_worker: numWorkers, shuffle: false, device: device);
			valDataPath = string.IsNullOrEmpty(valDataPath) ? trainDataPath : valDataPath;

			YoloDataClass valDataSet = new YoloDataClass(valDataPath, imageSize, TaskType.Segmentation, imageProcessType);
			DataLoader valDataLoader = new DataLoader(valDataSet, 4, num_worker: 0, shuffle: true, device: device);

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
					Tensor[] masks = new Tensor[indexs.Length];
					List<float> batch_idx = new List<float>();
					List<float> cls = new List<float>();
					List<Tensor> bboxes = new List<Tensor>();
					for (int i = 0; i < indexs.Length; i++)
					{
						ImageData imageData = trainDataSet.GetImageAndLabelData(indexs[i]);
						images[i] = Lib.GetTensorFromImage(imageData.ResizedImage).to(device).unsqueeze(0) / 255.0f;
						batch_idx.AddRange(Enumerable.Repeat((float)i, imageData.ResizedLabels.Count));
						cls.AddRange(imageData.ResizedLabels.Select(x => (float)x.LabelID));
						bboxes.AddRange(imageData.ResizedLabels.Select(x => torch.tensor(new float[] { x.CenterX, x.CenterY, x.Width, x.Height })));

						Mat maskMat = new Mat(maskSize, maskSize, MatType.CV_8UC1, new OpenCvSharp.Scalar(0));
						for (int j = 0; j < imageData.ResizedLabels.Count; j++)
						{
							Point2f[] points = imageData.ResizedLabels[j].MaskOutLine.Select(p => p.Multiply((float)maskSize / imageSize)).ToArray();
							Mat eachMaskMat = YoloDataClass.GetMaskFromOutlinePoints(points, maskSize, maskSize);
							Mat foreMat = new Mat(maskSize, maskSize, MatType.CV_8UC1, new OpenCvSharp.Scalar(j + 1f));
							foreMat.CopyTo(maskMat, eachMaskMat);
						}
						masks[i] = Lib.GetTensorFromImage(maskMat, torchvision.io.ImageReadMode.GRAY).to(device).unsqueeze(0);
					}

					Tensor batch_idx_tensor = torch.tensor(batch_idx, dtype: dtype, device: device).view(-1, 1);
					Tensor cls_tensor = torch.tensor(cls, dtype: dtype, device: device).view(-1, 1);
					Tensor bboxes_tensor = torch.stack(bboxes).to(dtype, device) / imageSize;
					Tensor imageTensor = concat(images);

					Tensor maskTensor = concat(masks);

					if (batch_idx.Count < 1)
					{
						continue;
					}

					Dictionary<string, Tensor> targets = new Dictionary<string, Tensor>()
					{
						{ "batch_idx", batch_idx_tensor },
						{ "cls", cls_tensor },
						{ "bboxes", bboxes_tensor },
						{ "masks", maskTensor}
					};

					Tensor[] list = yolo.forward(imageTensor);
					var (ls, ls_item) = loss.forward(list, targets);
					optimizer.zero_grad();
					ls.backward();
					optimizer.step();
					Console.WriteLine($"Process: Epoch {epoch}, Step/Total Step  {step}/{trainDataLoader.Count}");
				}

				Console.Write("Do val now... ");
				float valLoss = Val(valDataSet, valDataLoader, imageSize, maskSize);
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

		private float Val(YoloDataClass valDataSet, DataLoader loader, int imageSize, int maskSize)
		{
			float lossValue = float.MaxValue;
			foreach (var data in loader)
			{
				long[] indexs = data["index"].data<long>().ToArray();
				Tensor[] images = new Tensor[indexs.Length];
				Tensor[] masks = new Tensor[indexs.Length];
				List<float> batch_idx = new List<float>();
				List<float> cls = new List<float>();
				List<Tensor> bboxes = new List<Tensor>();
				for (int i = 0; i < indexs.Length; i++)
				{
					ImageData imageData = valDataSet.GetImageAndLabelData(indexs[i]);
					images[i] = Lib.GetTensorFromImage(imageData.ResizedImage).to(device).unsqueeze(0) / 255.0f;
					batch_idx.AddRange(Enumerable.Repeat((float)i, imageData.ResizedLabels.Count));
					cls.AddRange(imageData.ResizedLabels.Select(x => (float)x.LabelID));
					bboxes.AddRange(imageData.ResizedLabels.Select(x => torch.tensor(new float[] { x.CenterX, x.CenterY, x.Width, x.Height })));

					Mat maskMat = new Mat(maskSize, maskSize, MatType.CV_8UC1, new OpenCvSharp.Scalar(0));
					for (int j = 0; j < imageData.ResizedLabels.Count; j++)
					{
						Point2f[] points = imageData.ResizedLabels[j].MaskOutLine.Select(p => p.Multiply((float)maskSize / imageSize)).ToArray();
						Mat eachMaskMat = YoloDataClass.GetMaskFromOutlinePoints(points, maskSize, maskSize);
						Mat foreMat = new Mat(maskSize, maskSize, MatType.CV_8UC1, new OpenCvSharp.Scalar(j + 1f));
						foreMat.CopyTo(maskMat, eachMaskMat);
					}
					masks[i] = Lib.GetTensorFromImage(maskMat, torchvision.io.ImageReadMode.GRAY).to(device).unsqueeze(0);
				}

				Tensor batch_idx_tensor = torch.tensor(batch_idx, dtype: dtype, device: device).view(-1, 1);
				Tensor cls_tensor = torch.tensor(cls, dtype: dtype, device: device).view(-1, 1);
				Tensor bboxes_tensor = torch.stack(bboxes).to(dtype, device) / imageSize;
				Tensor imageTensor = concat(images);

				Tensor maskTensor = concat(masks);

				if (batch_idx.Count < 1)
				{
					continue;
				}

				Dictionary<string, Tensor> targets = new Dictionary<string, Tensor>()
					{
						{ "batch_idx", batch_idx_tensor },
						{ "cls", cls_tensor },
						{ "bboxes", bboxes_tensor },
						{ "masks", maskTensor}
					};

				Tensor[] list = yolo.forward(imageTensor);
				var (ls, ls_item) = loss.forward(list, targets);
				if (lossValue == float.MaxValue)
				{
					lossValue = ls.ToSingle();
				}
				else
				{
					lossValue = lossValue + ls.ToSingle();
				}
			}
			lossValue = lossValue / valDataSet.Count;
			return lossValue;
		}

		private List<YoloResult> ImagePredict(Tensor orgImage, float PredictThreshold = 0.25f, float IouThreshold = 0.5f, int imgSize = 640)
		{
			yolo.eval();
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
			if (proto.shape[0] > 0)
			{
				if (!Equals(preds[0], null))
				{
					int i = 0;
					Tensor masks = Utils.Ops.process_mask(proto[i], preds[i][.., 6..], preds[i][.., 0..4], new long[] { tensor.shape[2], tensor.shape[3] }, upsample: true);
					preds[i][.., ..4] = preds[i][.., ..4] * gain;
					preds[i][.., ..4] = Utils.Ops.clip_boxes(preds[i][.., ..4], new float[] { orgImage.shape[2], orgImage.shape[3] });
					masks = torchvision.transforms.functional.crop(masks, 0, 0, new_h, new_w);
					masks = torchvision.transforms.functional.resize(masks, (int)orgImage.shape[2], (int)orgImage.shape[3]);

					for (int j = 0; j < masks.shape[0]; j++)
					{
						byte[,] mask = new byte[masks.shape[2], masks.shape[1]];
						Buffer.BlockCopy(masks[j].transpose(0, 1).@byte().data<byte>().ToArray(), 0, mask, 0, mask.Length);

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
					}
				}
			}
			return results;
		}

		public List<YoloResult> ImagePredict(SKBitmap image, float PredictThreshold = 0.25f, float IouThreshold = 0.5f, int imgSize = 640)
		{
			Tensor orgImage = Lib.GetTensorFromImage(image).to(dtype, device);
			return ImagePredict(orgImage, PredictThreshold, IouThreshold, imgSize);
		}

		public List<YoloResult> ImagePredict(string imagePath, float PredictThreshold = 0.25f, float IouThreshold = 0.5f, int imgSize = 640)
		{
			Tensor orgImage = Lib.GetTensorFromImage(imagePath).to(dtype, device);
			return ImagePredict(orgImage, PredictThreshold, IouThreshold, imgSize);
		}

		public List<YoloResult> ImagePredict(Mat image, float PredictThreshold = 0.25f, float IouThreshold = 0.5f, int imgSize = 640)
		{
			Tensor orgImage = Lib.GetTensorFromImage(image).to(dtype, device);
			return ImagePredict(orgImage, PredictThreshold, IouThreshold, imgSize);
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
