using OpenCvSharp;
using TorchSharp;
using YoloSharp.Data;
using YoloSharp.Types;
using YoloSharp.Utils;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace YoloSharp.Models
{
	internal class Segmenter : YoloBaseTaskModel
	{
		internal Segmenter(int numberClasses = 80, YoloType yoloType = YoloType.Yolov8, YoloSize yoloSize = YoloSize.n, Types.DeviceType deviceType = Types.DeviceType.CUDA, Types.ScalarType dtype = Types.ScalarType.Float32)
		{
			torchvision.io.DefaultImager = new torchvision.io.SkiaImager();
			if (yoloType == YoloType.Yolov5 || yoloType == YoloType.Yolov5u || yoloType == YoloType.Yolov12)
			{
				throw new ArgumentException("Segmenter not support yolov5, yolov5u or yolov12. Please use yolov8 or yolov11 instead.");
			}

			device = new Device((TorchSharp.DeviceType)deviceType);
			this.dtype = (torch.ScalarType)dtype;
			this.sortCount = numberClasses;
			this.yoloType = yoloType;
			this.taskType = TaskType.Segmentation;

			yolo = yoloType switch
			{
				YoloType.Yolov8 => new Yolo.Yolov8Segment(numberClasses, yoloSize, device, this.dtype),
				YoloType.Yolov11 => new Yolo.Yolov11Segment(numberClasses, yoloSize, device, this.dtype),
				_ => throw new NotImplementedException("Yolo type not supported."),
			};
			loss = yoloType switch
			{
				YoloType.Yolov8 => new Loss.V8SegmentationLoss(this.sortCount),
				YoloType.Yolov11 => new Loss.V8SegmentationLoss(this.sortCount),
				_ => throw new NotImplementedException("Yolo type not supported."),
			};
			//Tools.TransModelFromSafetensors(yolo, @".\yolov8n-seg.safetensors", @".\PreTrainedModels\yolov11x-seg.bin");
		}

		/*
		public void Train(string rootPath, string trainDataPath = "", string valDataPath = "", string outputPath = "output", int imageSize = 640, int epochs = 100, float lr = 0.0001f, int batchSize = 8, int numWorkers = 0, ImageProcessType imageProcessType = ImageProcessType.Letterbox)
		{
			int maskSize = imageSize / 4;
			Console.WriteLine("Model will be write to: " + outputPath);
			Console.WriteLine("Load model...");

			YoloDataset trainDataSet = new YoloDataset(rootPath, trainDataPath, imageSize, TaskType.Segmentation, imageProcessType);
			if (trainDataSet.Count == 0)
			{
				throw new FileNotFoundException("No data found in the path: " + trainDataPath);
			}
			DataLoader trainDataLoader = new DataLoader(trainDataSet, batchSize, num_worker: numWorkers, shuffle: false, device: device);
			valDataPath = string.IsNullOrEmpty(valDataPath) ? trainDataPath : valDataPath;

			YoloDataset valDataSet = new YoloDataset(rootPath, valDataPath, imageSize, TaskType.Segmentation, imageProcessType);
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
					using (NewDisposeScope())
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
							if (imageData.ResizedLabels is not null)
							{
								batch_idx.AddRange(Enumerable.Repeat((float)i, imageData.ResizedLabels.Count));
								cls.AddRange(imageData.ResizedLabels.Select(x => (float)x.LabelID));
								bboxes.AddRange(imageData.ResizedLabels.Select(x => tensor(new float[] { x.CenterX, x.CenterY, x.Width, x.Height })));

								Mat maskMat = new Mat(maskSize, maskSize, MatType.CV_8UC1, new OpenCvSharp.Scalar(0));
								for (int j = 0; j < imageData.ResizedLabels.Count; j++)
								{
									Point[] points = imageData.ResizedLabels[j].MaskOutLine.Select(p => p.Multiply((float)maskSize / imageSize)).ToArray();
									Mat eachMaskMat = YoloDataset.GetMaskFromOutlinePoints(points, maskSize, maskSize);
									Mat foreMat = new Mat(maskSize, maskSize, MatType.CV_8UC1, new OpenCvSharp.Scalar(j + 1f));
									foreMat.CopyTo(maskMat, eachMaskMat);
								}
								masks[i] = Lib.GetTensorFromImage(maskMat, torchvision.io.ImageReadMode.GRAY).to(device).unsqueeze(0);
							}
						}

						Tensor batch_idx_tensor = tensor(batch_idx, dtype: dtype, device: device).view(-1, 1);
						Tensor cls_tensor = tensor(cls, dtype: dtype, device: device).view(-1, 1);
						Tensor bboxes_tensor = stack(bboxes).to(dtype, device) / imageSize;
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

		private float Val(YoloDataset valDataSet, DataLoader loader, int imageSize, int maskSize)
		{
			float lossValue = float.MaxValue;
			foreach (var data in loader)
			{
				using (NewDisposeScope())
				using (no_grad())
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
						if (imageData.ResizedLabels is not null)
						{
							batch_idx.AddRange(Enumerable.Repeat((float)i, imageData.ResizedLabels.Count));
							cls.AddRange(imageData.ResizedLabels.Select(x => (float)x.LabelID));
							bboxes.AddRange(imageData.ResizedLabels.Select(x => tensor(new float[] { x.CenterX, x.CenterY, x.Width, x.Height })));

							Mat maskMat = new Mat(maskSize, maskSize, MatType.CV_8UC1, new OpenCvSharp.Scalar(0));
							for (int j = 0; j < imageData.ResizedLabels.Count; j++)
							{
								Point[] points = imageData.ResizedLabels[j].MaskOutLine.Select(p => p.Multiply((float)maskSize / imageSize)).ToArray();
								Mat eachMaskMat = YoloDataset.GetMaskFromOutlinePoints(points, maskSize, maskSize);
								Mat foreMat = new Mat(maskSize, maskSize, MatType.CV_8UC1, new OpenCvSharp.Scalar(j + 1f));
								foreMat.CopyTo(maskMat, eachMaskMat);
							}
							masks[i] = Lib.GetTensorFromImage(maskMat, torchvision.io.ImageReadMode.GRAY).to(device).unsqueeze(0);
						}
					}

					Tensor batch_idx_tensor = tensor(batch_idx, dtype: dtype, device: device).view(-1, 1);
					Tensor cls_tensor = tensor(cls, dtype: dtype, device: device).view(-1, 1);
					Tensor bboxes_tensor = stack(bboxes).to(dtype, device) / imageSize;
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
			}
			lossValue = lossValue / valDataSet.Count;
			return lossValue;
		}
		*/

		internal override List<YoloResult> ImagePredict(Tensor orgImage, float PredictThreshold = 0.25f, float IouThreshold = 0.5f)
		{
			// Change RGB → BGR
			orgImage = orgImage.to(dtype, device).unsqueeze(0);

			int w = (int)orgImage.shape[3];
			int h = (int)orgImage.shape[2];
			int padHeight = 32 - (int)(orgImage.shape[2] % 32);
			int padWidth = 32 - (int)(orgImage.shape[3] % 32);

			padHeight = padHeight == 32 ? 0 : padHeight;
			padWidth = padWidth == 32 ? 0 : padWidth;

			Tensor input = functional.pad(orgImage, new long[] { 0, padWidth, 0, padHeight }, PaddingModes.Zeros, 114) / 255.0f;
			yolo.eval();

			Tensor[] outputs = yolo.forward(input);

			(List<Tensor> preds, var _) = Ops.non_max_suppression(outputs[0], nc: this.sortCount, conf_thres: PredictThreshold, iou_thres: IouThreshold);
			Tensor proto = outputs[4];

			List<YoloResult> results = new List<YoloResult>();
			if (proto.shape[0] > 0)
			{
				if (!Equals(preds[0], null))
				{
					int i = 0;
					Tensor masks = Ops.process_mask(proto[i], preds[i][.., 6..], preds[i][.., 0..4], new long[] { input.shape[2], input.shape[3] }, upsample: true);
					preds[i][.., ..4] = Ops.clip_boxes(preds[i][.., ..4], new float[] { orgImage.shape[2], orgImage.shape[3] });
					masks = torchvision.transforms.functional.crop(masks, 0, 0, (int)input.shape[2], (int)input.shape[3]);
					masks = torchvision.transforms.functional.resize(masks, (int)orgImage.shape[2], (int)orgImage.shape[3]);

					for (int j = 0; j < masks.shape[0]; j++)
					{
						byte[,] mask = new byte[masks.shape[2], masks.shape[1]];
						Buffer.BlockCopy(masks[j].transpose(0, 1).@byte().data<byte>().ToArray(), 0, mask, 0, mask.Length);

						int x = preds[i][j, 0].ToInt32();
						int y = preds[i][j, 1].ToInt32();

						int ww = preds[i][j, 2].ToInt32() - x;
						int hh = preds[i][j, 3].ToInt32() - y;

						results.Add(new YoloResult()
						{
							ClassID = preds[i][j, 5].ToInt32(),
							Score = preds[i][j, 4].ToSingle(),
							CenterX = x + ww / 2,
							CenterY = y + hh / 2,
							Width = ww,
							Height = hh,
							Mask = mask
						});
					}
				}
			}
			return results;
		}

		internal override Dictionary<string, Tensor> GetTargets(long[] indexs, YoloDataset dataset)
		{
			int maskSize = dataset.ImageSize / 4;
			using (NewDisposeScope())
			using (no_grad())
			{
				Tensor[] images = new Tensor[indexs.Length];
				Tensor[] masks = new Tensor[indexs.Length];
				List<float> batch_idx = new List<float>();
				List<float> cls = new List<float>();
				List<Tensor> bboxes = new List<Tensor>();
				for (int i = 0; i < indexs.Length; i++)
				{
					ImageData imageData = dataset.GetImageAndLabelData(indexs[i]);
					images[i] = Lib.GetTensorFromImage(imageData.ResizedImage).to(device).unsqueeze(0) / 255.0f;
					if (imageData.ResizedLabels is not null)
					{
						batch_idx.AddRange(Enumerable.Repeat((float)i, imageData.ResizedLabels.Count));
						cls.AddRange(imageData.ResizedLabels.Select(x => (float)x.LabelID));
						bboxes.AddRange(imageData.ResizedLabels.Select(x => tensor(new float[] { x.CenterX, x.CenterY, x.Width, x.Height })));

						using (Mat maskMat = new Mat(maskSize, maskSize, MatType.CV_8UC1, new OpenCvSharp.Scalar(0)))
						{
							for (int j = 0; j < imageData.ResizedLabels.Count; j++)
							{
								Point[] points = imageData.ResizedLabels[j].MaskOutLine.Select(p => p.Multiply((float)maskSize / dataset.ImageSize)).ToArray();
								Mat eachMaskMat = YoloDataset.GetMaskFromOutlinePoints(points, maskSize, maskSize);
								Mat foreMat = new Mat(maskSize, maskSize, MatType.CV_8UC1, new OpenCvSharp.Scalar(j + 1f));
								foreMat.CopyTo(maskMat, eachMaskMat);
							}
							masks[i] = Lib.GetTensorFromImage(maskMat, torchvision.io.ImageReadMode.GRAY).to(device).unsqueeze(0);
						}
					}
				}

				Tensor batch_idx_tensor = tensor(batch_idx, dtype: dtype, device: device).view(-1, 1);
				Tensor cls_tensor = tensor(cls, dtype: dtype, device: device).view(-1, 1);
				Tensor bboxes_tensor = stack(bboxes).to(dtype, device) / dataset.ImageSize;
				Tensor imageTensor = concat(images);

				Tensor maskTensor = concat(masks);

				Dictionary<string, Tensor> targets = new Dictionary<string, Tensor>()
				{
					{ "batch_idx", batch_idx_tensor.MoveToOuterDisposeScope() },
					{ "cls", cls_tensor.MoveToOuterDisposeScope() },
					{ "bboxes", bboxes_tensor.MoveToOuterDisposeScope() },
					{ "masks", maskTensor.MoveToOuterDisposeScope() },
					{ "images", imageTensor.MoveToOuterDisposeScope() },
				};
				GC.Collect();
				return targets;
			}
		}


	}
}
