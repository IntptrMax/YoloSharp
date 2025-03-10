﻿using System.Drawing;
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
				YoloType.Yolov5 => new Yolov5(socrCount, yoloSize),
				YoloType.Yolov5u => new Yolov5u(socrCount, yoloSize),
				YoloType.Yolov8 => new Yolov8(socrCount, yoloSize),
				YoloType.Yolov11 => new Yolov11(socrCount, yoloSize),
				YoloType.Yolov12 => new Yolov12(socrCount, yoloSize),
				_ => throw new NotImplementedException(),
			};

			yolo.to(this.device, this.dtype);

			loss = yoloType switch
			{
				YoloType.Yolov5 => new Loss.Yolov5DetectionLoss(this.socrCount),
				YoloType.Yolov5u => new Loss.YoloDetectionLoss(this.socrCount),
				YoloType.Yolov8 => new Loss.YoloDetectionLoss(this.socrCount),
				YoloType.Yolov11 => new Loss.YoloDetectionLoss(this.socrCount),
				YoloType.Yolov12 => new Loss.YoloDetectionLoss(this.socrCount),
				_ => throw new NotImplementedException(),
			};
			loss = loss.to(this.device, this.dtype);

			predict = yoloType switch
			{
				YoloType.Yolov5 => new Predict.Yolov5Predict(),
				YoloType.Yolov5u => new Predict.YoloPredict(),
				YoloType.Yolov8 => new Predict.YoloPredict(),
				YoloType.Yolov11 => new Predict.YoloPredict(),
				YoloType.Yolov12 => new Predict.YoloPredict(),
				_ => throw new NotImplementedException(),
			};

			//Tools.TransModelFromSafetensors(yolo, @"D:\DeepLearning\yolo\ultralytics\yolov12x.safetensors", "yolov12x.bin");

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

		public List<PredictResult> ImagePredict(Bitmap image, float PredictThreshold = 0.25f, float IouThreshold = 0.5f)
		{
			Tensor orgImage = Lib.GetTensorFromBitmap(image).to(dtype, device);
			orgImage = torch.stack(new Tensor[] { orgImage[2], orgImage[1], orgImage[0] }, dim: 0).unsqueeze(0) / 255.0f;
			int w = (int)orgImage.shape[3];
			int h = (int)orgImage.shape[2];
			int padHeight = 32 - (int)(orgImage.shape[2] % 32);
			int padWidth = 32 - (int)(orgImage.shape[3] % 32);

			padHeight = padHeight == 32 ? 0 : padHeight;
			padWidth = padWidth == 32 ? 0 : padWidth;

			Tensor input = torch.nn.functional.pad(orgImage, [0, padWidth, 0, padHeight], PaddingModes.Zeros);
			yolo.eval();

			Tensor[] tensors = yolo.forward(input);

			//Predict.YoloPredict predict = new Predict.YoloPredict(PredictThreshold, IouThreshold);
			Tensor results = predict.forward(tensors[0], PredictThreshold, IouThreshold);
			List<PredictResult> predResults = new List<PredictResult>();
			for (int i = 0; i < results.shape[0]; i++)
			{
				int rw = results[i, 2].ToInt32();
				int rh = results[i, 3].ToInt32();
				int x = results[i, 0].ToInt32() - rw / 2;
				int y = results[i, 1].ToInt32() - rh / 2;

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
				yolo.to(modelType);
				List<string> skipList = new List<string>();
				long nc = 0;

				if (skipNcNotEqualLayers)
				{
					switch (yoloType)
					{
						case YoloType.Yolov5:
							{
								skipList = state_dict.Keys.Select(x => x).Where(x => x.Contains("model.24.m")).ToList();
								nc = state_dict[skipList[0]].shape[0] / 3 - 5;
								break;
							}
						case YoloType.Yolov5u:
							{
								skipList = state_dict.Keys.Select(x => x).Where(x => x.Contains("model.24.cv3")).ToList();
								nc = state_dict[skipList[0]].shape[0];
								break;
							}
						case YoloType.Yolov8:
							{
								skipList = state_dict.Keys.Select(x => x).Where(x => x.Contains("model.22.cv3")).ToList();
								nc = state_dict[skipList[0]].shape[0];
								break;
							}
						case YoloType.Yolov11:
							{
								skipList = state_dict.Keys.Select(x => x).Where(x => x.Contains("model.23.cv3")).ToList();
								nc = state_dict[skipList[3]].shape[0];
								break;
							}
						case YoloType.Yolov12:
							{
								skipList = state_dict.Keys.Select(x => x).Where(x => x.Contains("model.21.cv3")).ToList();
								nc = state_dict[skipList[12]].shape[0];
								break;
							}
						default:
							break;
					}
					if (nc == socrCount)
					{
						skipList.Clear();
					}
				};

				yolo.to(modelType);
				var (miss, err) = yolo.load_state_dict(state_dict, skip: skipList);
				if (skipList.Count > 0)
				{
					Console.WriteLine("Waring! You are skipping nc reference layers.");
					Console.WriteLine("This will get wrong result in Predict, sort count in weight loaded is " + nc);
				}
				yolo.to(dtype);
			}
		}

	}
}
