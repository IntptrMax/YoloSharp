using System.Text.RegularExpressions;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;
using static YoloSharp.Yolo;

namespace YoloSharp
{
	public class Segmenter
	{
		public class SegmentResult
		{
			public int ClassID;
			public float Score;
			public int X;
			public int Y;
			public int W;
			public int H;
			public bool[,] Mask;
		}

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
				YoloType.Yolov8 => new Loss.SegmentationLoss(this.sortCount),
				YoloType.Yolov11 => new Loss.SegmentationLoss(this.sortCount),
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
						var (img, lb, mask) = trainDataSet.GetSegmentDataTensor(indexs[i]);
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


		public (List<SegmentResult>, ImageMagick.MagickImage) ImagePredict(ImageMagick.MagickImage image, float PredictThreshold = 0.25f, float IouThreshold = 0.5f, int imgSize = 640)
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

			List<Tensor> preds = NonMaxSuppression(outputs[0], PredictThreshold, IouThreshold);
			Tensor proto = outputs[4];

			List<SegmentResult> results = new List<SegmentResult>();
			ImageMagick.MagickImage maskBitmap = new ImageMagick.MagickImage();
			if (proto.shape[0] > 0)
			{
				if (!Equals(preds[0], null))
				{
					int i = 0;
					Tensor masks = process_mask(proto[i], preds[i][.., 6..], preds[i][.., 0..4], new long[] { tensor.shape[2], tensor.shape[3] }, upsample: true);
					preds[i][.., ..4] = preds[i][.., ..4] * gain;
					preds[i][.., ..4] = Lib.ClipBox(preds[i][.., ..4], new float[] { orgImage.shape[2], orgImage.shape[3] });
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

						results.Add(new SegmentResult()
						{
							ClassID = preds[i][j, 5].ToInt32(),
							Score = preds[i][j, 4].ToSingle(),
							X = x,
							Y = y,
							W = w,
							H = h,
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
					string? layerPattern = yoloType switch
					{
						YoloType.Yolov8 => @"model\.22\.cv3",
						YoloType.Yolov11 => @"model\.23\.cv3",
						_ => null
					};

					if (layerPattern != null)
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

		private List<Tensor> NonMaxSuppression(Tensor prediction, float confThreshold = 0.25f, float iouThreshold = 0.45f, bool agnostic = false, int max_det = 300, int nc = 80)
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
			var nm = prediction.shape[1] - nc - 4; // number of mask
			var mi = 4 + nc; // mask start index
			var xc = prediction[TensorIndex.Colon, 4..(int)mi].amax(1) > confThreshold; // candidates

			prediction = prediction.transpose(1, 2);
			// Settings
			var max_wh = 7680; // maximum box width and height
			var max_nms = 30000; // maximum number of boxes into torchvision.ops.nms()
			var time_limit = 0.5f + 0.05f * bs; // seconds to quit after

			var t = DateTime.Now;

			var output = new List<Tensor>(new Tensor[bs]);
			for (int xi = 0; xi < bs; xi++)
			{
				var x = prediction[xi];
				x = x[xc[xi]]; // confidence

				Tensor[] box_cls_mask = x.split(new long[] { 4, nc, nm }, 1);
				Tensor box = torchvision.ops.box_convert(box_cls_mask[0], torchvision.ops.BoxFormats.cxcywh, torchvision.ops.BoxFormats.xyxy); // box
				Tensor cls = box_cls_mask[1]; // class
				Tensor mask = box_cls_mask[2]; // mask
											   // Box/Mask

				// Detections matrix nx6 (xyxy, conf, cls)

				var conf = x[TensorIndex.Colon, TensorIndex.Slice(4, mi)].max(1, true);
				var j = conf.indexes;
				x = torch.cat(new Tensor[] { box, conf.values, j.to_type(scalType), mask }, 1)[conf.values.view(-1) > confThreshold];

				var n = x.shape[0]; // number of boxes
				if (n == 0)
				{
					continue;
				}
				if (n > max_nms)
				{
					x = x[x[TensorIndex.Ellipsis, 4].argsort(descending: true)][TensorIndex.Slice(0, max_nms)]; // sort by confidence and remove excess boxes
				}
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

		private Tensor process_mask(Tensor protos, Tensor masks_in, Tensor bboxes, long[] shape, bool upsample = false)
		{
			long c = protos.shape[0]; //  # CHW
			long mh = protos.shape[1];
			long mw = protos.shape[2];

			long ih = shape[0];
			long iw = shape[1];
			Tensor protos_reshaped = protos.view(c, -1);
			Tensor masks = masks_in.matmul(protos_reshaped);  //  # CHW
			masks = masks.view(-1, mh, mw);
			float width_ratio = (float)mw / iw;
			float height_ratio = (float)mh / ih;

			Tensor downsampled_bboxes = bboxes.clone();
			downsampled_bboxes[.., 0] *= width_ratio;
			downsampled_bboxes[.., 2] *= width_ratio;
			downsampled_bboxes[.., 3] *= height_ratio;
			downsampled_bboxes[.., 1] *= height_ratio;
			masks = crop_mask(masks, downsampled_bboxes); //  # CHW

			if (upsample)
			{
				masks = torch.nn.functional.interpolate(masks[TensorIndex.None], size: shape, mode: InterpolationMode.Bilinear, align_corners: false)[0];// # CHW
			}
			return masks.gt_(0.0);

		}

		/// <summary>
		/// It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box.
		/// </summary>
		/// <param name="masks">[n, h, w] tensor of masks</param>
		/// <param name="boxes">[n, 4] tensor of bbox coordinates in relative point form</param>
		/// <returns>The masks are being cropped to the bounding box.</returns>
		private Tensor crop_mask(Tensor masks, Tensor boxes)
		{
			long h = masks.shape[1];
			long w = masks.shape[2];
			Tensor[] x1y1x2y2 = torch.chunk(boxes[.., .., TensorIndex.None], 4, 1);  // x1 shape(n,1,1)
			Tensor x1 = x1y1x2y2[0];
			Tensor y1 = x1y1x2y2[1];
			Tensor x2 = x1y1x2y2[2];
			Tensor y2 = x1y1x2y2[3];
			Tensor r = torch.arange(w, device: masks.device, dtype: x1.dtype)[TensorIndex.None, TensorIndex.None, ..];  // rows shape(1,1,w)
			Tensor c = torch.arange(h, device: masks.device, dtype: x1.dtype)[TensorIndex.None, .., TensorIndex.None];  // cols shape(1,h,1)

			return (masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2)));
		}

	}
}
