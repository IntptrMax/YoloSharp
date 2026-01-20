using TorchSharp;
using YoloSharp.Data;
using YoloSharp.Types;
using YoloSharp.Utils;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace YoloSharp.Models
{
	internal class Detector : YoloBaseTaskModel
	{
		private Module<Tensor, float, float, Tensor> predict;

		internal Detector(int numberClasses = 80, YoloType yoloType = YoloType.Yolov8, YoloSize yoloSize = YoloSize.n, Types.DeviceType deviceType = Types.DeviceType.CUDA, Types.ScalarType dtype = Types.ScalarType.Float32)
		{
			torchvision.io.DefaultImager = new torchvision.io.SkiaImager();

			device = new Device((TorchSharp.DeviceType)deviceType);
			this.dtype = (torch.ScalarType)dtype;
			this.sortCount = numberClasses;
			this.yoloType = yoloType;
			this.taskType = TaskType.Detection;

			yolo = yoloType switch
			{
				YoloType.Yolov5 => new Yolo.Yolov5(numberClasses, yoloSize, device, this.dtype),
				YoloType.Yolov5u => new Yolo.Yolov5u(numberClasses, yoloSize, device, this.dtype),
				YoloType.Yolov8 => new Yolo.Yolov8(numberClasses, yoloSize, device, this.dtype),
				YoloType.Yolov11 => new Yolo.Yolov11(numberClasses, yoloSize, device, this.dtype),
				YoloType.Yolov12 => new Yolo.Yolov12(numberClasses, yoloSize, device, this.dtype),
				_ => throw new NotImplementedException(),
			};

			loss = yoloType switch
			{
				YoloType.Yolov5 => new Loss.V5DetectionLoss(this.sortCount),
				YoloType.Yolov5u => new Loss.V8DetectionLoss(this.sortCount),
				YoloType.Yolov8 => new Loss.V8DetectionLoss(this.sortCount),
				YoloType.Yolov11 => new Loss.V8DetectionLoss(this.sortCount),
				YoloType.Yolov12 => new Loss.V8DetectionLoss(this.sortCount),
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

		internal override Dictionary<string, Tensor> GetTargets(long[] indexs, YoloDataset dataset)
		{
			using (NewDisposeScope())
			using (no_grad())
			{
				Tensor[] images = new Tensor[indexs.Length];
				List<float> batch_idx = new List<float>();
				List<float> cls = new List<float>();
				List<Tensor> bboxes = new List<Tensor>();
				for (int i = 0; i < indexs.Length; i++)
				{
					ImageData imageData = dataset.GetImageAndLabelData(indexs[i]);
					images[i] = Lib.GetTensorFromImage(imageData.ResizedImage).to(dtype, device).unsqueeze(0) / 255.0f;
					if (imageData.ResizedLabels is not null)
					{
						batch_idx.AddRange(Enumerable.Repeat((float)i, imageData.ResizedLabels.Count));
						cls.AddRange(imageData.ResizedLabels.Select(x => (float)x.LabelID));
						bboxes.AddRange(imageData.ResizedLabels.Select(x => tensor(new float[] { x.CenterX, x.CenterY, x.Width, x.Height })));
					}
				}

				Tensor batch_idx_tensor = tensor(batch_idx, dtype: dtype, device: device).view(-1, 1);
				Tensor cls_tensor = tensor(cls, dtype: dtype, device: device).view(-1, 1);
				Tensor bboxes_tensor = bboxes.Count == 0 ? zeros(new long[] { 0, 4 }) : stack(bboxes).to(dtype, device) / dataset.ImageSize;
				Tensor imageTensor = concat(images);

				Dictionary<string, Tensor> targets = new Dictionary<string, Tensor>()
				{
					{ "batch_idx", batch_idx_tensor.MoveToOuterDisposeScope() },
					{ "cls", cls_tensor.MoveToOuterDisposeScope() },
					{ "bboxes", bboxes_tensor.MoveToOuterDisposeScope() },
					{ "images", imageTensor.MoveToOuterDisposeScope()}
				};
				GC.Collect();
				return targets;
			}
		}

		internal override List<YoloResult> ImagePredict(Tensor orgImage, float PredictThreshold = 0.25f, float IouThreshold = 0.5f)
		{
			using (no_grad())
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

				Tensor[] tensors = yolo.forward(input);
				Tensor outputs = predict.forward(tensors[0], PredictThreshold, IouThreshold);
				List<YoloResult> predResults = new List<YoloResult>();
				for (int i = 0; i < outputs.shape[0]; i++)
				{
					int x = outputs[i][0].ToInt32();
					int y = outputs[i][1].ToInt32();
					int rw = outputs[i][2].ToInt32() - x;
					int rh = outputs[i][3].ToInt32() - y;

					float score = outputs[i][4].ToSingle();
					int sort = outputs[i][5].ToInt32();

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
					x = cat(new Tensor[] { box, conf.values, j.to_type(scalType) }, 1)[conf.values.view(-1) > confThreshold];

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
