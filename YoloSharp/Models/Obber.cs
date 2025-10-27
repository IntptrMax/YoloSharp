using TorchSharp;
using YoloSharp.Data;
using YoloSharp.Types;
using YoloSharp.Utils;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace YoloSharp.Models
{
	internal class Obber : YoloBaseTaskModel
	{
		internal Obber(int numberClasses = 15, YoloType yoloType = YoloType.Yolov8, YoloSize yoloSize = YoloSize.n, Types.DeviceType deviceType = Types.DeviceType.CUDA, Types.ScalarType dtype = Types.ScalarType.Float32)
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
			this.taskType = TaskType.Obb;

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

		internal override List<YoloResult> ImagePredict(Tensor orgImage, float PredictThreshold = 0.25f, float IouThreshold = 0.5f)
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
					images[i] = Lib.GetTensorFromImage(imageData.ResizedImage).to(device).unsqueeze(0) / 255.0f;
					if (imageData.ResizedLabels is not null)
					{
						batch_idx.AddRange(Enumerable.Repeat((float)i, imageData.ResizedLabels.Count));
						cls.AddRange(imageData.ResizedLabels.Select(x => (float)x.LabelID));
						bboxes.AddRange(imageData.ResizedLabels.Select(x => tensor(new float[] { x.CenterX / dataset.ImageSize, x.CenterY / dataset.ImageSize, x.Width / dataset.ImageSize, x.Height / dataset.ImageSize, x.Radian })));
					}
				}

				Tensor batch_idx_tensor = tensor(batch_idx, dtype: dtype, device: device).view(-1, 1);
				Tensor cls_tensor = tensor(cls, dtype: dtype, device: device).view(-1, 1);
				Tensor bboxes_tensor = bboxes.Count == 0 ? zeros(new long[] { 0, 5 }) : stack(bboxes).to(dtype, device);
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

	}
}
