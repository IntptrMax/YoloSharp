using TorchSharp;
using YoloSharp.Data;
using YoloSharp.Types;
using YoloSharp.Utils;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace YoloSharp.Models
{
	internal class PoseDetector : YoloBaseTaskModel
	{
		internal PoseDetector(int numberClasses = 80, int[] kpt_shape = null, YoloType yoloType = YoloType.Yolov8, YoloSize yoloSize = YoloSize.n, Types.DeviceType deviceType = Types.DeviceType.CUDA, Types.ScalarType dtype = Types.ScalarType.Float32)
		{
			torchvision.io.DefaultImager = new torchvision.io.SkiaImager();
			if (yoloType == YoloType.Yolov5 || yoloType == YoloType.Yolov5u || yoloType == YoloType.Yolov12)
			{
				throw new ArgumentException("Pose not support yolov5, yolov5u or yolov12. Please use yolov8 or yolov11 instead.");
			}

			device = new Device((TorchSharp.DeviceType)deviceType);
			this.dtype = (torch.ScalarType)dtype;
			this.sortCount = numberClasses;
			this.yoloType = yoloType;
			this.taskType = TaskType.Pose;
			this.keyPointsShape = kpt_shape;

			yolo = yoloType switch
			{
				YoloType.Yolov8 => new Yolo.Yolov8Pose(numberClasses, kpt_shape, yoloSize, device, this.dtype),
				YoloType.Yolov11 => new Yolo.Yolov8Pose(numberClasses, kpt_shape, yoloSize, device, this.dtype),
				_ => throw new NotImplementedException("Yolo type not supported."),
			};
			loss = yoloType switch
			{
				YoloType.Yolov8 => new Loss.V8PoseLoss(numberClasses, kpt_shape),
				YoloType.Yolov11 => new Loss.V8PoseLoss(numberClasses, kpt_shape),
				_ => throw new NotImplementedException("Yolo type not supported."),
			};

			//Tools.TransModelFromSafetensors(yolo, @".\yolov8n-pose.safetensors", @".\PreTrainedModels\yolov8n-pose.bin");
		}


		internal override List<YoloResult> ImagePredict(Tensor orgImage, float PredictThreshold = 0.25F, float IouThreshold = 0.5F)
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
							result.ClassID = nms_result[0][i][5].ToInt32();
							result.Score = nms_result[0][i][4].ToSingle();
							long keyPointsCount = (nms_result[0].shape[1] - 6) / 3;
							Types.KeyPoint[] keyPoints = new Types.KeyPoint[keyPointsCount];
							for (int j = 0; j < keyPointsCount; j++)
							{
								keyPoints[j] = new Types.KeyPoint()
								{
									X = nms_result[0][i][6 + j * 3].ToSingle(),
									Y = nms_result[0][i][6 + j * 3 + 1].ToSingle(),
									VisibilityScore = nms_result[0][i][6 + j * 3 + 2].ToSingle()
								};
							}
							result.KeyPoints = keyPoints;

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
				List<Tensor> kpts = new List<Tensor>();
				for (int i = 0; i < indexs.Length; i++)
				{
					ImageData imageData = dataset.GetImageAndLabelData(indexs[i]);
					images[i] = Lib.GetTensorFromImage(imageData.ResizedImage).to(device).unsqueeze(0) / 255.0f;
					if (imageData.ResizedLabels is not null)
					{
						batch_idx.AddRange(Enumerable.Repeat((float)i, imageData.ResizedLabels.Count));
						cls.AddRange(imageData.ResizedLabels.Select(x => (float)x.LabelID));
						bboxes.AddRange(imageData.ResizedLabels.Select(x => tensor(new float[] { x.CenterX, x.CenterY, x.Width, x.Height })));
						imageData.ResizedLabels.Select(x =>
						{
							float[] kpt_array = new float[x.KeyPoints.Count() * 3];
							for (int j = 0; j < x.KeyPoints.Count(); j++)
							{
								kpt_array[j * 3] = x.KeyPoints[j].X / imageData.ResizedImage.Width;
								kpt_array[j * 3 + 1] = x.KeyPoints[j].Y / imageData.ResizedImage.Height;
								kpt_array[j * 3 + 2] = x.KeyPoints[j].VisibilityScore;
							}
							return tensor(kpt_array).view(x.KeyPoints.Count(), 3);
						}).ToList().ForEach(x => kpts.Add(x.unsqueeze(0)));
					}
				}

				Tensor batch_idx_tensor = tensor(batch_idx, dtype: dtype, device: device).view(-1, 1);
				Tensor cls_tensor = tensor(cls, dtype: dtype, device: device).view(-1, 1);
				Tensor bboxes_tensor = stack(bboxes).to(dtype, device) / dataset.ImageSize;
				Tensor imageTensor = concat(images);
				Tensor kpts_tensor = concat(kpts);

				Dictionary<string, Tensor> targets = new Dictionary<string, Tensor>()
				{
					{ "batch_idx", batch_idx_tensor.MoveToOuterDisposeScope() },
					{ "cls", cls_tensor.MoveToOuterDisposeScope() },
					{ "bboxes", bboxes_tensor.MoveToOuterDisposeScope() },
					{ "images", imageTensor.MoveToOuterDisposeScope() },
					{ "keypoints", kpts_tensor.MoveToOuterDisposeScope()}
				};
				GC.Collect();
				return targets;
			}
		}


	}
}
