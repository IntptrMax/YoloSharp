using Data;
using TorchSharp;
using YoloSharp.Types;
using YoloSharp.Utils;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace YoloSharp.Models
{
	internal class PoseDetector : YoloBaseTaskModel
	{
		internal PoseDetector(Config config)
		{
			this.config = config;
			if (config.YoloType == YoloType.Yolov5 || config.YoloType == YoloType.Yolov5u || config.YoloType == YoloType.Yolov12)
			{
				throw new ArgumentException("Pose not support yolov5, yolov5u or yolov12. Please use yolov8 or yolov11 instead.");
			}

			yolo = config.YoloType switch
			{
				YoloType.Yolov8 => new Yolo.Yolov8Pose(config.NumberClass, config.KeyPointShape, config.YoloSize, config.Device, config.Dtype),
				YoloType.Yolov11 => new Yolo.Yolov11Pose(config.NumberClass, config.KeyPointShape, config.YoloSize, config.Device, config.Dtype),
				_ => throw new NotImplementedException("Yolo type not supported."),
			};
			loss = config.YoloType switch
			{
				YoloType.Yolov8 => new Loss.V8PoseLoss(config.NumberClass, config.KeyPointShape),
				YoloType.Yolov11 => new Loss.V8PoseLoss(config.NumberClass, config.KeyPointShape),
				_ => throw new NotImplementedException("Yolo type not supported."),
			};

			//Tools.TransModelFromSafetensors(yolo, @".\yolov8n-pose.safetensors", @".\PreTrainedModels\yolov8n-pose.bin");
		}


		internal override List<YoloResult> ImagePredict(Tensor orgImage, float predictThreshold, float iouThreshold)
		{
			using (no_grad())
			{
				yolo.eval();
				// Change RGB → BGR
				orgImage = orgImage.to(config.Dtype, config.Device).unsqueeze(0);

				int w = (int)orgImage.shape[3];
				int h = (int)orgImage.shape[2];
				int padHeight = 32 - (int)(orgImage.shape[2] % 32);
				int padWidth = 32 - (int)(orgImage.shape[3] % 32);

				padHeight = padHeight == 32 ? 0 : padHeight;
				padWidth = padWidth == 32 ? 0 : padWidth;

				Tensor input = functional.pad(orgImage, new long[] { 0, padWidth, 0, padHeight }, PaddingModes.Zeros, 114) / 255.0f;
				Tensor[] tensors = yolo.forward(input);
				(List<Tensor> nms_result, var _) = Ops.non_max_suppression(tensors[0], nc: config.NumberClass, conf_thres: predictThreshold, iou_thres: iouThreshold, rotated: true);
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


	}
}
