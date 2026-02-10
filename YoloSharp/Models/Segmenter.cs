using Data;
using TorchSharp;
using YoloSharp.Types;
using YoloSharp.Utils;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace YoloSharp.Models
{
	internal class Segmenter : YoloBaseTaskModel
	{
		internal Segmenter(Config config)
		{
			this.config = config;
			if (config.YoloType == YoloType.Yolov5 || config.YoloType == YoloType.Yolov5u || config.YoloType == YoloType.Yolov12)
			{
				throw new ArgumentException("Segmenter not support yolov5, yolov5u or yolov12. Please use yolov8 or yolov11 instead.");
			}

			yolo = config.YoloType switch
			{
				YoloType.Yolov8 => new Yolo.Yolov8Segment(config.NumberClass, config.YoloSize, config.Device, config.Dtype),
				YoloType.Yolov11 => new Yolo.Yolov11Segment(config.NumberClass, config.YoloSize, config.Device, config.Dtype),
				_ => throw new NotImplementedException("Yolo type not supported."),
			};
			loss = config.YoloType switch
			{
				YoloType.Yolov8 => new Loss.V8SegmentationLoss(config.NumberClass),
				YoloType.Yolov11 => new Loss.V8SegmentationLoss(config.NumberClass),
				_ => throw new NotImplementedException("Yolo type not supported."),
			};
			//Tools.TransModelFromSafetensors(yolo, @".\yolov8n-seg.safetensors", @".\PreTrainedModels\yolov11x-seg.bin");
		}

		internal override List<YoloResult> ImagePredict(Tensor orgImage, float predictThreshold, float iouThreshold)
		{
			// Change RGB → BGR
			orgImage = orgImage.to(config.Dtype, config.Device).unsqueeze(0);

			int w = (int)orgImage.shape[3];
			int h = (int)orgImage.shape[2];
			int padHeight = 32 - (int)(orgImage.shape[2] % 32);
			int padWidth = 32 - (int)(orgImage.shape[3] % 32);

			padHeight = padHeight == 32 ? 0 : padHeight;
			padWidth = padWidth == 32 ? 0 : padWidth;

			Tensor input = functional.pad(orgImage, new long[] { 0, padWidth, 0, padHeight }, PaddingModes.Zeros, 114) / 255.0f;
			yolo.eval();

			Tensor[] outputs = yolo.forward(input);

			(List<Tensor> preds, var _) = Ops.non_max_suppression(outputs[0], nc: this.config.NumberClass, conf_thres: predictThreshold, iou_thres: iouThreshold);
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

	}
}
