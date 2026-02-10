using Data;
using TorchSharp;
using YoloSharp.Data;
using YoloSharp.Types;
using YoloSharp.Utils;
using static TorchSharp.torch;

namespace YoloSharp.Models
{
	internal class Classifier : YoloBaseTaskModel
	{
		public Classifier(Config config)
		{
			this.config = config;
			yolo = config.YoloType switch
			{
				YoloType.Yolov8 => new Yolo.Yolov8Classify(config.NumberClass, config.YoloSize, config.Device, config.Dtype),
				YoloType.Yolov11 => new Yolo.Yolov11Classify(config.NumberClass, config.YoloSize, config.Device, config.Dtype),
				_ => throw new NotImplementedException("Yolo type not supported."),
			};
			loss = config.YoloType switch
			{
				YoloType.Yolov5u => new Loss.V8ClassificationLoss(),
				YoloType.Yolov8 => new Loss.V8ClassificationLoss(),
				YoloType.Yolov11 => new Loss.V8ClassificationLoss(),
				YoloType.Yolov12 => new Loss.V8ClassificationLoss(),
				_ => throw new NotImplementedException("Yolo type not supported."),
			};

			// Tools.TransModelFromSafetensors(yolo, @".\yolov8n-cls.safetensors", @".\PreTrainedModels\yolov8n-cls.bin");
		}

		internal Dictionary<string, torch.Tensor> GetTargets(long[] indexs, YoloDataset dataset)
		{
			using (NewDisposeScope())
			using (no_grad())
			{
				Tensor[] images = new Tensor[indexs.Length];
				List<float> batch_idx = new List<float>();
				List<float> cls = new List<float>();
				for (int i = 0; i < indexs.Length; i++)
				{
					ImageData imageData = dataset.GetImageAndLabelData(indexs[i]);
					images[i] = Lib.GetTensorFromImage(imageData.ResizedImage).to(config.Device).unsqueeze(0) / 255.0f;
					if (imageData.ResizedLabels is not null)
					{
						batch_idx.AddRange(Enumerable.Repeat((float)i, imageData.ResizedLabels.Count));
						cls.AddRange(imageData.ResizedLabels.Select(x => (float)x.LabelID));
					}
				}

				torchvision.ITransform[] transformers = this.yolo.training switch
				{
					true => new torchvision.ITransform[] {
						 torchvision.transforms.RandomHorizontalFlip(p: 0.3),
						 torchvision.transforms.RandomVerticalFlip(0),
						 torchvision.transforms.RandomRotation(15),
						 torchvision.transforms.RandomPerspective(0.2, 0.3),
						 torchvision.transforms.ColorJitter(brightness: 0.2f, contrast: 0.2f, saturation: 0.2f, hue: 0.1f),
					 },
					_ => new torchvision.ITransform[] { }
				};

				Tensor batch_idx_tensor = tensor(batch_idx, dtype: config.Dtype, device: config.Device).view(-1, 1);
				Tensor cls_tensor = tensor(cls, dtype: torch.ScalarType.Int64, device: config.Device);
				Tensor imageTensor = concat(images);
				torchvision.ITransform transformer = torchvision.transforms.Compose(transformers);
				imageTensor = transformer.call(imageTensor).to(config.Dtype, config.Device);

				Dictionary<string, Tensor> targets = new Dictionary<string, Tensor>()
				{
					{ "batch_idx", batch_idx_tensor.MoveToOuterDisposeScope() },
					{ "cls", cls_tensor.MoveToOuterDisposeScope() },
					{ "images", imageTensor.MoveToOuterDisposeScope()}
				};

				GC.Collect();
				return targets;
			}
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

				Tensor input = torch.nn.functional.pad(orgImage, new long[] { 0, padWidth, 0, padHeight }, PaddingModes.Zeros, 114) / 255.0f;
				Tensor[] tensors = yolo.forward(input);
				List<YoloResult> results = new List<YoloResult>();
				for (int i = 0; i < config.NumberClass; i++)
				{
					results.Add(new YoloResult()
					{
						ClassID = i,
						Score = tensors[0][0][i].ToSingle(),
					});
				}
				results.Sort((a, b) => b.Score.CompareTo(a.Score));
				return results;
			}
		}
	}
}
