using TorchSharp;
using YoloSharp.Data;
using YoloSharp.Types;
using YoloSharp.Utils;
using static TorchSharp.torch;

namespace YoloSharp.Models
{
	internal class Classifier : YoloBaseTaskModel
	{
		public Classifier(int numberClasses = 80, YoloType yoloType = YoloType.Yolov8, YoloSize yoloSize = YoloSize.n, Types.DeviceType deviceType = Types.DeviceType.CUDA, Types.ScalarType dtype = Types.ScalarType.Float32)
		{
			torchvision.io.DefaultImager = new torchvision.io.SkiaImager();

			device = new Device((TorchSharp.DeviceType)deviceType);
			this.dtype = (torch.ScalarType)dtype;
			this.sortCount = numberClasses;
			this.yoloType = yoloType;
			this.taskType = TaskType.Classification;

			yolo = yoloType switch
			{
				YoloType.Yolov8 => new Yolo.Yolov8Classify(numberClasses, yoloSize, device, this.dtype),
				YoloType.Yolov11 => new Yolo.Yolov11Classify(numberClasses, yoloSize, device, this.dtype),
				_ => throw new NotImplementedException("Yolo type not supported."),
			};
			loss = yoloType switch
			{
				YoloType.Yolov5u => new Loss.V8ClassificationLoss(),
				YoloType.Yolov8 => new Loss.V8ClassificationLoss(),
				YoloType.Yolov11 => new Loss.V8ClassificationLoss(),
				YoloType.Yolov12 => new Loss.V8ClassificationLoss(),
				_ => throw new NotImplementedException("Yolo type not supported."),
			};

			// Tools.TransModelFromSafetensors(yolo, @".\yolov8n-cls.safetensors", @".\PreTrainedModels\yolov8n-cls.bin");
		}

		internal override Dictionary<string, torch.Tensor> GetTargets(long[] indexs, YoloDataset dataset)
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
					images[i] = Lib.GetTensorFromImage(imageData.ResizedImage).to(device).unsqueeze(0) / 255.0f;
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

				Tensor batch_idx_tensor = tensor(batch_idx, dtype: dtype, device: device).view(-1, 1);
				Tensor cls_tensor = tensor(cls, dtype: torch.ScalarType.Int64, device: device);
				Tensor imageTensor = concat(images);
				torchvision.ITransform transformer = torchvision.transforms.Compose(transformers);
				imageTensor = transformer.call(imageTensor);

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

		internal override List<YoloResult> ImagePredict(torch.Tensor orgImage, float PredictThreshold = 0.25F, float IouThreshold = 0.5F)
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

				Tensor input = torch.nn.functional.pad(orgImage, new long[] { 0, padWidth, 0, padHeight }, PaddingModes.Zeros, 114) / 255.0f;
				Tensor[] tensors = yolo.forward(input);
				List<YoloResult> results = new List<YoloResult>();
				for (int i = 0; i < sortCount; i++)
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
