using Data;
using OpenCvSharp;
using SkiaSharp;
using TorchSharp;
using YoloSharp.Types;
using YoloSharp.Utils;
using static TorchSharp.torch;

namespace YoloSharp.Models
{
	public class YoloTask
	{
		private readonly YoloBaseTaskModel yolo;

		private bool Initialized => yolo != null;

		public YoloTask(Config config)
		{
			if (string.IsNullOrEmpty(config.OutputPath))
			{
				config.OutputPath = Path.Combine("result", config.TaskType.ToString().ToUpper(), DateTime.Now.ToString("yyyyMMddHHmmss"));
			}
			torchvision.io.DefaultImager = new torchvision.io.SkiaImager();
			yolo = config.TaskType switch
			{
				TaskType.Detection => new Detector(config),
				TaskType.Segmentation => new Segmenter(config),
				TaskType.Obb => new Obber(config),
				TaskType.Pose => new PoseDetector(config),
				TaskType.Classification => new Classifier(config),
				_ => throw new NotImplementedException("Task type not support now.")
			};
		}

		public void LoadModel(string path, bool skipNcNotEqualLayers = false)
		{
			if (!Initialized)
			{
				throw new ArgumentNullException("Yolo is not Initialized.");
			}
			yolo?.LoadModel(path, skipNcNotEqualLayers);
		}

		public void Train()
		{
			if (!Initialized)
			{
				throw new ArgumentNullException("Yolo is not Initialized.");
			}
			yolo?.Train();
		}

		public List<YoloResult> ImagePredict(Tensor orgImage)
		{
			if (!Initialized)
			{
				throw new ArgumentNullException("Yolo is not Initialized.");
			}
			return yolo.ImagePredict(orgImage);
		}

		public List<YoloResult> ImagePredict(SKBitmap image)
		{
			Tensor orgImage = Lib.GetTensorFromImage(image);
			return ImagePredict(orgImage);
		}

		public List<YoloResult> ImagePredict(string imagePath)
		{
			Tensor orgImage = Lib.GetTensorFromImage(imagePath);
			return ImagePredict(orgImage);
		}

		public List<YoloResult> ImagePredict(Mat mat)
		{
			Tensor orgImage = Lib.GetTensorFromImage(mat);
			return ImagePredict(orgImage);
		}

		public List<YoloResult> ImagePredict(Tensor orgImage, float predictThreshold, float iouThreshold)
		{
			if (!Initialized)
			{
				throw new ArgumentNullException("Yolo is not Initialized.");
			}
			return yolo.ImagePredict(orgImage, predictThreshold, iouThreshold);
		}

		public List<YoloResult> ImagePredict(SKBitmap image, float predictThreshold, float iouThreshold)
		{
			Tensor orgImage = Lib.GetTensorFromImage(image);
			return ImagePredict(orgImage, predictThreshold, iouThreshold);
		}

		public List<YoloResult> ImagePredict(string imagePath, float predictThreshold, float iouThreshold)
		{
			Tensor orgImage = Lib.GetTensorFromImage(imagePath);
			return ImagePredict(orgImage, predictThreshold, iouThreshold);
		}

		public List<YoloResult> ImagePredict(Mat mat, float predictThreshold, float iouThreshold)
		{
			Tensor orgImage = Lib.GetTensorFromImage(mat);
			return ImagePredict(orgImage, predictThreshold, iouThreshold);
		}


	}
}
