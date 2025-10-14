using OpenCvSharp;
using SkiaSharp;
using YoloSharp.Types;
using YoloSharp.Utils;
using static TorchSharp.torch;

namespace YoloSharp.Models
{
	public class YoloTask
	{
		private readonly YoloBaseTaskModel yolo;

		private bool Initialized => yolo != null;

		public YoloTask(TaskType taskType, int numberClasses, YoloType yoloType, YoloSize yoloSize, DeviceType deviceType = DeviceType.CUDA, Types.ScalarType dtype = Types.ScalarType.Float32)
		{
			yolo = taskType switch
			{
				TaskType.Detection => new Detector(numberClasses, yoloType, yoloSize, deviceType, dtype),
				TaskType.Segmentation => new Segmenter(numberClasses, yoloType, yoloSize, deviceType, dtype),
				TaskType.Obb => new Obber(numberClasses, yoloType, yoloSize, deviceType, dtype),
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

		public void Train(string rootPath, string trainDataPath = "", string valDataPath = "", string outputPath = "output", int imageSize = 640, int epochs = 100, float lr = 0.0001f, int batchSize = 8, int numWorkers = 0, ImageProcessType imageProcessType = ImageProcessType.Letterbox)
		{
			if (!Initialized)
			{
				throw new ArgumentNullException("Yolo is not Initialized.");
			}
			yolo?.Train(rootPath, trainDataPath, valDataPath, outputPath, imageSize, epochs, lr, batchSize, numWorkers, imageProcessType);
		}

		public List<YoloResult> ImagePredict(Tensor orgImage, float PredictThreshold = 0.25f, float IouThreshold = 0.5f)
		{
			if (!Initialized)
			{
				throw new ArgumentNullException("Yolo is not Initialized.");
			}
			return yolo.ImagePredict(orgImage, PredictThreshold, IouThreshold);
		}


		public List<YoloResult> ImagePredict(SKBitmap image, float PredictThreshold = 0.25f, float IouThreshold = 0.5f)
		{
			Tensor orgImage = Lib.GetTensorFromImage(image);
			return ImagePredict(orgImage, PredictThreshold, IouThreshold);
		}

		public List<YoloResult> ImagePredict(string imagePath, float PredictThreshold = 0.25f, float IouThreshold = 0.5f)
		{
			Tensor orgImage = Lib.GetTensorFromImage(imagePath);
			return ImagePredict(orgImage, PredictThreshold, IouThreshold);
		}

		public List<YoloResult> ImagePredict(Mat mat, float PredictThreshold = 0.25f, float IouThreshold = 0.5f)
		{
			Tensor orgImage = Lib.GetTensorFromImage(mat);
			return ImagePredict(orgImage, PredictThreshold, IouThreshold);
		}


	}
}
