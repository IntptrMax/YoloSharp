using System.Text;
using TorchSharp;
using YoloSharp.Types;
using static TorchSharp.torch;

namespace Data
{
	/// <summary>
	/// Yolo Config
	/// </summary>
	public class Config
	{
		/// <summary>
		/// Data root path
		/// </summary>
		public string RootPath { get; set; } = @"..\..\..\Assets\DataSets\coco128";

		/// <summary>
		/// Train data path
		/// </summary>
		public string TrainDataPath { get; set; } = "train.txt";

		/// <summary>
		/// Val data path
		/// </summary>
		public string ValDataPath { get; set; } = "val.txt";

		/// <summary>
		/// Weights and logs output path
		/// </summary>
		public string OutputPath { get; set; } = "";

		/// <summary>
		/// Train image size
		/// </summary>
		public int ImageSize { get; set; } = 640;

		/// <summary>
		/// Train batch size
		/// </summary>
		public int BatchSize { get; set; } = 16;

		/// <summary>
		/// Number of class
		/// </summary>
		public int NumberClass { get; set; } = 80;

		/// <summary>
		/// Train epochs
		/// </summary>
		public int Epochs { get; set; } = 100;

		/// <summary>
		/// Predict threshold
		/// </summary>
		public float PredictThreshold { get; set; } = 0.3f;

		/// <summary>
		/// Iou threshold
		/// </summary>
		public float IouThreshold { get; set; } = 0.7f;

		/// <summary>
		/// Learning rate
		/// </summary>
		public float LearningRate { get; set; } = 1e-4f;

		/// <summary>
		/// Workers for training
		/// </summary>
		public int Workers { get; set; } = Math.Min(Environment.ProcessorCount / 2, 4);

		/// <summary>
		/// Yolo type, can be Yolov5, Yolov8, Yolov11, Yolov12
		/// </summary>
		public YoloType YoloType { get; set; } = YoloType.Yolov8;

		/// <summary>
		/// Yolo size, can be n, s, m, l, x
		/// </summary>
		public YoloSize YoloSize { get; set; } = YoloSize.n;

		/// <summary>
		/// Yolo task, can be Detection, Segment, Obb, Pose, Classification
		/// </summary>
		public TaskType TaskType { get; set; } = TaskType.Detection;

		/// <summary>
		/// Device for Yolo running, can be CPU or Cuda
		/// </summary>
		public YoloSharp.Types.DeviceType DeviceType { get; set; } = YoloSharp.Types.DeviceType.CUDA;

		/// <summary>
		/// Scalar Type for Yolo running, can be Float32, Float16, BFloat16
		/// </summary>
		public YoloSharp.Types.ScalarType ScalarType { get; set; } = YoloSharp.Types.ScalarType.Float16;

		/// <summary>
		/// Image process type, can be Mosiac or Letterbox
		/// </summary>
		public ImageProcessType ImageProcessType { get; set; } = ImageProcessType.Mosiac;

		/// <summary>
		/// Early stop patience
		/// </summary>
		public int Patience { get; set; } = 30;

		/// <summary>
		/// Early stop delta
		/// </summary>
		public float Delta { get; set; } = 1e-5f;

		/// <summary>
		/// Key point shape, it will be use only in YoloPose
		/// </summary>
		public int[] KeyPointShape { get; set; } = new int[] { 17, 3 };

		/// <summary>
		/// Brightness for Image process ColorJitter
		/// </summary>
		public float Brightness { get; set; } = 0.1f;

		/// <summary>
		/// Contrast for Image process ColorJitter
		/// </summary>
		public float Contrast { get; set; } = 0.1f;

		/// <summary>
		/// Saturation for Image process ColorJitter
		/// </summary>
		public float Saturation { get; set; } = 0.1f;

		/// <summary>
		/// Hue for Image process ColorJitter
		/// </summary>
		public float Hue { get; set; } = 0.02f;

		public Config(string? rootPath = null, string? trainDataPath = null, string? valDataPath = null, string? outputPath = null,
			int? imageSize = null, int? batchSize = null, int? numberClass = null, int? epochs = null, float? predictThreshold = null,
			float? iouThreshold = null, float? learningRate = null, int? workers = null, YoloType? yoloType = null,
			YoloSize? yoloSize = null, TaskType? taskType = null, YoloSharp.Types.DeviceType? deviceType = null,
			YoloSharp.Types.ScalarType? dtype = null, ImageProcessType? imageProcessType = null,
			int? patience = null, float? delta = null, int[]? keyPointShape = null, float? brightness = null, float? contrast = null,
			float? saturation = null, float? hue = null)
		{
			RootPath = rootPath ?? RootPath;
			TrainDataPath = trainDataPath ?? TrainDataPath;
			ValDataPath = valDataPath ?? ValDataPath;
			TaskType = taskType ?? TaskType;
			OutputPath = string.IsNullOrEmpty(outputPath) ? "" : outputPath;
			ImageSize = imageSize ?? ImageSize;
			BatchSize = batchSize ?? BatchSize;
			NumberClass = numberClass ?? NumberClass;
			Epochs = epochs ?? Epochs;
			PredictThreshold = predictThreshold ?? PredictThreshold;
			IouThreshold = iouThreshold ?? IouThreshold;
			LearningRate = learningRate ?? LearningRate;
			Workers = workers ?? Math.Min(Environment.ProcessorCount / 2, 4);
			YoloType = yoloType ?? YoloType;
			YoloSize = yoloSize ?? YoloSize;
			DeviceType = deviceType ?? DeviceType;
			ScalarType = dtype ?? ScalarType;
			ImageProcessType = imageProcessType ?? ImageProcessType;
			Patience = patience ?? Patience;
			Delta = delta ?? Delta;
			KeyPointShape = keyPointShape ?? new int[] { 17, 3 };
			Brightness = brightness ?? Brightness;
			Contrast = contrast ?? Contrast;
			Saturation = saturation ?? Saturation;
			Hue = hue ?? Hue;
		}

		public torch.Device Device => new Device((TorchSharp.DeviceType)DeviceType);
		public torch.ScalarType Dtype => (torch.ScalarType)ScalarType;

		public override string ToString()
		{
			StringBuilder stringBuilder = new StringBuilder();
			stringBuilder.AppendLine($"Yolo task type: {TaskType}");
			stringBuilder.AppendLine($"Yolo type: {YoloType}");
			stringBuilder.AppendLine($"Yolo size: {YoloSize}");
			stringBuilder.AppendLine($"Image Process Type: {ImageProcessType}");
			stringBuilder.AppendLine($"Precision type: {Dtype}");
			stringBuilder.AppendLine($"Device type: {Device}");
			stringBuilder.AppendLine($"Number Classes: {NumberClass}");
			stringBuilder.AppendLine($"Image Size: {ImageSize}");
			stringBuilder.AppendLine($"Epochs: {Epochs}");
			stringBuilder.AppendLine($"Learning Rate: {LearningRate}");
			stringBuilder.AppendLine($"Batch Size: {BatchSize}");
			stringBuilder.AppendLine($"Num Workers: {Workers}");
			stringBuilder.AppendLine($"Key Points Shape (Only use in pose): [{KeyPointShape[0]}, {KeyPointShape[1]}]");
			stringBuilder.AppendLine($"Root Path: \"{Path.GetFullPath(RootPath)}\"");
			stringBuilder.AppendLine($"Train Data Path: {TrainDataPath}");
			stringBuilder.AppendLine($"Val Data Path: {ValDataPath}");
			stringBuilder.AppendLine($"Output Path: \"{Path.GetFullPath(OutputPath)}\"");
			stringBuilder.AppendLine($"Early Stop Patience: {Patience}");
			stringBuilder.AppendLine($"Early Stop Delta: {Delta}");
			stringBuilder.AppendLine($"Predict Threshold: {PredictThreshold}");
			stringBuilder.AppendLine($"Iou Threshold: {IouThreshold}");
			stringBuilder.AppendLine($"Brightness Augmentation: {Brightness}");
			stringBuilder.AppendLine($"Contrast Augmentation: {Contrast}");
			stringBuilder.AppendLine($"Saturation Augmentation: {Saturation}");
			stringBuilder.AppendLine($"Hue Augmentation: {Hue}");
			return stringBuilder.ToString();
		}

	}
}
