using System.Runtime;
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
        /// Use Cos LR
        /// </summary>
        public bool UseCosLR { get; set; } = false;

        /// <summary>
        /// Lrf
        /// </summary>
        public float Lrf { get; set; } = 0.01f;

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
        public int Patience { get; set; } = 50;

        /// <summary>
        /// Gets or sets the number of key points used in the operation.
        /// </summary>
        public int KeyPoint_Num { get; set; } = 17;

        /// <summary>
        /// Gets or sets the dimensionality of the key point representation.
        /// </summary>
        public int KeyPoint_Dim { get; set; } = 3;

        /// <summary>
        /// Brightness for Image process ColorJitter
        /// </summary>
        public float HSV_V { get; set; } = 0.4f;

        /// <summary>
        /// Saturation for Image process ColorJitter
        /// </summary>
        public float HSV_S { get; set; } = 0.7f;

        /// <summary>
        /// Hue for Image process ColorJitter
        /// </summary>
        public float HSV_H { get; set; } = 0.015f;

        /// <summary>
        /// Gets or sets the mask ratio used for image processing or augmentation.
        /// </summary>
        public int MaskRatio { get; set; } = 4;

        /// <summary>
        /// Mosaic augmentation ratio, it will be use only in Mosiac image process type, and the value should be between 0 and 1, the default value is 1.0f, which means all the images will be used for mosaic augmentation.
        /// </summary>
        public float Mosaic { get; set; } = 1.0f;

        /// <summary>
        /// The number of images to be used in mosaic augmentation, it will be use only in Mosiac image process type, and the value should be 4 or 9.
        /// </summary>
        public int MosaicCount { get; set; } = 4;

        /// <summary>
        /// Gets or sets the angle in degrees.
        /// </summary>
        public float Degrees { get; set; } = 0.0f;

        /// <summary>
        /// Gets or sets the translation value applied to the object.
        /// </summary>
        public float Translate { get; set; } = 0.1f;

        /// <summary>
        /// Gets or sets the scale factor applied to the object.
        /// </summary>
        public float Scale { get; set; } = 0.5f;

        /// <summary>
        /// Gets or sets the shear factor applied to the transformation.
        /// </summary>
        public float Shear { get; set; } = 0.0f;

        /// <summary>
        /// Gets or sets the perspective value used for rendering or transformation calculations.
        /// </summary>
        public float Perspective { get; set; } = 0.0f;

        /// <summary>
        /// Gets or sets the probability of flipping an image horizontally (left to right) during processing.
        /// </summary>
        /// <remarks>Set this value between 0.0 and 1.0 to control the likelihood of a horizontal flip. A
        /// value of 0.0 disables flipping, while 1.0 always applies the flip.</remarks>
        public float FlipLR { get; set; } = 0.5f;

        /// <summary>
        /// Gets or sets the vertical flip factor for the object.
        /// </summary>
        /// <remarks>A value of 0.0f indicates no vertical flip. Positive values may represent the degree
        /// or presence of a vertical flip, depending on the object's implementation. Refer to the object's
        /// documentation for details on how this value is interpreted.</remarks>
        public float FlipUD { get; set; } = 0.0f;

        /// <summary>
        /// Gets or sets the maximum allowed classification ratio.
        /// </summary>
        public float ClassifyRatioMax { get; set; } = 4f / 3;

        /// <summary>
        /// Gets or sets the minimum classification ratio required for a result to be considered valid.
        /// </summary>
        public float ClassifyRatioMin { get; set; } = 0.75f;

        /// <summary>
        /// Gets or sets the maximum scale value used for classification operations.
        /// </summary>
        public float ClassifyScaleMax { get; set; } = 1f;

        /// <summary>
        /// Gets or sets the minimum scale value used for classification operations.
        /// </summary>
        public float ClassifyScaleMin { get; set; } = 0.08f;

        /// <summary>
        /// Gets or sets the ratio used for erasing operations.
        /// </summary>
        /// <remarks>The erasing ratio determines the proportion of the area or content that will be
        /// erased during an operation. Adjust this value to control the strength or extent of erasing
        /// effects.</remarks>
        public float Erasing { get; set; } = 0.4f;

        /// <summary>
        /// Gets or sets the type of auto-augmentation strategy to be applied during data processing or training.
        /// </summary>
        public AutoAugmentType Auto_Augment { get; set; } = AutoAugmentType.AutoAugment;

        /// <summary>
        /// Warm Up Epoches
        /// </summary>
        public int WarmUpEpoches { get; set; } = 3;

        public double WarmUpBiasLr { get; set; } = 0.1;

        public int CloseMosaic { get; set; } = 0;

        public Config(string? rootPath = null, string? trainDataPath = null, string? valDataPath = null, string? outputPath = null,
            int? imageSize = null, int? batchSize = null, int? numberClass = null, int? epochs = null, float? predictThreshold = null,
            float? iouThreshold = null, float? learningRate = null, bool? useCosLR = null, float? lrf = null, int? workers = null, YoloType? yoloType = null,
            YoloSize? yoloSize = null, TaskType? taskType = null, YoloSharp.Types.DeviceType? deviceType = null,
            YoloSharp.Types.ScalarType? dtype = null, ImageProcessType? imageProcessType = null,
            int? patience = null, float? delta = null, int? keyPoint_Num = null, int? keyPoint_dim = null, float? hsv_v = null,
            float? hsv_s = null, float? hsv_h = null, int? maskRation = null, float? mosaic = null, int? mosaicCount = null,
            float? degrees = null, float? translate = null, float? scale = null, float? shear = null, float? perspective = null,
            float? flipLR = null, float? flipUD = null, float? classifyRatioMax = null, float? classifyRatioMin = null,
            float? classifyScaleMax = null, float? classifyScaleMin = null, float? erasing = null, AutoAugmentType? autoAugment = null,
            int? warmUpEpoches = null, double? warmUpBiasLr = null, int? closeMosaic = null)
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
            KeyPoint_Num = keyPoint_Num ?? KeyPoint_Num;
            KeyPoint_Dim = keyPoint_dim ?? KeyPoint_Dim;
            HSV_V = hsv_v ?? HSV_V;
            HSV_S = hsv_s ?? HSV_S;
            HSV_H = hsv_h ?? HSV_H;
            MaskRatio = maskRation ?? MaskRatio;
            Mosaic = mosaic ?? Mosaic;
            MosaicCount = mosaicCount ?? MosaicCount;
            Degrees = degrees ?? Degrees;
            Translate = translate ?? Translate;
            Scale = scale ?? Scale;
            Shear = shear ?? Shear;
            Perspective = perspective ?? Perspective;
            FlipLR = flipLR ?? FlipLR;
            FlipUD = flipUD ?? FlipUD;
            ClassifyRatioMax = classifyRatioMax ?? ClassifyRatioMax;
            ClassifyRatioMin = classifyRatioMin ?? ClassifyRatioMin;
            ClassifyScaleMax = classifyScaleMax ?? ClassifyScaleMax;
            ClassifyScaleMin = classifyScaleMin ?? ClassifyScaleMin;
            Erasing = erasing ?? Erasing;
            Auto_Augment = autoAugment ?? Auto_Augment;
            UseCosLR = useCosLR ?? UseCosLR;
            Lrf = lrf ?? Lrf;
            WarmUpEpoches = warmUpEpoches ?? WarmUpEpoches;
            WarmUpBiasLr = warmUpBiasLr ?? WarmUpBiasLr;
            CloseMosaic = closeMosaic ?? CloseMosaic;
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
            stringBuilder.AppendLine($"Use Cos LR: {UseCosLR}");
            stringBuilder.AppendLine($"Lrf: {Lrf}");
            stringBuilder.AppendLine($"Warm Up Epoches: {WarmUpEpoches}");
            stringBuilder.AppendLine($"Warm Up Bias Learning Rate:{WarmUpBiasLr}");
            stringBuilder.AppendLine($"Batch Size: {BatchSize}");
            stringBuilder.AppendLine($"Num Workers: {Workers}");
            stringBuilder.AppendLine($"Key Points Shape (Only use in pose): [{KeyPoint_Num}, {KeyPoint_Dim}]");
            stringBuilder.AppendLine($"Root Path: \"{Path.GetFullPath(RootPath)}\"");
            stringBuilder.AppendLine($"Train Data Path: {TrainDataPath}");
            stringBuilder.AppendLine($"Val Data Path: {ValDataPath}");
            stringBuilder.AppendLine($"Output Path: \"{Path.GetFullPath(OutputPath)}\"");
            stringBuilder.AppendLine($"Early Stop Patience: {Patience}");
            stringBuilder.AppendLine($"Predict Threshold: {PredictThreshold}");
            stringBuilder.AppendLine($"Iou Threshold: {IouThreshold}");
            stringBuilder.AppendLine($"HSV_V Augmentation: {HSV_V}");
            stringBuilder.AppendLine($"HSV_S Augmentation: {HSV_S}");
            stringBuilder.AppendLine($"HSV_H Augmentation: {HSV_H}");
            stringBuilder.AppendLine($"Mask Ratio: {MaskRatio}");
            stringBuilder.AppendLine($"Mosaic Augmentation Ratio: {Mosaic}");
            stringBuilder.AppendLine($"Mosaic Count: {MosaicCount}");
            stringBuilder.AppendLine($"Close Mosaic: {CloseMosaic}");
            stringBuilder.AppendLine($"Degrees: {Degrees}");
            stringBuilder.AppendLine($"Translate: {Translate}");
            stringBuilder.AppendLine($"Scale: {Scale}");
            stringBuilder.AppendLine($"Shear: {Shear}");
            stringBuilder.AppendLine($"Perspective: {Perspective}");
            stringBuilder.AppendLine($"FlipLR: {FlipLR}");
            stringBuilder.AppendLine($"FlipUD: {FlipUD}");
            stringBuilder.AppendLine($"Classify Ratio Max: {ClassifyRatioMax}");
            stringBuilder.AppendLine($"Classify Ratio Min: {ClassifyRatioMin}");
            stringBuilder.AppendLine($"Classify Scale Max: {ClassifyScaleMax}");
            stringBuilder.AppendLine($"Classify Scale Min: {ClassifyScaleMin}");
            stringBuilder.AppendLine($"Classify Erasing: {Erasing}");
            stringBuilder.AppendLine($"Auto Augment Type: {Auto_Augment}");
            return stringBuilder.ToString();
        }

    }
}
