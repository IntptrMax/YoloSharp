using SkiaSharp;
using YoloSharp;

namespace YoloSharpDemo
{
	internal class Program
	{
		static void Main(string[] args)
		{
			string trainDataPath = @"..\..\..\Assets\DataSets\coco128"; // Training data path, it should be the same as coco dataset.
			string valDataPath = string.Empty; // If valDataPath is "", it will use trainDataPath as validation data.
			string outputPath = "result";    // Trained model output path.
			string preTrainedModelPath = @"..\..\..\Assets\PreTrainedModels\yolov8n.bin"; // Pretrained model path.
			string predictImagePath = @"..\..\..\Assets\TestImage\bus.jpg";
			//predictImagePath = @"..\..\..\Assets\TestImage\trucks.jpg";
			int batchSize = 16;
			int sortCount = 80;
			int epochs = 10;
			float predictThreshold = 0.25f;
			float iouThreshold = 0.7f;

			YoloType yoloType = YoloType.Yolov8;
			DeviceType deviceType = DeviceType.CUDA;
			ScalarType dtype = ScalarType.Float32;
			YoloSize yoloSize = YoloSize.n;
			SKBitmap predictImage = SKBitmap.Decode(predictImagePath);

			////Create obber
			//Obber obber = new Obber(15);
			//obber.LoadModel(@"..\..\..\Assets\PreTrainedModels\yolov8n-obb.bin");

			//var predictResult = obber.ImagePredict(predictImage, IouThreshold: iouThreshold);
			//var resultImage = predictImage.Copy();

			// Create predictor
			Predictor predictor = new Predictor(sortCount, yoloType: yoloType, deviceType: deviceType, yoloSize: yoloSize, dtype: dtype);
			predictor.LoadModel(preTrainedModelPath, skipNcNotEqualLayers: true);

			// Train model
			predictor.Train(trainDataPath, valDataPath, outputPath: outputPath, batchSize: batchSize, epochs: epochs, useMosaic: false);
			predictor.LoadModel(Path.Combine(outputPath, "best.bin"));

			// ImagePredict image
			List<YoloResult> predictResult = predictor.ImagePredict(predictImage, predictThreshold, iouThreshold);
			var resultImage = predictImage.Copy();

			////Create segmenter
			//Segmenter segmenter = new Segmenter(sortCount, yoloType: yoloType, deviceType: deviceType, yoloSize: yoloSize, dtype: dtype);
			//segmenter.LoadModel(preTrainedModelPath, skipNcNotEqualLayers: true);

			//// Train model
			//segmenter.Train(trainDataPath, valDataPath, outputPath: outputPath, batchSize: batchSize, epochs: epochs, useMosaic: false);
			//segmenter.LoadModel(Path.Combine(outputPath, "best.bin"));

			//// ImagePredict image
			//var (predictResult, resultImage) = segmenter.ImagePredict(predictImage, predictThreshold, iouThreshold);

			using (var canvas = new SKCanvas(resultImage))
			using (var paint = new SKPaint())
			{
				paint.Color = SKColors.Red;
				paint.StrokeWidth = 2;
				paint.Style = SKPaintStyle.Stroke;
				paint.IsAntialias = true;

				SKPaint textPaint = new SKPaint
				{
					Color = SKColors.Red,
					TextSize = 20,
					Typeface = SKTypeface.FromFamilyName("Consolas"),
					IsAntialias = true,
				};

				foreach (var result in predictResult)
				{
					canvas.Translate(result.CenterX, result.CenterY);
					canvas.RotateRadians(result.Radian);
					canvas.Translate(-result.CenterX, -result.CenterY);
					SKRect rect = new SKRect(result.X, result.Y, result.X + result.Width, result.Y + result.Height);
					canvas.DrawRect(rect, paint);

					string label = string.Format("{0}:{1:F1}%", result.ClassID, result.Score * 100);
					canvas.DrawText(label, result.X, result.Y + 20, textPaint);
					string consoleString = string.Format("ClassID: {0}, Score: {1:F1}%, X: {2}, Y: {3}, W: {4}, H: {5}, R:{6:F2}rnd", result.ClassID, result.Score * 100, result.X, result.Y, result.Width, result.Height, result.Radian);
					Console.WriteLine(consoleString);
					canvas.Translate(result.CenterX, result.CenterY);
					canvas.RotateRadians(-result.Radian);
					canvas.Translate(-result.CenterX, -result.CenterY);
				}
			}
			resultImage.Encode(SKEncodedImageFormat.Jpeg, 100).SaveTo(File.OpenWrite("result.jpg"));

			Console.WriteLine();
			Console.WriteLine("Image Predict done");
		}

	}
}

