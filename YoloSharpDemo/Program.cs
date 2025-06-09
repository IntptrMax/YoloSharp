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
			int batchSize = 16;
			int sortCount = 80;
			int epochs = 100;
			float predictThreshold = 0.25f;
			float iouThreshold = 0.7f;

			YoloType yoloType = YoloType.Yolov8;
			DeviceType deviceType = DeviceType.CUDA;
			ScalarType dtype = ScalarType.Float16;
			YoloSize yoloSize = YoloSize.n;

			SKBitmap predictImage = SKBitmap.Decode(predictImagePath);

			// Create predictor
			Predictor predictor = new Predictor(sortCount, yoloType: yoloType, deviceType: deviceType, yoloSize: yoloSize, dtype: dtype);
			predictor.LoadModel(preTrainedModelPath, skipNcNotEqualLayers: true);

			// Train model
			predictor.Train(trainDataPath, valDataPath, outputPath: outputPath, batchSize: batchSize, epochs: epochs, useMosaic: true);
			predictor.LoadModel(Path.Combine(outputPath, "best.bin"));

			// ImagePredict image
			List<Predictor.PredictResult> predictResult = predictor.ImagePredict(predictImage, predictThreshold, iouThreshold);
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
				paint.Color = SKColors.Black;
				paint.StrokeWidth = 2;
				paint.Style = SKPaintStyle.Stroke;
				paint.IsAntialias = true;

				SKPaint textPaint = new SKPaint
				{
					Color = SKColors.Red,
					TextSize = 20,
					Typeface = SKTypeface.FromFamilyName("Consolas"),
					IsAntialias = true
				};

				foreach (var result in predictResult)
				{
					var rect = new SKRect(result.X, result.Y, result.X + result.W, result.Y + result.H);
					canvas.DrawRect(rect, paint);

					string label = string.Format("Sort:{0}, Score:{1:F1}%", result.ClassID, result.Score * 100);
					canvas.DrawText(label, result.X, result.Y - 12, textPaint);

					Console.WriteLine(label);
				}
			}
			resultImage.Encode(SKEncodedImageFormat.Jpeg, 70).SaveTo(File.OpenWrite("result.jpg"));

			Console.WriteLine();
			Console.WriteLine("ImagePredict done");
		}

	}
}

