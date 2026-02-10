using Data;
using OpenCvSharp;
using YoloSharp.Models;
using YoloSharp.Types;

namespace YoloSharpDemo
{
	internal class Program
	{
		static void Main(string[] args)
		{
			string preTrainedModelPath = @"..\..\..\Assets\PreTrainedModels\yolov8n-obb.bin"; // Pretrained model path.
			string predictImagePath = @"..\..\..\Assets\TestImage\trucks.jpg";

			Mat predictImage = Cv2.ImRead(predictImagePath);

			// Create a Yolo config
			Config config = new Config
			{
				DeviceType = DeviceType.CUDA,
				ScalarType = ScalarType.Float16,
				RootPath = @"..\..\..\Assets\DataSets\dotav1",
				TrainDataPath = "train.txt",
				ValDataPath = "val.txt",
				YoloType = YoloType.Yolov8,
				YoloSize = YoloSize.n,
				TaskType = TaskType.Obb,
				ImageProcessType = ImageProcessType.Mosiac,
				ImageSize = 640,
				BatchSize = 16,
				NumberClass = 15,
				PredictThreshold = 0.3f,
				IouThreshold = 0.7f,
				Workers = 4,
				Epochs = 100,
			};

			// Create a yolo task.
			YoloTask yoloTask = new YoloTask(config);

			// Load pre-trained model. If you don't want to use pre-trained model, skip the step.
			yoloTask.LoadModel(preTrainedModelPath, skipNcNotEqualLayers: true);

			// Train model
			yoloTask.Train();

			// Predict image, if the model is not trained or loaded, it will use random weight to predict.
			List<YoloResult> predictResult = yoloTask.ImagePredict(predictImage);

			// Rand for mask color.
			Random rand = new Random(1024);

			// Draw results
			foreach (YoloResult result in predictResult)
			{
				float[] cxcywhr = new float[] { result.CenterX, result.CenterY, result.Width, result.Height, result.Radian };
				float[] points = cxcywhr2xyxyxyxy(cxcywhr);

				Point[] pts = new Point[4]
				{
					new Point(points[0], points[1]),
					new Point(points[2], points[3]),
					new Point(points[4], points[5]),
					new Point(points[6], points[7]),
				};
				Cv2.Polylines(predictImage, new Point[][] { pts }, true, OpenCvSharp.Scalar.Red, 2);
				string label = string.Format("{0}:{1:F1}%", result.ClassID, result.Score * 100);

				Size textSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, 0.5, 1, out int baseline);
				Cv2.Rectangle(predictImage, new Rect(new Point(result.X, result.Y - textSize.Height - baseline), new Size(textSize.Width, textSize.Height + baseline)), OpenCvSharp.Scalar.White, Cv2.FILLED);
				Cv2.PutText(predictImage, label, new Point(result.X, result.Y - baseline), HersheyFonts.HersheySimplex, 0.5, OpenCvSharp.Scalar.Black, 1);
				Console.WriteLine(string.Format("LabelID:{0}, Score:{1:F1}%, CenterX:{2}, CenterY:{3}, Width:{4}, Height:{5}, R:{6:F3} rnd", result.ClassID, result.Score * 100, result.CenterX, result.CenterY, result.Width, result.Height, result.Radian));

				// Draw mask
				if (result.Mask is not null)
				{
					Mat maskMat = Mat.FromArray<byte>(result.Mask);
					maskMat = maskMat * 255;
					maskMat = maskMat.Transpose();

					// Create random color
					int R = rand.Next(0, 255);
					int G = rand.Next(0, 255);
					int B = rand.Next(0, 255);
					OpenCvSharp.Scalar color = new OpenCvSharp.Scalar(R, G, B, 200);
					Mat backColor = new Mat(maskMat.Rows, maskMat.Cols, MatType.CV_8UC3, color);
					Cv2.Add(predictImage, backColor, predictImage, maskMat);
				}

				if (result.KeyPoints is not null)
				{
					foreach (YoloSharp.Types.KeyPoint keyPoint in result.KeyPoints)
					{
						Cv2.Circle(predictImage, (int)keyPoint.X, (int)keyPoint.Y, 3, OpenCvSharp.Scalar.Blue);
					}
				}
			}
			predictImage.SaveImage("result.jpg");

			Console.WriteLine();
			Console.WriteLine("Image Predict done");
		}

		/// <summary>
		/// CenterX, CenterY, Width, Height, r → P0(x0, y0), P1(x1, y1), P2(x2, y2), P3(x3, y3)
		/// </summary>
		/// <param name="x"></param>
		/// <returns></returns>
		private static float[] cxcywhr2xyxyxyxy(float[] x)
		{
			float cx = x[0];
			float cy = x[1];
			float w = x[2];
			float h = x[3];
			float r = x[4];
			float cosR = (float)Math.Cos(r);
			float sinR = (float)Math.Sin(r);
			float wHalf = w / 2;
			float hHalf = h / 2;
			return new float[]
			{
				cx - wHalf * cosR + hHalf * sinR,
				cy - wHalf * sinR - hHalf * cosR,
				cx + wHalf * cosR + hHalf * sinR,
				cy + wHalf * sinR - hHalf * cosR,
				cx + wHalf * cosR - hHalf * sinR,
				cy + wHalf * sinR + hHalf * cosR,
				cx - wHalf * cosR - hHalf * sinR,
				cy - wHalf * sinR + hHalf * cosR,
			};
		}

	}
}

