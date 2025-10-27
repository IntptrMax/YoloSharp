using OpenCvSharp;
using YoloSharp.Models;
using YoloSharp.Types;

namespace YoloSharpDemo
{
	internal class Program
	{
		static void Main(string[] args)
		{
			string rootPath = @"..\..\..\Assets\DataSets\coco8-pose"; // Training data path, it should be the same as coco dataset.
			string trainDataPath = Path.Combine(rootPath, "train.txt"); // If trainDataPath is "", it will use rootPath as training data.
			string valDataPath = Path.Combine(rootPath, "val.txt");// If valDataPath is "", it will use rootPath as validation data.
			string outputPath = "result";    // Trained model output path.
			string preTrainedModelPath = @"..\..\..\Assets\PreTrainedModels\yolov8n-pose.bin"; // Pretrained model path.

			string predictImagePath = @"..\..\..\Assets\TestImage\zidane.jpg";
			int batchSize = 16;
			int numberClass = 1;
			int epochs = 100;
			int imageSize = 640;
			float predictThreshold = 0.3f;
			float iouThreshold = 0.7f;

			// For pose estimation, number of keypoints and each keypoint has (x, y, visibility score)
			int[] keyPointShape = new int[] { 17, 3 };

			YoloType yoloType = YoloType.Yolov8;
			DeviceType deviceType = DeviceType.CUDA;
			ScalarType dtype = ScalarType.Float32;
			YoloSize yoloSize = YoloSize.n;
			ImageProcessType imageProcessType = ImageProcessType.Letterbox;
			TaskType taskType = TaskType.Pose;

			Mat predictImage = Cv2.ImRead(predictImagePath);

			// Create a yolo task.
			YoloTask yoloTask = new YoloTask(taskType, numberClass, yoloType: yoloType, deviceType: deviceType, yoloSize: yoloSize, dtype: dtype, keyPointShape: keyPointShape);

			// Load pre-trained model, if you don't want to load the model, you can skip this step.
			yoloTask.LoadModel(preTrainedModelPath, skipNcNotEqualLayers: true);

			// Train model
			yoloTask.Train(rootPath, trainDataPath, valDataPath, outputPath: outputPath, imageSize: imageSize, batchSize: batchSize, epochs: epochs, imageProcessType: imageProcessType);

			// Predict image, if the model is not trained or loaded, it will use random weight to predict.
			List<YoloResult> predictResult = yoloTask.ImagePredict(predictImage, predictThreshold, iouThreshold);

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
				Cv2.Polylines(predictImage, new Point[][] { pts }, true, Scalar.Red, 2);
				string label = string.Format("{0}:{1:F1}%", result.ClassID, result.Score * 100);

				Size textSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, 0.5, 1, out int baseline);
				Cv2.Rectangle(predictImage, new Rect(new Point(result.X, result.Y - textSize.Height - baseline), new Size(textSize.Width, textSize.Height + baseline)), Scalar.White, Cv2.FILLED);
				Cv2.PutText(predictImage, label, new Point(result.X, result.Y - baseline), HersheyFonts.HersheySimplex, 0.5, Scalar.Black, 1);

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
					Scalar color = new Scalar(R, G, B, 200);
					Mat backColor = new Mat(maskMat.Rows, maskMat.Cols, MatType.CV_8UC3, color);
					Cv2.Add(predictImage, backColor, predictImage, maskMat);
				}

				if (result.KeyPoints is not null)
				{
					foreach (YoloSharp.Types.KeyPoint keyPoint in result.KeyPoints)
					{
						Cv2.Circle(predictImage, (int)keyPoint.X, (int)keyPoint.Y, 2, Scalar.Red);
					}
				}
			}
			predictImage.SaveImage("result.jpg");

			Console.WriteLine();
			Console.WriteLine("Image Predict done");
		}

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

