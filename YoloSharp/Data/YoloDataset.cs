using OpenCvSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace Data
{
	internal class YoloDataset : utils.data.Dataset
	{
		private string rootPath = string.Empty;
		public int ImageSize => imageSize;
		private int imageSize = 640;
		private List<string> imageFiles = new List<string>();
		private ImageProcessType imageProcessType = ImageProcessType.Letterbox;
		public ImageProcessType ImageProcessType => imageProcessType;
		private TaskType taskType = TaskType.Detection;

		public YoloDataset(string rootPath, string dataPath = "", int imageSize = 640, TaskType taskType = TaskType.Detection, ImageProcessType imageProcessType = ImageProcessType.Letterbox)
		{
			torchvision.io.DefaultImager = new torchvision.io.SkiaImager();

			this.rootPath = rootPath;

			if (string.IsNullOrEmpty(dataPath))
			{
				string imagesFolder = Path.Combine(rootPath, "images");
				if (!Directory.Exists(imagesFolder))
				{
					throw new DirectoryNotFoundException($"The folder {imagesFolder} does not exist.");
				}

				string[] imagesFileNames = Directory.GetFiles(imagesFolder, "*.*", SearchOption.AllDirectories).Where(file =>
				{
					string extension = Path.GetExtension(file).ToLower();
					return extension == ".jpg" || extension == ".png" || extension == ".bmp";
				}).ToArray();
				imageFiles.AddRange(imagesFileNames);
			}
			else
			{
				string path = Path.Combine(rootPath, dataPath);
				if (!File.Exists(path))
				{
					throw new FileNotFoundException($"The file {dataPath} does not exist.");
				}

				string[] imagesFileNames = File.ReadAllLines(dataPath).Where(line =>
				{
					string trimmedLine = line.Trim();
					if (string.IsNullOrEmpty(trimmedLine))
					{
						return false;
					}
					string extension = Path.GetExtension(trimmedLine).ToLower();
					return extension == ".jpg" || extension == ".png" || extension == ".bmp";
				}).Select(line => Path.IsPathRooted(line) ? line : Path.Combine(rootPath, line.Trim())).ToArray();

				imageFiles.AddRange(imagesFileNames);

			}

			this.imageSize = imageSize;
			this.imageProcessType = imageProcessType;
			this.taskType = taskType;
		}

		private string GetLabelFileNameFromImageName(string imageFileName)
		{
			string imagesFolder = Path.Combine(rootPath, "images");
			string labelsFolder = Path.Combine(rootPath, "labels");
			string labelFileName = Path.ChangeExtension(imageFileName, ".txt").Replace(imagesFolder, labelsFolder);
			if (File.Exists(labelFileName))
			{
				return labelFileName;
			}
			else
			{
				return string.Empty;
			}
		}

		public override long Count => imageFiles.Count;

		private string GetFileNameByIndex(long index)
		{
			return imageFiles[(int)index];
		}

		public override Dictionary<string, Tensor> GetTensor(long index)
		{
			Dictionary<string, Tensor> outputs = new Dictionary<string, Tensor>();
			outputs.Add("index", tensor(index));
			return outputs;
		}

		public static Mat GetMaskFromOutlinePoints(Point2f[] points, int width, int height)
		{
			Mat mask = Mat.Zeros(height, width, MatType.CV_8UC1);
			Point[][] pts = new Point[1][];
			pts[0] = points.Select(p => new Point((int)p.X, (int)p.Y)).ToArray();
			Cv2.FillPoly(mask, pts, OpenCvSharp.Scalar.White);
			return mask;
		}

		public ImageData GetImageAndLabelData(long index)
		{
			return imageProcessType switch
			{
				ImageProcessType.Letterbox => GetImageAndLabelDataWithLetterBox(index),
				ImageProcessType.Mosiac => GetImageAndLabelDataWithMosic4(index),
				_ => throw new Exception($"The image process type {imageProcessType} is not supported."),
			};
		}

		public ImageData GetOrgImageAndLabelData(long index)
		{
			string imageFileName = imageFiles[(int)index];
			string labelFileName = GetLabelFileNameFromImageName(imageFileName);
			using (Mat orgImage = Cv2.ImRead(imageFileName))
			{
				int orgWidth = orgImage.Width;
				int orgHeight = orgImage.Height;

				List<LabelData> labels = new List<LabelData>();
				if (!string.IsNullOrEmpty(labelFileName))
				{
					string[] strings = File.ReadAllLines(labelFileName);
					foreach (string line in strings)
					{
						string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
						if (parts is null)
						{
							throw new Exception($"The label file {labelFileName} format is incorrect.");
						}
						switch (taskType)
						{
							case TaskType.Detection:
								{
									if (parts.Length != 5)
									{
										throw new Exception($"The label file {labelFileName} format is incorrect.");
									}
									labels.Add(new LabelData()
									{
										LabelID = int.Parse(parts[0]),
										CenterX = float.Parse(parts[1]) * orgWidth,
										CenterY = float.Parse(parts[2]) * orgHeight,
										Width = float.Parse(parts[3]) * orgWidth,
										Height = float.Parse(parts[4]) * orgHeight,
										Radian = 0,
									});
									break;
								}
							case TaskType.Obb:
								{
									if (parts.Length != 9)
									{
										throw new Exception($"The label file {labelFileName} format is incorrect.");
									}
									int label = int.Parse(parts[0]);
									float x1 = float.Parse(parts[1]) * orgWidth;
									float y1 = float.Parse(parts[2]) * orgHeight;
									float x2 = float.Parse(parts[3]) * orgWidth;
									float y2 = float.Parse(parts[4]) * orgHeight;
									float x3 = float.Parse(parts[5]) * orgWidth;
									float y3 = float.Parse(parts[6]) * orgHeight;
									float x4 = float.Parse(parts[7]) * orgWidth;
									float y4 = float.Parse(parts[8]) * orgHeight;
									float[] re = Utils.Ops.xyxyxyxy2xywhr(new float[] { x1, y1, x2, y2, x3, y3, x4, y4 });
									labels.Add(new LabelData()
									{
										LabelID = label,
										CenterX = re[0],
										CenterY = re[1],
										Width = re[2],
										Height = re[3],
										Radian = re[4],
									});
									break;
								}
							case TaskType.Segmentation:
								{
									if (parts.Length < 5)
									{
										throw new Exception($"The label file {labelFileName} format is incorrect.");
									}

									Point2f[] maskOutlinePoints = new Point2f[(parts.Length - 1) / 2];
									for (int i = 0; i < maskOutlinePoints.Length; i++)
									{
										maskOutlinePoints[i] = new Point2f(float.Parse(parts[1 + i * 2]) * orgWidth, float.Parse(parts[2 + i * 2]) * orgHeight);
									}

									Rect rect = Cv2.BoundingRect(maskOutlinePoints);

									labels.Add(new LabelData()
									{
										LabelID = int.Parse(parts[0]),
										CenterX = (rect.Left + rect.Right) / 2.0f,
										CenterY = (rect.Top + rect.Bottom) / 2.0f,
										Width = rect.Width,
										Height = rect.Height,
										Radian = 0,
										MaskOutLine = maskOutlinePoints
									});
									break;
								}
							default:
								throw new Exception($"The task type {taskType} is not supported.");
						}
					}


				}
				ImageData imageData = new ImageData
				{
					ImagePath = imageFileName,
					OrgWidth = orgWidth,
					OrgHeight = orgHeight,
					OrgLabels = labels
				};
				return imageData;

			}
		}

		public ImageData GetImageAndLabelDataWithLetterBox(long index)
		{
			ImageData imageData = GetOrgImageAndLabelData(index);
			LetterBox(imageData, imageSize);
			return imageData;
		}

		private void LetterBox(ImageData imageData, int size)
		{
			float r = Math.Min((float)size / imageData.OrgWidth, (float)size / imageData.OrgHeight);
			int newUnpadW = (int)Math.Round(imageData.OrgWidth * r);
			int newUnpadH = (int)Math.Round(imageData.OrgHeight * r);
			int dw = size - newUnpadW;
			int dh = size - newUnpadH;
			dw /= 2;
			dh /= 2;
			Mat resized = new Mat();
			Cv2.Resize(imageData.OrgImage, resized, new OpenCvSharp.Size(newUnpadW, newUnpadH));
			Cv2.CopyMakeBorder(resized, resized, dh, size - newUnpadH - dh, dw, size - newUnpadW - dw, BorderTypes.Constant, new OpenCvSharp.Scalar(114, 114, 114));
			imageData.ResizedImage = resized;

			// Adjust labels
			if (imageData.OrgLabels is not null)
			{
				imageData.ResizedLabels = new List<LabelData>();
				foreach (var label in imageData.OrgLabels)
				{
					LabelData resizedLabel = new LabelData();
					resizedLabel.CenterX = label.CenterX * r + dw;
					resizedLabel.CenterY = label.CenterY * r + dh;
					resizedLabel.Width = label.Width * r;
					resizedLabel.Height = label.Height * r;
					resizedLabel.Radian = label.Radian;
					resizedLabel.LabelID = label.LabelID;
					if (label.MaskOutLine is not null)
					{
						resizedLabel.MaskOutLine = new Point2f[label.MaskOutLine.Length];
						for (int i = 0; i < label.MaskOutLine.Length; i++)
						{
							resizedLabel.MaskOutLine[i].X = label.MaskOutLine[i].X * r + dw;
							resizedLabel.MaskOutLine[i].Y = label.MaskOutLine[i].Y * r + dh;
						}
					}
					imageData.ResizedLabels.Add(resizedLabel);
				}
			}
		}

		public ImageData GetImageAndLabelDataWithMosic4(long index)
		{
			int imgCount = 4;
			long[] indices = Sample(index, 0, (int)Count, imgCount);

			ImageData[] imageDatas = new ImageData[imgCount];

			Random random = new Random();
			int w = random.Next(1, imageSize - 1);
			int h = random.Next(1, imageSize - 1);
			Mat mosaicMat = new Mat(imageSize, imageSize, MatType.CV_8UC3, new OpenCvSharp.Scalar(114, 114, 114));

			for (int i = 0; i < imgCount; i++)
			{
				imageDatas[i] = GetOrgImageAndLabelData(indices[i]);
				Mat tempMat = imageDatas[i].OrgImage;

				int tempX = (i == 0 || i == 2) ? 0 : w;
				int tempY = (i == 0 || i == 1) ? 0 : h;
				int tempW = (i == 0 || i == 2) ? w : imageSize - w;
				int tempH = (i == 0 || i == 1) ? h : imageSize - h;
				int randomX = random.Next(0, Math.Max(0, tempMat.Width - tempW));
				int randomY = random.Next(0, Math.Max(0, tempMat.Height - tempH));
				Rect roi = new Rect(randomX, randomY, Math.Min(tempW, tempMat.Width - randomX), Math.Min(tempH, tempMat.Height - randomY));
				Mat cropped = new Mat(tempMat, roi);
				cropped.CopyTo(mosaicMat[new Rect(tempX, tempY, roi.Width, roi.Height)]);

				imageDatas[0].ResizedImage = mosaicMat;
				for (int j = 0; j < imageDatas[i].OrgLabels.Count; j++)
				{
					LabelData label = imageDatas[i].OrgLabels[j];
					float x1 = label.CenterX - label.Width / 2.0f;
					float y1 = label.CenterY - label.Height / 2.0f;
					float x2 = label.CenterX + label.Width / 2.0f;
					float y2 = label.CenterY + label.Height / 2.0f;
					
					// Calc the insection.
					float interX1 = Math.Max(x1, roi.Left);
					float interY1 = Math.Max(y1, roi.Top);
					float interX2 = Math.Min(x2, roi.Right);
					float interY2 = Math.Min(y2, roi.Bottom);
					if (interX1 < interX2 && interY1 < interY2)
					{
						LabelData newLabel = new LabelData();
						newLabel.LabelID = label.LabelID;
						newLabel.CenterX = (interX1 + interX2) / 2.0f - roi.Left + tempX;
						newLabel.CenterY = (interY1 + interY2) / 2.0f - roi.Top + tempY;
						newLabel.Width = interX2 - interX1;
						newLabel.Height = interY2 - interY1;
						newLabel.Radian = label.Radian;
						if (label.MaskOutLine is not null)
						{
							List<Point2f> newPoints = new List<Point2f>();
							foreach (var point in label.MaskOutLine)
							{
								float clampedX = Math.Clamp(point.X, roi.Left, roi.Right);
								float clampedY = Math.Clamp(point.Y, roi.Top, roi.Bottom);
								newPoints.Add(new Point2f(clampedX - roi.Left + tempX, clampedY - roi.Top + tempY));
							}
							if (newPoints.Count >= 3)
							{
								newLabel.MaskOutLine = newPoints.ToArray();
							}
						}
						if (imageDatas[0].ResizedLabels is null)
						{
							imageDatas[0].ResizedLabels = new List<LabelData>();
						}
						imageDatas[0].ResizedLabels.Add(newLabel);
					}
				}

			}

			return imageDatas[0];
		}

		private long[] Sample(long orgIndex, int min, int max, int count)
		{
			Random random = new Random();
			List<long> list = new List<long>();
			while (list.Count < count - 1)
			{
				int number = random.Next(min, max);
				if (!list.Contains(number) && number != orgIndex)
				{
					if (random.NextSingle() > 0.5f)
					{
						list.Add(number);
					}
					else
					{
						list.Insert(0, number);
					}
				}
			}
			int i = random.Next(0, count);
			list.Insert(i, orgIndex);

			return list.ToArray();
		}

		

	}
}
