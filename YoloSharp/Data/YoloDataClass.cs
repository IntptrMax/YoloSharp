using OpenCvSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace Data
{
	internal class YoloDataClass : utils.data.Dataset
	{
		private string rootPath = string.Empty;
		public int ImageSize => imageSize;
		private int imageSize = 640;
		private List<string> imageFiles = new List<string>();
		private ImageProcessType imageProcessType = ImageProcessType.Letterbox;
		private TaskType taskType = TaskType.Detection;
		private Device device;
		private int[] mosaic_border = new int[] { -320, -320 };

		public YoloDataClass(string rootPath, int imageSize = 640, TaskType taskType = TaskType.Detection, ImageProcessType imageProcessType = ImageProcessType.Letterbox)
		{
			torchvision.io.DefaultImager = new torchvision.io.SkiaImager();
			this.rootPath = rootPath;
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

		public ImageData GetImageAndLabelData(long index, ImageProcessType imageProcessType = ImageProcessType.Letterbox)
		{
			string imageFileName = imageFiles[(int)index];
			string labelFileName = GetLabelFileNameFromImageName(imageFileName);
			using (Mat orgImage = Cv2.ImRead(imageFileName))
			{
				int orgWidth = orgImage.Width;
				int orgHeight = orgImage.Height;

				if (!string.IsNullOrEmpty(labelFileName))
				{
					string[] strings = File.ReadAllLines(labelFileName);
					List<LabelData> labels = new List<LabelData>();
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

					ImageData imageData = new ImageData
					{
						ImagePath = imageFileName,
						OrgWidth = orgWidth,
						OrgHeight = orgHeight,
						OrgLabels = labels
					};

					switch (imageProcessType)
					{
						case ImageProcessType.Letterbox:
							LetterBox(imageData, imageSize);
							break;
						case ImageProcessType.Mosiac:
						default:
							throw new Exception($"The image process type {imageProcessType} is not supported.");
					}
					return imageData;
				}
				else
				{
					ImageData imageData = new ImageData
					{
						ImagePath = imageFileName,
						OrgWidth = orgWidth,
						OrgHeight = orgHeight,
						OrgLabels = new List<LabelData>()
					};

					switch (imageProcessType)
					{
						case ImageProcessType.Letterbox:
							LetterBox(imageData, imageSize);
							break;
						case ImageProcessType.Mosiac:
						default:
							throw new Exception($"The image process type {imageProcessType} is not supported.");
					}
					return imageData;
				}

			}
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



	}
}
