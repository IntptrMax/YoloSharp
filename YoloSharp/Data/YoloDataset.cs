using OpenCvSharp;
using TorchSharp;
using YoloSharp.Types;
using YoloSharp.Utils;
using static TorchSharp.torch;

namespace YoloSharp.Data
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
		List<string> ClasssNames;

		int kpt_count = 0;

		torchvision.ITransform transform;

		public YoloDataset(string rootPath, string dataPath = "", int imageSize = 640, TaskType taskType = TaskType.Detection, ImageProcessType imageProcessType = ImageProcessType.Letterbox, float brightness = 0.1f, float contrast = 0.1f, float saturation = 0.1f, float hue = 0.02f)
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
				if (Directory.Exists(path))
				{
					string[] imagesFileNames = Directory.GetFiles(path).Where(line =>
					{
						string trimmedLine = line.Trim();
						if (string.IsNullOrEmpty(trimmedLine))
						{
							return false;
						}
						string extension = Path.GetExtension(trimmedLine).ToLower();
						return extension == ".jpg" || extension == ".png" || extension == ".bmp";
					}).Select(line => Path.IsPathRooted(line) ? Path.GetFullPath(line) : Path.GetFullPath(Path.Combine(rootPath, line.Trim()))).ToArray();

					imageFiles.AddRange(imagesFileNames);
				}
				else
				{
					if (!File.Exists(path))
					{
						throw new FileNotFoundException($"The file {path} does not exist.");
					}
					string[] imagesFileNames = File.ReadAllLines(path).Where(line =>
					{
						string trimmedLine = line.Trim();
						if (string.IsNullOrEmpty(trimmedLine))
						{
							return false;
						}
						string extension = Path.GetExtension(trimmedLine).ToLower();
						return extension == ".jpg" || extension == ".png" || extension == ".bmp";
					}).Select(line => Path.IsPathRooted(line) ? Path.GetFullPath(line) : Path.GetFullPath(Path.Combine(rootPath, line.Trim()))).ToArray();

					imageFiles.AddRange(imagesFileNames);
				}
			}

			if (taskType == TaskType.Classification)
			{
				ClasssNames = new List<string>();
				DirectoryInfo[] Directories = Directory.GetParent(imageFiles[0]).Parent.GetDirectories();
				foreach (DirectoryInfo Directory in Directories)
				{
					ClasssNames.Add(Directory.Name);
				}
				transform = torchvision.transforms.Compose(new torchvision.ITransform[] {
						torchvision.transforms.Resize(imageSize+32,imageSize+32),
						torchvision.transforms.RandomResizedCrop(imageSize, imageSize,0.9),
						torchvision.transforms.RandomHorizontalFlip( 0.1),
						torchvision.transforms.RandomVerticalFlip(0.1),
						torchvision.transforms.RandomRotation(15),
						torchvision.transforms.ColorJitter(brightness: brightness, contrast: contrast, saturation: saturation, hue: hue),
				});
			}
			else
			{
				transform = torchvision.transforms.ColorJitter(brightness: brightness, contrast: contrast, saturation: saturation, hue: hue);
			}

			this.imageSize = imageSize;
			this.imageProcessType = imageProcessType;
			this.taskType = taskType;
		}

		private string GetLabelFileNameFromImageName(string imageFileName)
		{
			string imagesFolder = Path.GetFullPath(Path.Combine(rootPath, "images"));
			string labelsFolder = Path.GetFullPath(Path.Combine(rootPath, "labels"));
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
			//Dictionary<string, Tensor> outputs = new Dictionary<string, Tensor>();
			//outputs.Add("index", tensor(index));
			//return outputs;


			return GetTargets(index);
		}

		public static Mat GetMaskFromOutlinePoints(Point[] points, int width, int height)
		{
			Mat mask = Mat.Zeros(height, width, MatType.CV_8UC1);
			Point[][] pts = new Point[1][];
			pts[0] = points.Select(p => new Point((int)p.X, (int)p.Y)).ToArray();
			Cv2.FillPoly(mask, pts, OpenCvSharp.Scalar.White);
			return mask;
		}

		public ImageData GetImageAndLabelData(long index)
		{
			if (taskType == TaskType.Classification)
			{
				return GetImageAndLabelDataForClassification(index);
			}

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

									Point[] maskOutlinePoints = new Point[(parts.Length - 1) / 2];
									for (int i = 0; i < maskOutlinePoints.Length; i++)
									{
										maskOutlinePoints[i] = new Point(float.Parse(parts[1 + i * 2]) * orgWidth, float.Parse(parts[2 + i * 2]) * orgHeight);
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
							case TaskType.Pose:
								{
									if (parts.Length < 8)
									{
										throw new Exception($"The label file {labelFileName} format is incorrect.");
									}

									LabelData labelData = new LabelData()
									{
										LabelID = int.Parse(parts[0]),
										CenterX = float.Parse(parts[1]) * orgWidth,
										CenterY = float.Parse(parts[2]) * orgHeight,
										Width = float.Parse(parts[3]) * orgWidth,
										Height = float.Parse(parts[4]) * orgHeight,
									};
									int pointCount = (parts.Length - 5) / 3;
									kpt_count = pointCount;
									Types.KeyPoint[] keyPoints = new Types.KeyPoint[pointCount];
									for (int i = 0; i < pointCount; i++)
									{
										keyPoints[i] = new Types.KeyPoint
										{
											X = float.Parse(parts[i * 3 + 5 + 0]) * orgWidth,
											Y = float.Parse(parts[i * 3 + 5 + 1]) * orgHeight,
											VisibilityScore = float.Parse(parts[i * 3 + 5 + 2])
										};
									}
									labelData.KeyPoints = keyPoints;
									labels.Add(labelData);

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
						resizedLabel.MaskOutLine = new Point[label.MaskOutLine.Length];
						for (int i = 0; i < label.MaskOutLine.Length; i++)
						{
							resizedLabel.MaskOutLine[i] = new Point(label.MaskOutLine[i].X * r + dw, label.MaskOutLine[i].Y * r + dh);
						}
					}
					if (label.KeyPoints is not null)
					{
						resizedLabel.KeyPoints = new Types.KeyPoint[label.KeyPoints.Length];
						for (int i = 0; i < label.KeyPoints.Length; i++)
						{
							resizedLabel.KeyPoints[i] = new Types.KeyPoint(label.KeyPoints[i].X * r + dw, label.KeyPoints[i].Y * r + dh, label.KeyPoints[i].VisibilityScore);
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
			int ind = indices.ToList().IndexOf(index);
			ImageData result = new ImageData();

			int w = (ind == 0 || ind == 2) ? random.Next(imageSize / 2, imageSize - 1) : random.Next(1, imageSize / 2);
			int h = (ind == 0 || ind == 1) ? random.Next(imageSize / 2, imageSize - 1) : random.Next(1, imageSize / 2);

			Mat mosaicMat = new Mat(imageSize, imageSize, MatType.CV_8UC3, new OpenCvSharp.Scalar(114, 114, 114));

			for (int i = 0; i < imgCount; i++)
			{
				imageDatas[i] = GetOrgImageAndLabelData(indices[i]);
				Mat eachOrgMat = imageDatas[i].OrgImage;

				// mosaic cropped background x, y, w, h
				int croppedX = (i == 0 || i == 2) ? 0 : w;
				int croppedY = (i == 0 || i == 1) ? 0 : h;
				int croppedW = (i == 0 || i == 2) ? w : imageSize - w;
				int croppedH = (i == 0 || i == 1) ? h : imageSize - h;

				int randomX = random.Next(0, Math.Max(0, eachOrgMat.Width - croppedW));
				int randomY = random.Next(0, Math.Max(0, eachOrgMat.Height - croppedH));

				if (i == ind)
				{
					if (imageDatas[i].OrgLabels.Count > 0)
					{
						int ii = random.Next(0, imageDatas[i].OrgLabels.Count);
						int cx = (int)imageDatas[i].OrgLabels[ii].CenterX;
						int cy = (int)imageDatas[i].OrgLabels[ii].CenterY;

						randomX = Math.Clamp(randomX, cx - croppedW, cx + croppedW);
						randomY = Math.Clamp(randomY, cy - croppedH, cy + croppedH);

					}
				}

				// roi in org image.
				Rect roi = new Rect(randomX, randomY, Math.Min(croppedW, eachOrgMat.Width - randomX), Math.Min(croppedH, eachOrgMat.Height - randomY));
				Mat cropped = new Mat(eachOrgMat, roi);
				cropped.CopyTo(mosaicMat[new Rect(croppedX, croppedY, roi.Width, roi.Height)]);

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
						newLabel.CenterX = (interX1 + interX2) / 2.0f - roi.Left + croppedX;
						newLabel.CenterY = (interY1 + interY2) / 2.0f - roi.Top + croppedY;
						newLabel.Width = interX2 - interX1;
						newLabel.Height = interY2 - interY1;
						newLabel.Radian = label.Radian;
						if (label.MaskOutLine is not null)
						{
							List<Point> newPoints = new List<Point>();
							foreach (var point in label.MaskOutLine)
							{
								float clampedX = Math.Clamp(point.X, roi.Left, roi.Right);
								float clampedY = Math.Clamp(point.Y, roi.Top, roi.Bottom);
								newPoints.Add(new Point(clampedX - roi.Left + croppedX, clampedY - roi.Top + croppedY));
							}
							if (newPoints.Count >= 3)
							{
								newLabel.MaskOutLine = newPoints.ToArray();
							}
						}
						if (label.KeyPoints is not null)
						{
							List<Types.KeyPoint> newPoints = new List<Types.KeyPoint>();
							foreach (var point in label.KeyPoints)
							{
								float vis = (point.X < roi.Left || point.X > roi.Right || point.Y < roi.Top || point.Y > roi.Bottom) ? 0 : point.VisibilityScore;
								newPoints.Add(new Types.KeyPoint(point.X - roi.Left + croppedX, point.Y - roi.Top + croppedY, vis));
							}
							newLabel.KeyPoints = newPoints.ToArray();
						}
						if (result.ResizedLabels is null)
						{
							result.ResizedLabels = new List<LabelData>();
						}
						result.ResizedLabels.Add(newLabel);
					}
				}

			}
			result.ResizedImage = mosaicMat;
			result.OrgLabels = imageDatas[ind].OrgLabels;
			result.ImagePath = imageDatas[ind].ImagePath;

			return result;
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

		private void DrawResizedLabels(ImageData data, bool drawSegment = false)
		{
			Mat resizedImage = data.ResizedImage;

			// Draw segment
			if (drawSegment)
			{
				foreach (var result in data.ResizedLabels)
				{
					Cv2.FillPoly(resizedImage, new Point[][] { result.MaskOutLine.Select(x => new Point(x.X, x.Y)).ToArray() }, OpenCvSharp.Scalar.Red);
				}
				resizedImage.SaveImage("segment.jpg");
			}
			// Draw box
			else
			{
				foreach (var result in data.ResizedLabels)
				{
					float[] cxcywhr = new float[] { result.CenterX, result.CenterY, result.Width, result.Height, result.Radian };
					float[] points = Utils.Ops.cxcywhr2xyxyxyxy(cxcywhr);
					Point[] pts = new Point[4]
					{
						new Point(points[0], points[1]),
						new Point(points[2], points[3]),
						new Point(points[4], points[5]),
						new Point(points[6], points[7]),
					};
					Cv2.Polylines(resizedImage, new Point[][] { pts }, true, OpenCvSharp.Scalar.Red, 2);
					if (result.KeyPoints is not null)
					{
						foreach (var point in result.KeyPoints)
						{
							if (point.VisibilityScore != 0)
							{
								Cv2.Circle(resizedImage, (int)point.X, (int)point.Y, 2, OpenCvSharp.Scalar.Red);
							}
						}
					}

				}
				resizedImage.SaveImage("box.jpg");
			}

		}

		private ImageData GetImageAndLabelDataForClassification(long index)
		{
			string imageFileName = imageFiles[(int)index];
			string labelName = Directory.GetParent(imageFileName).Name;
			int id = ClasssNames.IndexOf(labelName);
			Mat orgImage = Cv2.ImRead(imageFileName);
			Mat resized = new Mat();
			Cv2.Resize(orgImage, resized, new OpenCvSharp.Size(imageSize, imageSize));

			ImageData imageData = new ImageData
			{
				ImagePath = imageFileName,
				OrgWidth = orgImage.Width,
				OrgHeight = orgImage.Height,
				OrgLabels = new List<LabelData>()
				{
					new LabelData()
					{
						LabelID = id
					}
				}
			};

			imageData.ResizedImage = resized;
			imageData.ResizedLabels = new List<LabelData>();
			imageData.ResizedLabels.Add(new LabelData()
			{
				LabelID = id
			});
			return imageData;
		}

		internal Dictionary<string, Tensor> GetTargets(long index)
		{
			using (NewDisposeScope())
			using (no_grad())
			{
				ImageData imageData = GetImageAndLabelData(index);
				int maskSize = imageSize / 4;
				int count = (imageData.ResizedLabels?.Count).GetValueOrDefault();
				Tensor imageTensor = Lib.GetTensorFromImage(imageData.ResizedImage).unsqueeze(0) / 255.0f;

				Tensor cls_tensor = count > 0 ? tensor(imageData.ResizedLabels.Select(x => (float)x.LabelID).ToArray()).unsqueeze(-1) : tensor(new float[0, 1]);

				if (TaskType.Classification == taskType)
				{
					cls_tensor = tensor(imageData.ResizedLabels.Select(x => (float)x.LabelID).ToArray(), torch.ScalarType.Int64);
					imageTensor = transform.call(imageTensor);

					return new Dictionary<string, Tensor>()
					{
						{ "cls", cls_tensor.MoveToOuterDisposeScope() },
						{ "images", imageTensor.MoveToOuterDisposeScope()},
					};
				}

				imageTensor = transform.call(imageTensor);

				Tensor bboxes_tensor = taskType switch
				{
					TaskType.Obb => count > 0 ? cat(imageData.ResizedLabels.Select(x => tensor(new float[] { x.CenterX / imageSize, x.CenterY / imageSize, x.Width / imageSize, x.Height / imageSize, x.Radian }).unsqueeze(0)).ToArray(), 0) : tensor(new float[0, 5]),
					_ => count > 0 ? cat(imageData.ResizedLabels.Select(x => tensor(new float[] { x.CenterX, x.CenterY, x.Width, x.Height }).unsqueeze(0)).ToArray(), 0) / imageSize : tensor(new float[0, 4]),
				};

				Tensor mask_tensor = torch.zeros(new long[] { 0, 1, maskSize, maskSize });

				if (TaskType.Segmentation == taskType)
				{
					using (Mat maskMat = new Mat(maskSize, maskSize, MatType.CV_8UC1, new OpenCvSharp.Scalar(0)))
					{
						for (int j = 0; j < count; j++)
						{
							Point[] points = imageData.ResizedLabels[j].MaskOutLine.Select(p => p.Multiply((float)maskSize / imageSize)).ToArray();
							using (Mat eachMaskMat = GetMaskFromOutlinePoints(points, maskSize, maskSize))
							using (Mat foreMat = new Mat(maskSize, maskSize, MatType.CV_8UC1, new OpenCvSharp.Scalar(j + 1f)))
							{
								foreMat.CopyTo(maskMat, eachMaskMat);
							}
						}
						mask_tensor = Lib.GetTensorFromImage(maskMat, torchvision.io.ImageReadMode.GRAY).unsqueeze(0);
					}
				}

				Tensor kpt_tensor = torch.zeros(new long[] { 0, kpt_count, 3 });
				if (TaskType.Pose == taskType)
				{
					List<Tensor> kpts = new List<Tensor>();
					imageData.ResizedLabels.Select(x =>
					{
						float[] kpt_array = new float[x.KeyPoints.Count() * 3];
						for (int j = 0; j < x.KeyPoints.Count(); j++)
						{
							kpt_array[j * 3] = x.KeyPoints[j].X / imageData.ResizedImage.Width;
							kpt_array[j * 3 + 1] = x.KeyPoints[j].Y / imageData.ResizedImage.Height;
							kpt_array[j * 3 + 2] = x.KeyPoints[j].VisibilityScore;
						}
						return tensor(kpt_array).view(x.KeyPoints.Count(), 3);
					}).ToList().ForEach(x => kpts.Add(x.unsqueeze(0)));
					if (kpts.Count > 0)
					{
						kpt_tensor = cat(kpts.ToArray(), 0);
					}
				}

				Dictionary<string, Tensor> targets = new Dictionary<string, Tensor>()
				{
					{ "cls", cls_tensor.MoveToOuterDisposeScope() },
					{ "bboxes", bboxes_tensor.MoveToOuterDisposeScope() },
					{ "images", imageTensor.MoveToOuterDisposeScope()},
					{ "masks", mask_tensor.MoveToOuterDisposeScope()},
					{ "keypoints", kpt_tensor.MoveToOuterDisposeScope()}
				};

				GC.Collect();
				return targets;
			}
		}

	}
}
