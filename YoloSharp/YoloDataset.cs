using SkiaSharp;
using System.Drawing;
using TorchSharp;
using static TorchSharp.torch;

namespace YoloSharp
{
	internal class YoloDataset : utils.data.Dataset
	{
		private string rootPath = string.Empty;
		private int imageSize = 640;
		private List<string> imageFiles = new List<string>();
		private bool useMosaic = true;
		private Device device;
		private int[] mosaic_border = new int[] { -320, -320 };

		public YoloDataset(string rootPath, int imageSize = 640, bool useMosaic = true, TorchSharp.DeviceType deviceType = TorchSharp.DeviceType.CUDA)
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
			foreach (string imageFileName in imagesFileNames)
			{
				string labelFileName = GetLabelFileNameFromImageName(imageFileName);
				if (!string.IsNullOrEmpty(labelFileName))
				{
					imageFiles.Add(imageFileName);
				}
			}
			this.imageSize = imageSize;
			this.useMosaic = useMosaic;
			device = new Device(deviceType);
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

		public string GetFileNameByIndex(long index)
		{
			return imageFiles[(int)index];
		}

		public override Dictionary<string, Tensor> GetTensor(long index)
		{
			Dictionary<string, Tensor> outputs = new Dictionary<string, Tensor>();
			outputs.Add("index", tensor(index));
			return outputs;
		}

		public (Tensor, Tensor) GetTensorByLetterBox(long index)
		{
			using var _ = NewDisposeScope();
			string file = imageFiles[(int)index];
			Tensor orgImageTensor = torchvision.io.read_image(file, torchvision.io.ImageReadMode.RGB).to(device);
			var (imgTensor, _, _) = Letterbox(orgImageTensor, imageSize, imageSize);
			Tensor lb = GetLetterBoxLabelTensor(index);
			return (imgTensor.unsqueeze(0).MoveToOuterDisposeScope(), lb.to(imgTensor.device).MoveToOuterDisposeScope());
		}

		public (Tensor, Tensor) GetTensorByMosaic(long index)
		{
			using var _ = NewDisposeScope();
			var (img, lb) = load_mosaic(index);
			return (img.unsqueeze(0).MoveToOuterDisposeScope(), lb.to(img.device).MoveToOuterDisposeScope());
		}

		public (Tensor, Tensor) GetDataTensor(long index)
		{
			using var _ = NewDisposeScope();
			var (image, label) = useMosaic ? GetTensorByMosaic(index) : GetTensorByLetterBox(index);
			if (image.shape.Length == 3)
			{
				image = image.unsqueeze(0);
			}
			image = image / 255.0f;
			return (image.MoveToOuterDisposeScope(), label.MoveToOuterDisposeScope());
		}

		public (Tensor, Tensor, Tensor) GetSegmentDataTensor(long index)
		{
			if (useMosaic)
			{
				Console.WriteLine("Mosaic method are not support now, it will come soon.");
				//load_mosaic(index);
			}
			return GetLetterBoxSegmentData(index);
		}


		private (Tensor, float, int) Letterbox(Tensor image, int targetWidth, int targetHeight)
		{
			using var _ = NewDisposeScope();
			// 获取图像的原始尺寸
			int originalWidth = (int)image.shape[2];
			int originalHeight = (int)image.shape[1];

			// 计算缩放比例
			float scale = Math.Min((float)targetWidth / originalWidth, (float)targetHeight / originalHeight);

			// 计算缩放后的尺寸
			int scaledWidth = (int)(originalWidth * scale);
			int scaledHeight = (int)(originalHeight * scale);

			// 计算填充后的尺寸
			int padLeft = (targetWidth - scaledWidth) / 2;
			//int padRight = targetWidth - scaledWidth - padLeft;
			int padTop = (targetHeight - scaledHeight) / 2;
			//int padBottom = targetHeight - scaledHeight - padTop;

			// 缩放图像
			Tensor scaledImage = torchvision.transforms.functional.resize(image, scaledHeight, scaledWidth);

			// 创建一个全零的张量，用于填充
			Tensor paddedImage = full(new long[] { 3, targetHeight, targetWidth }, 114, image.dtype, image.device);

			// 将缩放后的图像放置在填充后的图像中心
			paddedImage[TensorIndex.Ellipsis, padTop..(padTop + scaledHeight), padLeft..(padLeft + scaledWidth)].copy_(scaledImage);

			GC.Collect();

			return (paddedImage.MoveToOuterDisposeScope(), scale, Math.Max(padLeft, padTop));
		}

		public Tensor GetLetterBoxLabelTensor(long index)
		{
			using var _ = NewDisposeScope();
			Tensor orgImageTensor = torchvision.io.read_image(imageFiles[(int)index]);
			var (imgTensor, scale, pad) = Letterbox(orgImageTensor, imageSize, imageSize);
			bool isWidthLonger = orgImageTensor.shape[2] > orgImageTensor.shape[1];


			string labelName = GetLabelFileNameFromImageName(imageFiles[(int)index]);
			string[] lines = File.ReadAllLines(labelName);

			float[,] labelArray = new float[lines.Length, 5];

			for (int i = 0; i < lines.Length; i++)
			{
				string[] labels = lines[i].Split(' ');
				labelArray[i, 0] = float.Parse(labels[0]);
				if (isWidthLonger)
				{
					labelArray[i, 1] = float.Parse(labels[1]);
					labelArray[i, 3] = float.Parse(labels[3]);

					labelArray[i, 2] = (float.Parse(labels[2]) * (imageSize - 2 * pad) + pad) / imageSize;
					labelArray[i, 4] = float.Parse(labels[4]) * (imageSize - 2 * pad) / imageSize;
				}
				else
				{
					labelArray[i, 1] = (float.Parse(labels[1]) * (imageSize - 2 * pad) + pad) / imageSize;
					labelArray[i, 3] = float.Parse(labels[3]) * (imageSize - 2 * pad) / imageSize;

					labelArray[i, 2] = float.Parse(labels[2]);
					labelArray[i, 4] = float.Parse(labels[4]);
				}

			}
			Tensor labelTensor = tensor(labelArray);
			return labelTensor.MoveToOuterDisposeScope();

		}

		public (Tensor, Tensor, Tensor) GetLetterBoxSegmentData(long index)
		{
			using var _ = NewDisposeScope();
			int maskSize = 160;
			Tensor orgImageTensor = torchvision.io.read_image(imageFiles[(int)index], torchvision.io.ImageReadMode.RGB);

			int originalWidth = (int)orgImageTensor.shape[2];
			int originalHeight = (int)orgImageTensor.shape[1];

			float scale = Math.Min((float)imageSize / originalWidth, (float)imageSize / originalHeight);
			int padWidth = imageSize - (int)(scale * originalWidth);
			int padHeight = imageSize - (int)(scale * originalHeight);

			float maskWidthScale = scale * originalWidth / imageSize;
			float maskHeightScale = scale * originalHeight / imageSize;

			Tensor imgTensor = torchvision.transforms.functional.resize(orgImageTensor, (int)(originalHeight * scale), (int)(originalWidth * scale));
			imgTensor = torch.nn.functional.pad(imgTensor, new long[] { 0, padWidth, 0, padHeight }, PaddingModes.Zeros);

			Tensor outputImg = torch.zeros(new long[] { 3, imageSize, imageSize });
			outputImg[TensorIndex.Colon, ..(int)imgTensor.shape[1], ..(int)imgTensor.shape[2]] = imgTensor;

			string labelName = GetLabelFileNameFromImageName(imageFiles[(int)index]);
			string[] lines = File.ReadAllLines(labelName);
			float[,] labelArray = new float[lines.Length, 5];

			Tensor mask = torch.zeros(new long[] { maskSize, maskSize });
			for (int i = 0; i < lines.Length; i++)
			{
				string[] datas = lines[i].Split(' ');
				labelArray[i, 0] = float.Parse(datas[0]);

				List<PointF> points = new List<PointF>();
				for (int j = 1; j < datas.Length; j = j + 2)
				{
					points.Add(new PointF(float.Parse(datas[j]) * scale * originalWidth * maskSize / imageSize, float.Parse(datas[j + 1]) * scale * originalHeight * maskSize / imageSize));
				}

				float maxX = points.Max(p => p.X) / maskSize;
				float maxY = points.Max(p => p.Y) / maskSize;
				float minX = points.Min(p => p.X) / maskSize;
				float minY = points.Min(p => p.Y) / maskSize;

				float width = maxX - minX;
				float height = maxY - minY;
				labelArray[i, 1] = minX + width / 2;
				labelArray[i, 2] = minY + height / 2;
				labelArray[i, 3] = width;
				labelArray[i, 4] = height;

				using (var bitmap = new SKBitmap(maskSize, maskSize, SKColorType.Rgba8888, SKAlphaType.Premul))
				using (var canvas = new SKCanvas(bitmap))
				{
					// Fill with black background
					canvas.Clear(SKColors.Black);

					// Create paint for drawing
					using (var paint = new SKPaint())
					{
						paint.Color = SKColors.White;
						paint.Style = SKPaintStyle.Fill; // Fill the polygon
						paint.IsAntialias = true;       // Smooth edges

						// Convert points to SKPoint array
						SKPoint[]? skPoints = points.Select(p => new SKPoint((float)p.X, (float)p.Y)).ToArray();

						// Draw polygon
						using (var path = new SKPath())
						{
							path.AddPoly(skPoints);
							canvas.DrawPath(path, paint);
						}
					}
					Tensor msk = Lib.GetTensorFromImage(bitmap);
					msk = msk[0] > 0;
					mask[msk] = i + 1;
				}
			}
			Tensor labelTensor = tensor(labelArray);
			long p = imgTensor.shape[0];
			return (imgTensor.MoveToOuterDisposeScope(), labelTensor.MoveToOuterDisposeScope(), mask.MoveToOuterDisposeScope());
		}

		public Tensor GetOrgImage(long index)
		{
			return torchvision.io.read_image(imageFiles[(int)index], torchvision.io.ImageReadMode.RGB);
		}

		/// <summary>
		/// Loads a 4-image mosaic for YOLO, combining 1 selected and 3 random images, with labels and segments.
		/// </summary>
		/// <param name="index">The index in datasets</param>
		/// <returns></returns>
		public (Tensor, Tensor) load_mosaic(long index)
		{
			using var _ = NewDisposeScope();
			long[] indexs = Sample(index, 0, (int)Count, 4);
			Random random = new Random();
			int xc = random.Next(-mosaic_border[0], 2 * imageSize + mosaic_border[0]);
			int yc = random.Next(-mosaic_border[1], 2 * imageSize + mosaic_border[1]);

			var img4 = full(new long[] { 3, imageSize * 2, imageSize * 2 }, 114, torch.ScalarType.Byte, device); // base image with 4 tiles
			List<Tensor> label4 = new List<Tensor>();
			for (int i = 0; i < 4; i++)
			{
				int x1a = 0, y1a = 0, x2a = 0, y2a = 0, x1b = 0, y1b = 0, x2b = 0, y2b = 0;
				Tensor img = GetOrgImage(indexs[i]).to(device);
				//img = ResizeImage(img, resizeHeight);
				int h = (int)img.shape[1];
				int w = (int)img.shape[2];
				if (i == 0)  // top left
				{
					(x1a, y1a, x2a, y2a) = (Math.Max(xc - w, 0), Math.Max(yc - h, 0), xc, yc);  // xmin, ymin, xmax, ymax (large image))
					(x1b, y1b, x2b, y2b) = (w - (x2a - x1a), h - (y2a - y1a), w, h); // xmin, ymin, xmax, ymax (small image);
				}
				else if (i == 1)  // top right
				{
					(x1a, y1a, x2a, y2a) = (xc, Math.Max(yc - h, 0), Math.Min(xc + w, imageSize * 2), yc);
					(x1b, y1b, x2b, y2b) = (0, h - (y2a - y1a), Math.Min(w, x2a - x1a), h);
				}
				else if (i == 2)  // bottom left
				{
					(x1a, y1a, x2a, y2a) = (Math.Max(xc - w, 0), yc, xc, Math.Min(imageSize * 2, yc + h));
					(x1b, y1b, x2b, y2b) = (w - (x2a - x1a), 0, w, Math.Min(y2a - y1a, h));
				}
				else if (i == 3) // bottom right
				{
					(x1a, y1a, x2a, y2a) = (xc, yc, Math.Min(xc + w, imageSize * 2), Math.Min(imageSize * 2, yc + h));
					(x1b, y1b, x2b, y2b) = (0, 0, Math.Min(w, x2a - x1a), Math.Min(y2a - y1a, h));
				}
				img4[0..3, y1a..y2a, x1a..x2a] = img[0..3, y1b..y2b, x1b..x2b];

				int padw = x1a - x1b;
				int padh = y1a - y1b;

				Tensor labels = GetOrgLabelTensor(indexs[i]).to(device);
				labels[TensorIndex.Ellipsis, 1..5] = xywhn2xyxy(labels[TensorIndex.Ellipsis, 1..5], w, h, padw, padh);
				label4.Add(labels);
			}
			var labels4 = concat(label4, 0);

			labels4[TensorIndex.Ellipsis, 1..5] = labels4[TensorIndex.Ellipsis, 1..5].clip(0, 2 * imageSize);

			var (im, targets) = random_perspective(img4, labels4, degrees: 0, translate: 0.1f, scale: 0.5f, shear: 0, perspective: 0.0f, mosaic_border[0], mosaic_border[1]);

			targets[TensorIndex.Ellipsis, 1..5] = xyxy2xywhn(targets[TensorIndex.Ellipsis, 1..5], w: (int)im.shape[1], h: (int)im.shape[2], clip: true, eps: 1e-3f);
			return (im.MoveToOuterDisposeScope(), targets.MoveToOuterDisposeScope());
		}

		private Tensor ResizeImage(Tensor image, int targetWidth, int targetHeight)
		{
			// 获取图像的原始尺寸
			int originalWidth = (int)image.shape[2];
			int originalHeight = (int)image.shape[1];

			// 计算缩放比例
			float scale = Math.Min((float)targetWidth / originalWidth, (float)targetHeight / originalHeight);

			// 计算缩放后的尺寸
			int scaledWidth = (int)(originalWidth * scale);
			int scaledHeight = (int)(originalHeight * scale);

			return torchvision.transforms.functional.resize(image, scaledWidth, scaledHeight);
		}

		private Tensor ResizeImage(Tensor image, int targetSize)
		{
			return ResizeImage(image, targetSize, targetSize);
		}

		/// <summary>
		/// Get several random numbers between min and max, and contains orgIndex.
		/// </summary>
		/// <param name="orgIndex"></param>
		/// <param name="min"></param>
		/// <param name="max"></param>
		/// <param name="count"></param>
		/// <returns></returns>
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

		/// <summary>
		/// Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="w"></param>
		/// <param name="h"></param>
		/// <param name="padw"></param>
		/// <param name="padh"></param>
		/// <returns></returns>
		private Tensor xywhn2xyxy(Tensor x, int w = 640, int h = 640, int padw = 0, int padh = 0)
		{
			using var _ = NewDisposeScope();
			Tensor y = x.clone();
			y[TensorIndex.Ellipsis, 0] = w * (x[TensorIndex.Ellipsis, 0] - x[TensorIndex.Ellipsis, 2] / 2) + padw;  // top left x
			y[TensorIndex.Ellipsis, 1] = h * (x[TensorIndex.Ellipsis, 1] - x[TensorIndex.Ellipsis, 3] / 2) + padh;  // top left y
			y[TensorIndex.Ellipsis, 2] = w * (x[TensorIndex.Ellipsis, 0] + x[TensorIndex.Ellipsis, 2] / 2) + padw;  // bottom right x
			y[TensorIndex.Ellipsis, 3] = h * (x[TensorIndex.Ellipsis, 1] + x[TensorIndex.Ellipsis, 3] / 2) + padh;  // bottom right y
			return y.MoveToOuterDisposeScope();
		}

		/// <summary>
		/// Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="w"></param>
		/// <param name="h"></param>
		/// <param name="clip"></param>
		/// <param name="eps"></param>
		/// <returns></returns>
		private Tensor xyxy2xywhn(Tensor x, int w = 640, int h = 640, bool clip = false, float eps = 0.0f)
		{
			using var _ = NewDisposeScope();
			if (clip)
			{
				x = Lib.ClipBox(x, new float[] { h - eps, w - eps });
			}
			var y = x.clone();
			y[TensorIndex.Ellipsis, 0] = (x[TensorIndex.Ellipsis, 0] + x[TensorIndex.Ellipsis, 2]) / 2 / w;  // x center
			y[TensorIndex.Ellipsis, 1] = (x[TensorIndex.Ellipsis, 1] + x[TensorIndex.Ellipsis, 3]) / 2 / h;// y center
			y[TensorIndex.Ellipsis, 2] = (x[TensorIndex.Ellipsis, 2] - x[TensorIndex.Ellipsis, 0]) / w;  // width
			y[TensorIndex.Ellipsis, 3] = (x[TensorIndex.Ellipsis, 3] - x[TensorIndex.Ellipsis, 1]) / h;  // height
			return y.MoveToOuterDisposeScope();
		}

		private Tensor GetOrgLabelTensor(long index)
		{
			using var _ = NewDisposeScope();
			string labelName = GetLabelFileNameFromImageName(imageFiles[(int)index]);
			string[] lines = File.ReadAllLines(labelName);

			float[,] labelArray = new float[lines.Length, 5];

			for (int i = 0; i < lines.Length; i++)
			{
				string[] labels = lines[i].Split(' ');
				for (int j = 0; j < labels.Length; j++)
				{
					labelArray[i, j] = float.Parse(labels[j]);
				}
			}
			Tensor labelTensor = tensor(labelArray);
			return labelTensor.MoveToOuterDisposeScope();
		}

		private (Tensor, Tensor) GetOrgMaskLabelTensor(long index, int width = 160, int height = 160)
		{
			using var _ = NewDisposeScope();
			string labelName = GetLabelFileNameFromImageName(imageFiles[(int)index]);
			string[] lines = File.ReadAllLines(labelName);

			List<Tensor> labels = new List<Tensor>();
			List<Tensor> masks = new List<Tensor>();

			foreach (string line in lines)
			{
				// Parse points from label file
				List<SKPoint> points = new List<SKPoint>();
				string[] strs = line.Split(' ');
				for (int i = 0; i < strs.Length / 2; i++)
				{
					float x = float.Parse(strs[i * 2 + 1]) * width;
					float y = float.Parse(strs[i * 2 + 2]) * height;
					points.Add(new SKPoint(x, y));
				}

				// Calculate bounding box
				float x_max = points.Max(a => a.X);
				float y_max = points.Max(a => a.Y);
				float x_min = points.Min(a => a.X);
				float y_min = points.Min(a => a.Y);

				labels.Add(torch.tensor(new float[] { x_min, y_min, x_max - x_min, y_max - y_min }).unsqueeze(0));

				// Create mask using SkiaSharp
				using (var bitmap = new SKBitmap(width, height, SKColorType.Rgba8888, SKAlphaType.Premul))
				using (var canvas = new SKCanvas(bitmap))
				{
					// Fill with black background
					canvas.Clear(SKColors.Black);

					// Draw white polygon
					using (var paint = new SKPaint())
					{
						paint.Color = SKColors.White;
						paint.Style = SKPaintStyle.Fill;
						paint.IsAntialias = true;

						using (var path = new SKPath())
						{
							path.AddPoly(points.ToArray());
							canvas.DrawPath(path, paint);
						}
					}

					// Convert SKBitmap to Tensor
					var bytes = bitmap.Bytes;
					Tensor ts = torch.tensor(bytes, new long[] { 1, height, width }, torch.uint8);
					masks.Add(ts.unsqueeze(0));
				}
			}

			Tensor labelTensor = torch.cat(labels.ToArray());
			Tensor maskTensor = torch.cat(masks.ToArray());
			return (labelTensor.MoveToOuterDisposeScope(), maskTensor.MoveToOuterDisposeScope());
		}

		private (Tensor, Tensor) random_perspective(Tensor im, Tensor targets, int degrees = 10, float translate = 0.1f, float scale = 0.1f, int shear = 10, float perspective = 0.0f, int borderX = 0, int borderY = 0)
		{
			using var _ = NewDisposeScope();
			Device device = im.device;
			int height = (int)im.shape[1] + borderY * 2;
			int width = (int)im.shape[2] + borderX * 2;

			// Center
			Tensor C = eye(3).to(device);
			C[0, 2] = -im.shape[2] / 2; // x translation (pixels)
			C[1, 2] = -im.shape[1] / 2; // y translation (pixels)

			//Perspective
			Tensor P = eye(3).to(device);
			P[2, 0] = rand(1).ToSingle() * 2 * perspective - perspective;   // x perspective (about y)
			P[2, 1] = rand(1).ToSingle() * 2 * perspective - perspective;   // y perspective (about x)

			// Rotation and Scale
			float a = rand(1).ToSingle() * 2 * degrees - degrees;
			float s = 1 + scale - rand(1).ToSingle() * 2 * scale;

			Tensor R = GetRotationMatrix2D(angle: a, scale: s).to(device);

			// Shear
			Tensor S = eye(3).to(device);
			S[0, 1] = Math.Tan((rand(1).ToSingle() * 2 * shear - shear) * Math.PI / 180); // x shear (deg)
			S[1, 0] = Math.Tan((rand(1).ToSingle() * 2 * shear - shear) * Math.PI / 180); // y shear (deg)

			// Translation
			Tensor T = eye(3).to(device);
			T[0, 2] = (0.5f + translate - rand(1).ToSingle() * 2 * translate) * width;    // x translation(pixels)
			T[1, 2] = (0.5f + translate - rand(1).ToSingle() * 2 * translate) * height;   // y translation(pixels)

			//var M = T.mm(S).mm(R).mm(P).mm(C);
			var M = T.matmul(S).matmul(R).matmul(P).matmul(C);

			Tensor outTensor = zeros(new long[] { imageSize, imageSize, 3 }, torch.ScalarType.Byte).to(device);
			if (borderX != 0 || borderY != 0 || M.bytes != eye(3).bytes)
			{
				if (perspective != 0)
				{
					//im = WarpPerspective(im, M, width, height, (114, 114, 114));

					// 定义原始图像的四个角点
					var corners = tensor(new float[,]
										{{ 0, 0, 1 },					// 左上
										{ width - 1, 0, 1 },			// 右上
										{ 0, height - 1, 1 },			// 左下
										{ width - 1, height - 1, 1 }	// 右下
										}).to(device);

					// 将角点与变换矩阵 M 相乘
					var transformedCorners = corners.matmul(M.T);

					// 透视变换后，需要将齐次坐标归一化
					transformedCorners = transformedCorners[TensorIndex.Ellipsis, 0..2] / transformedCorners[TensorIndex.Ellipsis, 2..3];

					// 提取变换后的四个角点
					var topLeft = transformedCorners[0];      // 左上
					var topRight = transformedCorners[1];     // 右上
					var bottomLeft = transformedCorners[2];   // 左下
					var bottomRight = transformedCorners[3];  // 右下

					// 定义填充颜色 (R, G, B)
					var fillColor = new List<float> { 114, 114, 114 };

					// 定义原始图像的四个角点
					var startpoints = new List<List<int>>
									{
										new List<int> { 0, 0 },					// 左上
										new List<int> { width - 1, 0 },			// 右上
										new List<int> { 0, height - 1 },		// 左下
										new List<int> { width - 1, height - 1 } // 右下
									};

					// 定义变换后的四个角点
					List<List<int>> endpoints = new List<List<int>>
									{
										new List<int> { transformedCorners[0, 0].ToInt32(), transformedCorners[0, 1].ToInt32() }, // 左上
										new List<int> { transformedCorners[1, 0].ToInt32(), transformedCorners[1, 1].ToInt32() }, // 右上
										new List<int> { transformedCorners[2, 0].ToInt32(), transformedCorners[2, 1].ToInt32() }, // 左下
										new List<int> { transformedCorners[3, 0].ToInt32(), transformedCorners[3, 1].ToInt32() }  // 右下
									};

					// 显式转换为 IList<IList<int>>
					IList<IList<int>> startpointsIList = startpoints.Select(list => (IList<int>)list).ToList();
					IList<IList<int>> endpointsIList = endpoints.Select(list => (IList<int>)list).ToList();

					// 调用透视变换函数
					im = torchvision.transforms.functional.perspective(
						im,                         // 输入图像
						startpointsIList,           // 原始图像的四个角点
						endpointsIList,             // 变换后的四个角点
						InterpolationMode.Bilinear, // 插值方式
						fillColor                   // 填充颜色
					);

				}
				else
				{
					// 提取仿射变换的参数
					var shearParams = new List<float> { S[0, 1].ToSingle(), S[1, 0].ToSingle() };
					var translateParams = new List<int> { T[0, 2].ToInt32(), T[1, 2].ToInt32() };

					outTensor = torchvision.transforms.functional.affine(im, shearParams, a, translateParams, s);
					outTensor = torchvision.transforms.functional.crop(outTensor, (int)im.shape[1] - imageSize, (int)im.shape[2] - imageSize, imageSize, imageSize).contiguous();
				}
			}

			long n = targets.shape[0];
			if (n > 0)
			{
				Tensor newT = zeros(new long[] { n, 4 }).to(device);
				Tensor xy = ones(new long[] { n * 4, 3 }).to(device);
				xy[TensorIndex.Ellipsis, 0..2] = targets.index_select(1, tensor(new long[] { 1, 2, 3, 4, 1, 4, 3, 2 }).to(device)).reshape(n * 4, 2).to(device);  // x1y1, x2y2, x1y2, x2y1
				xy = xy.mm(M.T);
				xy = perspective == 0 ? xy[TensorIndex.Ellipsis, 0..2].reshape(n, 8) : xy[TensorIndex.Ellipsis, 0..2] / xy[TensorIndex.Ellipsis, 2..3];
				Tensor x = xy.index_select(1, tensor(new long[] { 0, 2, 4, 6 }).to(device));
				Tensor y = xy.index_select(1, tensor(new long[] { 1, 3, 5, 7 }).to(device));
				newT = concatenate(new Tensor[] { x.min(1).values, y.min(1).values, x.max(1).values, y.max(1).values }).reshape(4, n).T;

				newT = newT.index_put_(newT.index_select(1, tensor(new long[] { 0, 2 }).to(device)).clip(0, imageSize), new TensorIndex[] { TensorIndex.Ellipsis, TensorIndex.Slice(0, 3, 2) });
				newT = newT.index_put_(newT.index_select(1, tensor(new long[] { 1, 3 }).to(device)).clip(0, imageSize), new TensorIndex[] { TensorIndex.Ellipsis, TensorIndex.Slice(1, 4, 2) });

				Tensor idx = box_candidates(box1: targets[TensorIndex.Ellipsis, 1..5].T * s, box2: newT.T, area_thr: 0.1f);
				targets = targets[idx];
				targets[TensorIndex.Ellipsis, 1..5] = newT[idx];
			}

			return (outTensor.contiguous().MoveToOuterDisposeScope(), targets.MoveToOuterDisposeScope());
		}

		public static Tensor GetRotationMatrix2D(float angle, float scale)
		{
			using var _ = NewDisposeScope();
			// 将角度转换为弧度
			float theta = angle * (float)Math.PI / 180.0f;

			// 计算旋转矩阵的元素
			float cosTheta = (float)Math.Cos(theta);
			float sinTheta = (float)Math.Sin(theta);

			// 创建旋转矩阵
			var R = tensor(new float[,]
			{
				{ scale * cosTheta, -scale * sinTheta, 0 },
				{ scale * sinTheta, scale * cosTheta, 0 },
				{ 0, 0, 1 }
			});
			return R.MoveToOuterDisposeScope();
		}

		/// <summary>
		/// Filters bounding box candidates by minimum width-height threshold `wh_thr` (pixels), aspect ratio threshold `ar_thr`, and area ratio threshold `area_thr`.
		/// </summary>
		/// <param name="box1">(4,n) is before augmentation</param>
		/// <param name="box2">(4,n) is after augmentation</param>
		/// <param name="wh_thr"></param>
		/// <param name="ar_thr"></param>
		/// <param name="area_thr"></param>
		/// <param name="eps"></param>
		/// <returns></returns>
		private Tensor box_candidates(Tensor box1, Tensor box2, float wh_thr = 2, float ar_thr = 100, float area_thr = 0.1f, double eps = 1e-16)
		{
			using var _ = NewDisposeScope();
			var (w1, h1) = (box1[2] - box1[0], box1[3] - box1[1]);
			var (w2, h2) = (box2[2] - box2[0], box2[3] - box2[1]);
			var ar = maximum(w2 / (h2 + eps), h2 / (w2 + eps)); // aspect ratio
			Tensor result = w2 > wh_thr & h2 > wh_thr & w2 * h2 / (w1 * h1 + eps) > area_thr & ar < ar_thr; // candidates
			return result.MoveToOuterDisposeScope();
		}

	}
}
