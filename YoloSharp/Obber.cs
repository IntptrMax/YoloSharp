using SkiaSharp;
using System.Text.RegularExpressions;
using TorchSharp;
using Utils;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static YoloSharp.Yolo;

namespace YoloSharp
{
	public class Obber
	{
		private readonly torch.Device device;
		private readonly torch.ScalarType dtype;
		private readonly int sortCount;
		private readonly YoloType yoloType;
		private Module<Tensor, Tensor[]> yolo;

		public Obber(int sortCount = 15, YoloType yoloType = YoloType.Yolov8, YoloSize yoloSize = YoloSize.n, DeviceType deviceType = DeviceType.CUDA, ScalarType dtype = ScalarType.Float32)
		{
			torchvision.io.DefaultImager = new torchvision.io.SkiaImager();
			if (yoloType == YoloType.Yolov5 || yoloType == YoloType.Yolov5u || yoloType == YoloType.Yolov12)
			{
				throw new ArgumentException("Obb not support yolov5, yolov5u or yolov12. Please use yolov8 or yolov11 instead.");
			}

			this.device = new torch.Device((TorchSharp.DeviceType)deviceType);
			this.dtype = (torch.ScalarType)dtype;
			this.sortCount = sortCount;
			this.yoloType = yoloType;
			yolo = yoloType switch
			{
				YoloType.Yolov8 => new Yolov8Obb(sortCount, yoloSize, device, this.dtype),
				YoloType.Yolov11 => new Yolov11Obb(sortCount, yoloSize, device, this.dtype),
				_ => throw new NotImplementedException("Yolo type not supported."),
			};

			//Tools.TransModelFromSafetensors(yolo, @".\yolov8n-obb.safetensors", @".\PreTrainedModels\yolov8n-obb.bin");
		}

		public void LoadModel(string path, bool skipNcNotEqualLayers = false)
		{
			Dictionary<string, Tensor> state_dict = Lib.LoadModel(path, skipNcNotEqualLayers);
			if (state_dict.Count != yolo.state_dict().Count)
			{
				Console.WriteLine("Mismatched tensors count while loading. Model will run with random weight.");
			}
			else
			{
				torch.ScalarType modelType = state_dict.Values.First().dtype;
				yolo.to(modelType);

				List<string> skipList = new();
				if (skipNcNotEqualLayers)
				{
					string layerPattern = yoloType switch
					{
						YoloType.Yolov8 => @"model\.22\.cv3",
						YoloType.Yolov11 => @"model\.23\.cv3",
						_ => string.Empty
					};

					if (!string.IsNullOrEmpty(layerPattern))
					{
						skipList = state_dict.Keys.Where(x => Regex.IsMatch(x, layerPattern)).ToList();
						if (state_dict[skipList.LastOrDefault(a => a.EndsWith(".bias"))!].shape[0] == sortCount)
						{
							skipList.Clear();
						}
					}
				}

				var (miss, err) = yolo.load_state_dict(state_dict, skip: skipList);
				if (skipList.Count > 0)
				{
					Console.WriteLine("Warning! Skipping nc reference layers. This may cause incorrect predictions.");
				}
				yolo.to(dtype);
			}
		}

		public List<YoloResult> ImagePredict(SKBitmap image, float PredictThreshold = 0.25f, float IouThreshold = 0.5f)
		{
			using var _ = no_grad();
			yolo.eval();
			Tensor orgImage = Lib.GetTensorFromImage(image).to(dtype, device);

			// Change RGB → BGR
			orgImage = torch.stack(new Tensor[] { orgImage[2], orgImage[1], orgImage[0] }, dim: 0).unsqueeze(0) / 255.0f;

			int w = (int)orgImage.shape[3];
			int h = (int)orgImage.shape[2];
			int padHeight = 32 - (int)(orgImage.shape[2] % 32);
			int padWidth = 32 - (int)(orgImage.shape[3] % 32);

			padHeight = padHeight == 32 ? 0 : padHeight;
			padWidth = padWidth == 32 ? 0 : padWidth;

			Tensor input = torch.nn.functional.pad(orgImage, new long[] { 0, padWidth, 0, padHeight }, PaddingModes.Zeros);
			Tensor[] tensors = yolo.forward(input);
			(List<Tensor> nms_result, var _) = Ops.non_max_suppression(tensors[0], nc: sortCount, iou_thres: IouThreshold, rotated: true);
			//List<Tensor> nms_result = NonMaxSuppression(tensors[0], nm: 1, iouThreshold: IouThreshold);
			List<YoloResult> results = new List<YoloResult>();
			if (nms_result.Count > 0)
			{
				if (nms_result[0] is not null)
				{
					for (int i = 0; i < nms_result[0].shape[0]; i++)
					{
						YoloResult result = new YoloResult();
						result.CenterX = nms_result[0][i][0].ToInt32();
						result.CenterY = nms_result[0][i][1].ToInt32();
						result.Width = nms_result[0][i][2].ToInt32();
						result.Height = nms_result[0][i][3].ToInt32();
						result.Radian = nms_result[0][i][6].ToSingle();
						result.ClassID = nms_result[0][i][5].ToInt32();
						result.Score = nms_result[0][i][4].ToSingle();
						results.Add(result);
					}
				}
			}

			return results;
		}

		public void Train()
		{

		}

	}
}
