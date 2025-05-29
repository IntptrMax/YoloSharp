using System.Text.RegularExpressions;
using TorchSharp;
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

		public class ObbResult
		{
			public int ClassID;
			public float Score;
			public int X11;
			public int Y11;
			public int X12;
			public int Y12;
			public int X21;
			public int Y21;
			public int X22;
			public int Y22;
		}
		public Obber(int sortCount = 80, YoloType yoloType = YoloType.Yolov8, YoloSize yoloSize = YoloSize.n, DeviceType deviceType = DeviceType.CUDA, ScalarType dtype = ScalarType.Float32)
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
					string? layerPattern = yoloType switch
					{
						YoloType.Yolov8 => @"model\.22\.cv3",
						YoloType.Yolov11 => @"model\.23\.cv3",
						_ => null
					};

					if (layerPattern != null)
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
	}
}
