using YoloSharp.Types;
using static TorchSharp.torch;

namespace YoloSharp.Models
{
	internal interface YoloBaseTaskModel
	{
		public void LoadModel(string path, bool skipNcNotEqualLayers = false);

		public void Train(string rootPath, string trainDataPath = "", string valDataPath = "", string outputPath = "output", int imageSize = 640, int epochs = 100, float lr = 0.0001f, int batchSize = 8, int numWorkers = 0, ImageProcessType imageProcessType = ImageProcessType.Letterbox);

		public List<YoloResult> ImagePredict(Tensor orgImage, float PredictThreshold = 0.25f, float IouThreshold = 0.5f);

	}
}
