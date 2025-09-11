using SkiaSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace YoloSharp
{
	internal class Lib
	{
		internal static Dictionary<string, Tensor> LoadModel(string path, bool skipNcNotEqualLayers = false)
		{
			long Decode(BinaryReader reader)
			{
				long num = 0L;
				int num2 = 0;
				while (true)
				{
					long num3 = reader.ReadByte();
					num += (num3 & 0x7F) << num2 * 7;
					if ((num3 & 0x80) == 0L)
					{
						break;
					}
					num2++;
				}

				return num;
			}

			using FileStream input = File.OpenRead(path);
			using BinaryReader reader = new BinaryReader(input);
			long tensorCount = Decode(reader);

			Dictionary<string, Tensor> state_dict = new Dictionary<string, Tensor>();
			torch.ScalarType modelType = torch.ScalarType.Float32;
			for (int i = 0; i < tensorCount; i++)
			{
				string tensorName = reader.ReadString();
				torch.ScalarType dtype = (torch.ScalarType)Decode(reader);
				if (i == 0)
				{
					modelType = dtype;
				}
				long length = Decode(reader);
				long[] shape = new long[length];
				for (int j = 0; j < length; j++)
				{
					shape[j] = Decode(reader);
				}
				Tensor tensor = torch.zeros(shape, dtype: dtype);
				tensor.ReadBytesFromStream(reader.BaseStream);
				state_dict.Add(tensorName, tensor);
			}
			return state_dict;
		}

		

		internal static Tensor GetTensorFromImage(SKBitmap skBitmap)
		{
			using (MemoryStream stream = new MemoryStream())
			{
				skBitmap.Encode(stream, SKEncodedImageFormat.Png, 100);
				stream.Position = 0;
				Tensor tensor = torchvision.io.read_image(stream, torchvision.io.ImageReadMode.RGB);
				return tensor;
			}
		}

		internal static SKBitmap GetImageFromTensor(Tensor tensor)
		{
			using (MemoryStream memoryStream = new MemoryStream())
			{
				torchvision.io.write_png(tensor.cpu(), memoryStream);
				memoryStream.Position = 0;
				SKBitmap skBitmap = SKBitmap.Decode(memoryStream);
				return skBitmap;
			}
		}

	}
}
