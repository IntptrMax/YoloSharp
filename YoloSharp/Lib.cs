using ImageMagick;
using System.Diagnostics;
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

		/// <summary>
		/// Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.
		/// </summary>
		/// <param name="x">The bounding boxes to clip</param>
		/// <param name="shape">The shape of the image</param>
		/// <returns>The clipped boxes</returns>
		internal static Tensor ClipBox(Tensor x, float[] shape)
		{
			//	using var _ = NewDisposeScope();
			Tensor box = torch.zeros_like(x);
			box[TensorIndex.Ellipsis, 0] = x[TensorIndex.Ellipsis, 0].clamp_(0, shape[1]);  // x1
			box[TensorIndex.Ellipsis, 1] = x[TensorIndex.Ellipsis, 1].clamp_(0, shape[0]);  // y1
			box[TensorIndex.Ellipsis, 2] = x[TensorIndex.Ellipsis, 2].clamp_(0, shape[1]);  // x2
			box[TensorIndex.Ellipsis, 3] = x[TensorIndex.Ellipsis, 3].clamp_(0, shape[0]);  // y2
			return box.MoveToOuterDisposeScope();
		}

		internal static Tensor GetTensorFromImage(MagickImage image)
		{
			using (MemoryStream memoryStream = new MemoryStream())
			{
				image.Write(memoryStream, MagickFormat.Png);
				memoryStream.Position = 0;
				Tensor result = torchvision.io.read_image(memoryStream, torchvision.io.ImageReadMode.RGB);
				return result;
			}
		}

		internal static MagickImage GetImageFromTensor(Tensor tensor)
		{
			MemoryStream memoryStream = new MemoryStream();
			torchvision.io.write_png(tensor.cpu(), memoryStream);
			memoryStream.Position = 0;
			return new MagickImage(memoryStream, MagickFormat.Png);
		}

	}
}
