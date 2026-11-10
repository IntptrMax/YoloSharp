using OpenCvSharp;
using SkiaSharp;
using TorchSharp;

namespace YoloSharp.Utils
{
	internal class Lib
	{
		internal static Dictionary<string, torch.Tensor> LoadModel(string path, bool skipNcNotEqualLayers = false)
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

			Dictionary<string, torch.Tensor> state_dict = new Dictionary<string, torch.Tensor>();
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
				// [Fix 2026-10-28] ReadBytesFromStream for BF16/FP16 on TorchSharp 0.105.2 reads
				// half-precision bytes as 4-byte elements, misaligning the stream (all later tensors corrupted → weights garbled after load).
				// Fix: read half-precision tensors as raw 2-byte data and convert to Float32 manually (BF16 = F32 high 16 bits).
				// 修复：半精度按原始字节手工读入（BF16=F32截断高16位），再转Float32——旧ReadBytesFromStream会按4字节读导致流错位。
				torch.Tensor tensor;
				if (dtype == torch.ScalarType.BFloat16 || dtype == torch.ScalarType.Float16)
				{
					long elemCount = 1;
					foreach (var d in shape) elemCount *= d;
					byte[] raw = reader.ReadBytes((int)(elemCount * 2));
					float[] f32 = new float[elemCount];
					if (dtype == torch.ScalarType.BFloat16)
					{
						// BF16→F32: high 16 bits shifted left by 16 (low bits zero, lossless) / BF16→F32：高16位左移16位（低位补0，无损还原）
						for (long k = 0; k < elemCount; k++)
						{
							int bits = raw[(int)(2 * k + 1)] << 24 | raw[(int)(2 * k)] << 16;
							f32[k] = BitConverter.ToSingle(BitConverter.GetBytes(bits), 0);
						}
					}
					else
					{
						// FP16→F32: expand per IEEE754 half precision / FP16→F32：按IEEE754半精度展开
						for (long k = 0; k < elemCount; k++)
						{
							ushort h = (ushort)(raw[(int)(2 * k + 1)] << 8 | raw[(int)(2 * k)]);
							f32[k] = HalfToFloat(h);
						}
					}
					tensor = torch.tensor(f32, dtype: torch.ScalarType.Float32).reshape(shape);
				}
				else
				{
					tensor = torch.zeros(shape, dtype: dtype);
					tensor.ReadBytesFromStream(reader.BaseStream);
				}
				state_dict.Add(tensorName, tensor);
			}
			return state_dict;
		}

		/// <summary>IEEE754 half precision → single precision (for FP16 bin loading) / IEEE754半精度→单精度（FP16 bin加载用）</summary>
		internal static float HalfToFloat(ushort h)
		{
			uint sign = (uint)(h >> 15) << 31;
			uint exp = (uint)((h >> 10) & 0x1F);
			uint frac = (uint)(h & 0x3FF);
			uint bits;
			if (exp == 0)
			{
				if (frac == 0) bits = sign;
				else
				{
					// subnormal: normalize / 次正规数：规格化
					exp = (uint)(127 - 15 + 1);
					while ((frac & 0x400) == 0) { frac <<= 1; exp--; }
					frac &= 0x3FF;
					bits = sign | (exp << 23) | (frac << 13);
				}
			}
			else if (exp == 0x1F)
			{
				bits = sign | 0x7F800000 | (frac << 13);   // Inf/NaN
			}
			else
			{
				bits = sign | ((exp - 15 + 127) << 23) | (frac << 13);
			}
			return BitConverter.ToSingle(BitConverter.GetBytes((int)bits), 0);
		}



		internal static torch.Tensor GetTensorFromImage(SKBitmap skBitmap, torchvision.io.ImageReadMode readMode = torchvision.io.ImageReadMode.RGB)
		{
			using (MemoryStream stream = new MemoryStream())
			{
				skBitmap.Encode(stream, SKEncodedImageFormat.Png, 100);
				stream.Position = 0;
                torch.Tensor tensor = torchvision.io.read_image(stream, readMode);
				return tensor;
			}
		}

		internal static torch.Tensor GetTensorFromImage(string imagePath, torchvision.io.ImageReadMode readMode = torchvision.io.ImageReadMode.RGB)
		{
			using (FileStream stream = new FileStream(imagePath, FileMode.Open, FileAccess.Read))
			{
                torch.Tensor tensor = torchvision.io.read_image(stream, readMode);
				return tensor;
			}
		}

		internal static torch.Tensor GetTensorFromImage(Mat mat, torchvision.io.ImageReadMode readMode = torchvision.io.ImageReadMode.RGB)
		{
            torch.Tensor tensor = torchvision.io.read_image(mat.ToMemoryStream(), readMode);
			return tensor;
		}

		internal static SKBitmap GetSKBitmapFromTensor(torch.Tensor tensor)
		{
			using (MemoryStream memoryStream = new MemoryStream())
			{
				torchvision.io.write_png(tensor.cpu(), memoryStream);
				memoryStream.Position = 0;
				SKBitmap skBitmap = SKBitmap.Decode(memoryStream);
				return skBitmap;
			}
		}

		internal static Mat GetMatFromTensor(torch.Tensor tensor)
		{
			using (MemoryStream memoryStream = new MemoryStream())
			{
				long channelCount = tensor.ndim == 4 ? tensor.shape[1] : tensor.shape[0];
				torchvision.io.write_png(tensor.cpu(), memoryStream);
				memoryStream.Position = 0;
				Mat mat = Mat.FromStream(memoryStream, channelCount == 1 ? ImreadModes.Grayscale : ImreadModes.Color);
				return mat;
			}
		}


	}
}
