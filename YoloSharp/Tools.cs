using System.IO.Compression;
using System.Text;
using TorchSharp;
using YoloSharp.ModelLoader;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace YoloSharp
{
	internal class Tools
	{
		/// <summary>
		/// Load weight from safetensors and write it to disk.
		/// </summary>
		/// <param name="model">Model in TorchSharp</param>
		/// <param name="safetensorName">safetensors file</param>
		/// <param name="outName">file to write on disk</param>
		public static void TransModelFromSafetensors(Module<Tensor, Tensor[]> model, string safetensorName, string outName)
		{
			Dictionary<string, Tensor> dict = new Dictionary<string, Tensor>();
			SafetensorsLoader safetensorLoader = new SafetensorsLoader();
			var safetensorTensors = safetensorLoader.ReadTensorsInfoFromFile(safetensorName);

			foreach (var li in safetensorTensors)
			{
				Tensor t = torch.zeros(li.Shape.ToArray(), dtype: li.Type);
				byte[] dt = safetensorLoader.ReadByteFromFile(li);
				t.bytes = dt;
				dict.Add(li.Name, t);
			}
			Dictionary<string, Tensor> state_dict = model.state_dict();
			var (loadMissing, unexp) = model.load_state_dict(dict, false);
			model.save(outName);
		}

		/// <summary>
		/// Load Python .pt tensor file
		/// </summary>
		/// <param name="path">tensor path</param>
		/// <returns>Tensor in TorchSharp</returns>
		public static Tensor LoadTensorFromPT(string path)
		{
			torch.ScalarType dtype = torch.ScalarType.Float32;
			List<long> shape = new List<long>();
			ZipArchive zip = ZipFile.OpenRead(path);
			ZipArchiveEntry headerEntry = zip.Entries.First(e => e.Name == "data.pkl");

			// Header is always small enough to fit in memory, so we can read it all at once
			using Stream headerStream = headerEntry.Open();
			byte[] headerBytes = new byte[headerEntry.Length];
			headerStream.Read(headerBytes, 0, headerBytes.Length);

			string headerStr = Encoding.Default.GetString(headerBytes);
			if (headerStr.Contains("HalfStorage"))
			{
				dtype = torch.ScalarType.Float16;
			}
			else if (headerStr.Contains("BFloat"))
			{
				dtype = torch.ScalarType.Float16;
			}
			else if (headerStr.Contains("FloatStorage"))
			{
				dtype = torch.ScalarType.Float32;
			}
			for (int i = 0; i < headerBytes.Length; i++)
			{
				if (headerBytes[i] == 81 && headerBytes[i + 1] == 75 && headerBytes[i + 2] == 0)
				{
					for (int j = i + 2; j < headerBytes.Length; j++)
					{
						if (headerBytes[j] == 75)
						{
							shape.Add(headerBytes[j + 1]);
							j++;
						}
						else if (headerBytes[j] == 77)
						{
							shape.Add(headerBytes[j + 1] + headerBytes[j + 2] * 256);
							j += 2;
						}
						else if (headerBytes[j] == 113)
						{
							break;
						}

					}
					break;
				}
			}

			Tensor tensor = torch.zeros(shape.ToArray(), dtype: dtype);
			ZipArchiveEntry dataEntry = zip.Entries.First(e => e.Name == "0");

			using Stream dataStream = dataEntry.Open();
			byte[] data = new byte[dataEntry.Length];
			dataStream.Read(data, 0, data.Length);
			tensor.bytes = data;
			return tensor;
		}

		/// <summary>
		/// Load Python .pt tensor file and change dtype and device the same as given tensor.
		/// </summary>
		/// <param name="path">tensor path</param>
		/// <param name="tensor">the given tensor</param>
		/// <returns>Tensor in TorchSharp</returns>
		public static Tensor LoadTensorFromPT(string path, Tensor tensor)
		{
			return LoadTensorFromPT(path).to(tensor.dtype, tensor.device);
		}
	}
}
