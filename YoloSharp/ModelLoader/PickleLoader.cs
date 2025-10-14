using System.Collections.ObjectModel;
using System.IO.Compression;
using TorchSharp;

namespace YoloSharp.ModelLoader
{
	public class PickleLoader
	{
		private ZipArchive zip;
		private ReadOnlyCollection<ZipArchiveEntry> entries;

		public List<CommonTensor> ReadTensorsInfoFromFile(string fileName)
		{
			List<CommonTensor> tensors = new List<CommonTensor>();

			zip = ZipFile.OpenRead(fileName);
			entries = zip.Entries;
			ZipArchiveEntry headerEntry = entries.First(e => e.Name == "data.pkl");
			byte[] headerBytes = new byte[headerEntry.Length];
			// Header is always small enough to fit in memory, so we can read it all at once
			using (Stream stream = headerEntry.Open())
			{
				stream.Read(headerBytes, 0, headerBytes.Length);
			}

			if (headerBytes[0] != 0x80 || headerBytes[1] != 0x02)
			{
				throw new ArgumentException("Not a valid pickle file");
			}

			int index = 1;
			bool finished = false;
			bool readStrides = false;
			bool binPersid = false;

			CommonTensor tensor = new CommonTensor() { FileName = fileName, Offset = { 0 } };

			int deepth = 0;

			Dictionary<int, string> BinPut = new Dictionary<int, string>();

			while (index < headerBytes.Length && !finished)
			{
				byte opcode = headerBytes[index];
				switch (opcode)
				{
					case (byte)'}':  // EMPTY_DICT     = b'}'   # push empty dict
						break;
					case (byte)']':  // EMPTY_LIST     = b']'   # push empty list
						break;
					// skip unused sections
					case (byte)'h':  // BINGET         = b'h'   #   "    "    "    "   "   "  ;   "    " 1-byte arg
						{
							int id = headerBytes[index + 1];
							BinPut.TryGetValue(id, out string precision);
							if (precision != null)
							{
								if (precision.Contains("FloatStorage"))
								{
									tensor.Type = TorchSharp.torch.ScalarType.Float32;
								}
								else if (precision.Contains("HalfStorage"))
								{
									tensor.Type = TorchSharp.torch.ScalarType.Float16;
								}
								else if (precision.Contains("BFloat16Storage"))
								{
									tensor.Type = TorchSharp.torch.ScalarType.BFloat16;
								}
							}
							index++;
							break;
						}
					case (byte)'q':  // BINPUT         = b'to_q'   #   "     "    "   "   " ;   "    " 1-byte arg
						{
							index++;
							break;
						}
					case (byte)'Q':  // BINPERSID      = b'Q'   #  "       "         "  ;  "  "   "     "  stack
						binPersid = true;
						break;
					case (byte)'r':  // LONG_BINPUT    = b'r'   #   "     "    "   "   " ;   "    " 4-byte arg
						index += 4;
						break;
					case 0x95:  // FRAME            = b'\x95'  # indicate the beginning of a new frame
						index += 8;
						break;
					case 0x94:  // MEMOIZE          = b'\x94'  # store top of the stack in memo
						break;
					case (byte)'(':  // MARK           = b'('   # push special markobject on stack
						deepth++;
						break;
					case (byte)'K':  // BININT1        = b'K'   # push 1-byte unsigned int
						{
							int value = headerBytes[index + 1];
							index++;

							if (deepth > 1 && value != 0 && binPersid)
							{
								if (readStrides)
								{
									//tensor.Stride.Add((ulong)value);
									tensor.Stride.Add((ulong)value);
								}
								else
								{
									tensor.Shape.Add(value);
								}
							}
						}
						break;
					case (byte)'M':  // BININT2        = b'M'   # push 2-byte unsigned int
						{
							UInt16 value = BitConverter.ToUInt16(headerBytes, index + 1);
							index += 2;

							if (deepth > 1 && value != 0 && binPersid)
							{
								if (readStrides)
								{
									tensor.Stride.Add(value);
								}
								else
								{
									tensor.Shape.Add(value);
								}
							}

						}
						break;
					case (byte)'J':  // BININT         = b'J'   # push four-byte signed int
						{
							int value = BitConverter.ToInt32(headerBytes, index + 1);
							//int value = headerBytes[index + 4] << 24 + headerBytes[index + 3] << 16 + headerBytes[index + 2] << 8 + headerBytes[index + 1];
							index += 4;

							if (deepth > 1 && value != 0 && binPersid)
							{
								if (readStrides)
								{
									tensor.Stride.Add((ulong)value);
								}
								else
								{
									tensor.Shape.Add(value);
								}
							}
						}
						break;

					case (byte)'X':  // BINUNICODE     = b'X'   #   "     "       "  ; counted UTF-8 string argument
						{
							int length = headerBytes[index + 1];
							int start = index + 5;
							byte module = headerBytes[index + 1];
							string name = System.Text.Encoding.UTF8.GetString(headerBytes, start, length);
							index = index + 4 + length;

							if (deepth == 1)
							{
								tensor.Name = name;
							}
							else if (deepth == 3)
							{
								if ("cpu" != name && !name.Contains("cuda"))
								{
									tensor.DataNameInZipFile = name;
								}
							}
						}
						break;
					case 0x8C:  // SHORT_BINUNICODE = b'\x8c'  # push short string; UTF-8 length < 256 bytes
						{

						}
						break;
					case (byte)'c':  // GLOBAL         = b'c'   # push self.find_class(modname, name); 2 string args
						{
							int start = index + 1;
							while (headerBytes[index + 1] != (byte)'q')
							{
								index++;
							}
							int length = index - start + 1;

							string global = System.Text.Encoding.UTF8.GetString(headerBytes, start, length);

							// precision is stored in the global variable
							// next tensor will read the precision
							// so we can set the Type here

							BinPut.Add(headerBytes[index + 2], global);

							if (global.Contains("FloatStorage"))
							{
								tensor.Type = TorchSharp.torch.ScalarType.Float32;
							}
							else if (global.Contains("HalfStorage"))
							{
								tensor.Type = TorchSharp.torch.ScalarType.Float16;
							}
							else if (global.Contains("BFloat16Storage"))
							{
								tensor.Type = TorchSharp.torch.ScalarType.BFloat16;
							}
							break;
						}
					case 0x86:  // TUPLE2         = b'\x86'  # build 2-tuple from two topmost stack items
						{
							if (binPersid)
							{
								readStrides = true;
							}
							break;
						}
					case 0x85:  // TUPLE1         = b'\x85'  # build 1-tuple from stack top
						if (binPersid)
						{
							readStrides = true;
						}
						break;
					case (byte)'t':   // TUPLE          = b't'   # build tuple from topmost stack items
						deepth--;
						if (binPersid)
						{
							readStrides = true;
						}
						break;
					case (byte)'R': // REDUCE         = b'R'   # apply callable to argtuple, both on stack
						if (deepth == 1)
						{
							if (tensor.Name.Contains("metadata"))
							{
								break;
							}

							if (string.IsNullOrEmpty(tensor.DataNameInZipFile))
							{
								tensor.DataNameInZipFile = tensors.Last().DataNameInZipFile;
								tensor.Offset = new List<ulong> { (ulong)(tensor.Shape[0] * tensor.Type.ElementSize()) };
								tensor.Shape.RemoveAt(0);
								//tensor.offset = tensors.Last().
							}
							tensors.Add(tensor);

							tensor = new CommonTensor() { FileName = fileName, Offset = { 0 } };
							readStrides = false;
							binPersid = false;
						}
						break;
					case (byte)'.':  // STOP           = b'.'   # every pickle ends with STOP
						finished = true;
						break;
					default:
						break;
				}
				index++;
			}
			CommonTensor metaTensor = tensors.Find(x => x.Name.Contains("_metadata"));
			if (metaTensor != null)
			{
				tensors.Remove(metaTensor);
			}
			return tensors;
		}

		private byte[] ReadByteFromFile(CommonTensor tensor)
		{
			if (entries is null)
			{
				throw new ArgumentNullException(nameof(entries));
			}

			ZipArchiveEntry dataEntry = entries.First(e => e.Name == tensor.DataNameInZipFile);
			long i = 1;
			foreach (var ne in tensor.Shape)
			{
				i *= ne;
			}
			ulong length = (ulong)(tensor.Type.ElementSize() * i);
			byte[] data = new byte[dataEntry.Length];

			using (Stream stream = dataEntry.Open())
			{
				stream.Read(data, 0, data.Length);
			}

			//data = data.Take(new Range((int)tensor.Offset[0], (int)(tensor.Offset[0] + length))).ToArray();
			byte[] result = new byte[length];
			for (int j = 0; j < (int)length; j++)
			{
				result[j] = data[j + (int)tensor.Offset[0]];
			}
			return result;
			//return data;
		}

		internal Dictionary<string, torch.Tensor> Load(string fileName, string addString = "")
		{
			Dictionary<string, torch.Tensor> tensors = new Dictionary<string, torch.Tensor>();
			List<CommonTensor> tensorInfos = ReadTensorsInfoFromFile(fileName);
			foreach (CommonTensor tensorInfo in tensorInfos)
			{
				torch.Tensor tensor = torch.empty(tensorInfo.Shape.ToArray(), dtype: tensorInfo.Type);
				tensor.bytes = ReadByteFromFile(tensorInfo);
				tensors.Add(addString + tensorInfo.Name, tensor);
			}
			return tensors;
		}
	}
}
