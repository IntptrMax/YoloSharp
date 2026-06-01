using Newtonsoft.Json.Linq;
using System.Text;
using TorchSharp;

namespace YoloSharp.ModelLoader
{
    internal class SafetensorsLoader
    {
        public List<CommonTensor> ReadTensorsInfoFromFile(string inputFileName)
        {
            using (FileStream stream = File.OpenRead(inputFileName))
            {
                long len = stream.Length;
                if (len < 10)
                {
                    throw new ArgumentOutOfRangeException("File cannot be valid safetensors: too short");
                }

                // Safetensors file first 8 byte to int64 is the header length
                byte[] headerBlock = new byte[8];
                stream.Read(headerBlock, 0, 8);
                long headerSize = BitConverter.ToInt64(headerBlock, 0);
                if (len < 8 + headerSize || headerSize <= 0 || headerSize > 100_000_000)
                {
                    throw new ArgumentOutOfRangeException($"File cannot be valid safetensors: header len wrong, size:{headerSize}");
                }

                // Read the header, header file is a json file
                byte[] headerBytes = new byte[headerSize];
                stream.Read(headerBytes, 0, (int)headerSize);

                string header = Encoding.UTF8.GetString(headerBytes);
                long bodyPosition = stream.Position;
                JToken token = JToken.Parse(header);
                List<CommonTensor> tensors = new List<CommonTensor>();

                foreach (KeyValuePair<string, JToken> subTokens in token.ToObject<Dictionary<string, JToken>>())
                {
                    JToken subToken = subTokens.Value;
                    string key = subTokens.Key;
                    Dictionary<string, JToken> value = subToken.ToObject<Dictionary<string, JToken>>();
                    value.TryGetValue("data_offsets", out JToken offsets);
                    value.TryGetValue("dtype", out JToken dtype);
                    value.TryGetValue("shape", out JToken shape);

                    ulong[] offsetArray = offsets?.ToObject<ulong[]>();
                    if (null == offsetArray)
                    {
                        continue;
                    }
                    long[] shapeArray = shape.ToObject<long[]>();

                    torch.ScalarType scalar_type = torch.ScalarType.Float32;
                    switch (dtype.ToString())
                    {
                        case "I8": scalar_type = torch.ScalarType.Int8; break;
                        case "I16": scalar_type = torch.ScalarType.Int16; break;
                        case "I32": scalar_type = torch.ScalarType.Int32; break;
                        case "I64": scalar_type = torch.ScalarType.Int64; break;
                        case "BF16": scalar_type = torch.ScalarType.BFloat16; break;
                        case "F16": scalar_type = torch.ScalarType.Float16; break;
                        case "F32": scalar_type = torch.ScalarType.Float32; break;
                        case "F64": scalar_type = torch.ScalarType.Float64; break;
                        case "U8": scalar_type = torch.ScalarType.Byte; break;
                        case "U16":
                        case "U32":
                        case "U64":
                        case "BOOL":
                        case "F8_E4M3":
                        case "F8_E5M2": break;
                    }

                    CommonTensor tensor = new CommonTensor
                    {
                        Name = key,
                        Type = scalar_type,
                        Shape = shapeArray.ToList(),
                        Offset = offsetArray.ToList(),
                        FileName = inputFileName,
                        BodyPosition = bodyPosition
                    };

                    tensors.Add(tensor);
                }
                return tensors;
            }
        }

        private byte[] ReadByteFromFile(string inputFileName, long bodyPosition, long offset, int size)
        {
            using (FileStream stream = File.OpenRead(inputFileName))
            {
                stream.Seek(bodyPosition + offset, SeekOrigin.Begin);
                byte[] dest = new byte[size];
                stream.Read(dest, 0, size);
                return dest;
            }
        }

        public byte[] ReadByteFromFile(CommonTensor tensor)
        {
            string inputFileName = tensor.FileName;
            long bodyPosition = tensor.BodyPosition;
            ulong offset = tensor.Offset[0];
            int size = (int)(tensor.Offset[1] - tensor.Offset[0]);
            return ReadByteFromFile(inputFileName, bodyPosition, (long)offset, size);
        }
    }
}
