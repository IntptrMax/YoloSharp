using TorchSharp;
using static TorchSharp.torch;

namespace YoloSharp.ModelLoader.PtReader
{
    public static class TensorBuilder
    {
        public static Tensor Build(TorchTensor t, byte[] storageBytes)
        {
            long elemSize = t.Storage.ElementSize;
            long byteOffset = t.StorageOffset * elemSize;
            long byteLength = t.ElementCount * elemSize;

            if (byteOffset + byteLength > storageBytes.Length)
                throw new InvalidDataException(
                    $"Storage overflow: need {byteOffset + byteLength}, got {storageBytes.Length}");

            return t.Storage.TypeName switch
            {
                "HalfStorage" => BuildHalf(storageBytes, byteOffset, t.Size),
                "FloatStorage" => BuildFloat(storageBytes, byteOffset, t.Size),
                "DoubleStorage" => BuildDouble(storageBytes, byteOffset, t.Size),
                "LongStorage" => BuildLong(storageBytes, byteOffset, t.Size),
                "IntStorage" => BuildInt(storageBytes, byteOffset, t.Size),
                "ShortStorage" => BuildShort(storageBytes, byteOffset, t.Size),
                "ByteStorage" => BuildByte(storageBytes, byteOffset, t.Size),
                "BoolStorage" => BuildBool(storageBytes, byteOffset, t.Size),
                _ => throw new NotSupportedException(t.Storage.TypeName)
            };
        }

        /// <summary>stripModelPrefix: 若 PyTorch state_dict 键形如 "model.0.conv.weight"，
        /// 而 TorchSharp 模型顶层已经叫 "model"，传 false 即可；否则传 true 去掉前缀。</summary>
        public static Dictionary<string, Tensor> BuildStateDict(
            YoloPtLoader.LoadResult result,
            bool stripModelPrefix = false)
        {
            var state = new Dictionary<string, Tensor>();
            foreach (var kv in result.Tensors)
            {
                string key = kv.Key;
                if (stripModelPrefix && key.StartsWith("model."))
                    key = key.Substring("model.".Length);

                var tensor = Build(kv.Value, result.RawStorages[kv.Value.Storage.Key]);
                if (tensor.dtype != ScalarType.Float32)
                    tensor = tensor.to(ScalarType.Float32);
                state[key] = tensor;
            }
            return state;
        }

        private static Tensor BuildHalf(byte[] src, long off, long[] shape)
        {
            long n = 1; foreach (var s in shape) n *= s;
            var data = new float[n];
            for (long i = 0; i < n; i++)
            {
                long o = off + i * 2;
                ushort u = (ushort)(src[o] | (src[o + 1] << 8));
                data[i] = HalfToFloat(u);
            }
            return torch.tensor(data).reshape(shape);
        }
        private static Tensor BuildFloat(byte[] src, long off, long[] shape)
        {
            long n = 1; foreach (var s in shape) n *= s;
            var data = new float[n];
            Buffer.BlockCopy(src, (int)off, data, 0, (int)(n * 4));
            return torch.tensor(data).reshape(shape);
        }
        private static Tensor BuildDouble(byte[] src, long off, long[] shape)
        {
            long n = 1; foreach (var s in shape) n *= s;
            var data = new double[n];
            Buffer.BlockCopy(src, (int)off, data, 0, (int)(n * 8));
            return torch.tensor(data).reshape(shape);
        }
        private static Tensor BuildLong(byte[] src, long off, long[] shape)
        {
            long n = 1; foreach (var s in shape) n *= s;
            var data = new long[n];
            Buffer.BlockCopy(src, (int)off, data, 0, (int)(n * 8));
            return torch.tensor(data).reshape(shape);
        }
        private static Tensor BuildInt(byte[] src, long off, long[] shape)
        {
            long n = 1; foreach (var s in shape) n *= s;
            var data = new int[n];
            Buffer.BlockCopy(src, (int)off, data, 0, (int)(n * 4));
            return torch.tensor(data).reshape(shape);
        }
        private static Tensor BuildShort(byte[] src, long off, long[] shape)
        {
            long n = 1; foreach (var s in shape) n *= s;
            var data = new short[n];
            Buffer.BlockCopy(src, (int)off, data, 0, (int)(n * 2));
            return torch.tensor(data).reshape(shape);
        }
        private static Tensor BuildByte(byte[] src, long off, long[] shape)
        {
            long n = 1; foreach (var s in shape) n *= s;
            var data = new byte[n];
            Array.Copy(src, off, data, 0, n);
            return torch.tensor(data).reshape(shape);
        }
        private static Tensor BuildBool(byte[] src, long off, long[] shape)
        {
            long n = 1; foreach (var s in shape) n *= s;
            var data = new bool[n];
            for (long i = 0; i < n; i++) data[i] = src[off + i] != 0;
            return torch.tensor(data).reshape(shape);
        }

        // netstandard2.0 兼容：BitConverter.Int32BitsToSingle 是 .NET Core 3.0+ 才有的
        private static float Int32BitsToSingle(int value)
        {
#if NET6_0_OR_GREATER
            return BitConverter.Int32BitsToSingle(value);
#else
            return BitConverter.ToSingle(BitConverter.GetBytes(value), 0);
#endif
        }

        private static float HalfToFloat(ushort bits)
        {
            int sign = (bits >> 15) & 0x1;
            int exp = (bits >> 10) & 0x1F;
            int mant = bits & 0x3FF;
            if (exp == 0)
            {
                if (mant == 0) return sign == 0 ? 0f : -0f;
                float v = mant / 1024f; v *= (float)Math.Pow(2, -14);
                return sign == 0 ? v : -v;
            }
            if (exp == 31)
                return mant == 0 ? (sign == 0 ? float.PositiveInfinity : float.NegativeInfinity) : float.NaN;
            int fbits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
            return Int32BitsToSingle(fbits);
        }
    }
}