using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace YoloSharp.Utils
{
    /// <summary>
    /// [Add 2026-11-04] Lightweight .bin (platform training-weight format) inspector: reads ONLY
    /// tensor key names / shapes without materializing tensors — lets ModelConverter detect the
    /// architecture (v9 head at model.29 vs v8 at model.22) and nc before building the skeleton.
    /// Layout (mirrors Lib.SaveModel): varint tensorCount, then per tensor:
    /// string name | varint dtype | varint dimCount | varint dim[] | raw data (shape×elemsize).
    /// 轻量bin检查器：只读键名与形状（不实例化张量）。字节布局与Lib.SaveModel镜像。
    /// </summary>
    public static class LibPeeker
    {
        static long Decode(BinaryReader reader)
        {
            long num = 0L;
            int num2 = 0;
            while (true)
            {
                long num3 = reader.ReadByte();
                num += (num3 & 0x7F) << num2 * 7;
                if ((num3 & 0x80) == 0L) break;
                num2++;
            }
            return num;
        }

        static int ElementSize(long dtype)
        {
            // torch.ScalarType: 0=uint8 1=int8 2=int16 3=int32 4=int64 5=fp16 6=fp32 7=fp64 8=bf16(varies by build)
            switch (dtype)
            {
                case 0: case 1: return 1;
                case 2: return 2;
                case 3: case 5: case 8: return 4;
                case 4: return 8;
                case 6: return 4;
                case 7: return 8;
                default: return 4;
            }
        }

        /// <summary>Read up to maxKeys tensor names (header pass only). / 只读前maxKeys个键名。</summary>
        public static List<string> PeekKeyNames(string path, int maxKeys)
        {
            var names = new List<string>();
            using FileStream input = File.OpenRead(path);
            using BinaryReader reader = new BinaryReader(input);
            long tensorCount = Decode(reader);
            for (int i = 0; i < tensorCount && names.Count < maxKeys; i++)
            {
                string tensorName = reader.ReadString();
                names.Add(tensorName);
                long dtype = Decode(reader);
                long dimCount = Decode(reader);
                long elems = 1;
                for (int d = 0; d < dimCount; d++) elems *= Decode(reader);
                // skip raw data: BF16/FP16 stored as 2-byte elements; others via native size
                int es = dtype == 5 || dtype == 8 ? 2 : ElementSize(dtype);
                input.Seek(elems * es, SeekOrigin.Current);
            }
            return names;
        }

        /// <summary>Read shapes of the named tensors (single pass). / 单遍读取指定键的形状。</summary>
        public static Dictionary<string, long[]> PeekShapes(string path, string[] wanted)
        {
            var result = new Dictionary<string, long[]>();
            var want = new HashSet<string>(wanted);
            using FileStream input = File.OpenRead(path);
            using BinaryReader reader = new BinaryReader(input);
            long tensorCount = Decode(reader);
            for (int i = 0; i < tensorCount; i++)
            {
                string tensorName = reader.ReadString();
                long dtype = Decode(reader);
                long dimCount = Decode(reader);
                var dims = new long[dimCount];
                long elems = 1;
                for (int d = 0; d < dimCount; d++) { dims[d] = Decode(reader); elems *= Math.Max(1, dims[d]); }
                if (want.Contains(tensorName)) result[tensorName] = dims;
                int es = dtype == 5 || dtype == 8 ? 2 : ElementSize(dtype);
                input.Seek(elems * es, SeekOrigin.Current);
            }
            return result;
        }

        /// <summary>
        /// [Add 2026-11-04] Full-load probe via Lib (authoritative parser) — returns key names and
        /// per-key shapes for architecture detection. Bridges internal Lib to Txm.AI.
        /// 经Lib全量加载探测（权威解析）：返回键名与形状供架构识别（桥接internal Lib给Txm.AI）。
        /// </summary>
        public static Dictionary<string, long[]> ProbeAll(string path)
        {
            var dict = Lib.LoadModel(path);
            var r = new Dictionary<string, long[]>();
            foreach (var kv in dict) r[kv.Key] = kv.Value.shape;
            return r;
        }

        /// <summary>
        /// [Add 2026-11-08] Infer YoloSize from probed weights: first conv's out-channels follows
        /// the width_multiple table (16=n 0.25x, 32=s 0.5x, 64=m 1.0x on v8-family). A .bin carries
        /// no scale metadata — without this a s-trained bin rebuilds as an n skeleton on predict
        /// (96/357 transferred, garbage boxes).
        /// 按首层卷积宽度推断规模：bin无规模元数据——s训练的bin在推理侧会重建为n骨架。
        /// v8系宽度表：首层conv输出16=n/32=s/64=m（width 0.25/0.5/1.0）。
        /// </summary>
        public static int InferFirstConvWidth(Dictionary<string, long[]> probe)
        {
            // 首层键形如 model.0.conv.weight [out,in,k,k]（各架构同构）
            var k = probe.Keys.FirstOrDefault(x =>
                x.StartsWith("model.0.conv.weight") || x.StartsWith("model.0.0.conv.weight"));
            if (k != null && probe.TryGetValue(k, out var sh) && sh.Length == 4) return (int)sh[0];
            return -1;
        }
    }
}
