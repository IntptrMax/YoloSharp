using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;

namespace YoloSharp.ModelLoader.PtReader
{
    public sealed class YoloPtLoader
    {
        public sealed class LoadResult
        {
            public Dictionary<string, TorchTensor> Tensors { get; set; } = new Dictionary<string, TorchTensor>();
            public Dictionary<string, byte[]> RawStorages { get; set; } = new Dictionary<string, byte[]>();
            public Dictionary<string, object?> TopLevel { get; set; } = new Dictionary<string, object?>();
        }

        public LoadResult LoadFromFile(string ptPath)
        {
            using var zip = ZipFile.OpenRead(ptPath);

            // 1) data.pkl（一般在 archive/data.pkl）
            var pklEntry = zip.Entries.FirstOrDefault(e =>
                e.FullName.EndsWith("data.pkl", StringComparison.OrdinalIgnoreCase))
                ?? throw new FileNotFoundException($"data.pkl not found in {ptPath}");

            string dataPrefix = pklEntry.FullName.Substring(
                0, pklEntry.FullName.Length - "data.pkl".Length) + "data/";

            // 2) 解析 pickle
            byte[] pklBytes;
            using (var s = pklEntry.Open())
            using (var ms = new MemoryStream())
            { s.CopyTo(ms); pklBytes = ms.ToArray(); }

            var root = new TorchPickleReader().Load(pklBytes);
            var topLevel = root as Dictionary<string, object?> ?? new Dictionary<string, object?>();

            // 3) 从 "model" 分支收集 tensor（rootName 传 ""，得到 "model.0.conv.weight" 这种键）
            var tensors = new Dictionary<string, TorchTensor>();
            if (topLevel.TryGetValue("model", out var modelObj) && modelObj != null)
                foreach (var kv in ModelWalker.CollectTensors(modelObj, ""))
                    tensors[kv.Key] = kv.Value;

            // 4) 读取涉及到的 storage 二进制
            var rawStorages = new Dictionary<string, byte[]>();
            foreach (var key in tensors.Values.Select(t => t.Storage.Key).Distinct())
            {
                var entry = zip.GetEntry(dataPrefix + key)
                    ?? throw new FileNotFoundException($"storage entry not found: {dataPrefix + key}");
                using var s = entry.Open();
                using var ms = new MemoryStream();
                s.CopyTo(ms);
                rawStorages[key] = ms.ToArray();
            }

            return new LoadResult
            {
                Tensors = tensors,
                RawStorages = rawStorages,
                TopLevel = topLevel
            };
        }
    }
}