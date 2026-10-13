using System;
using System.Collections.Generic;

namespace YoloSharp.ModelLoader.PtReader
{
    /// <summary>pickle 中的 GLOBAL/STACK_GLOBAL 引用，如 torch.HalfStorage</summary>
    public sealed class GlobalRef
    {
        public string Module { get; }
        public string Name { get; }
        public string FullName => $"{Module}.{Name}";
        public GlobalRef(string module, string name) { Module = module; Name = name; }
        public override string ToString() => FullName;
    }

    /// <summary>指向 data/{Key} 文件里的存储块</summary>
    public sealed class StorageRef
    {
        public string TypeName { get; set; } = "";
        public string Key { get; set; } = "0";
        public string Location { get; set; } = "cpu";
        public long Numel { get; set; }

        public int ElementSize => TypeName switch
        {
            "HalfStorage" => 2,
            "FloatStorage" => 4,
            "DoubleStorage" => 8,
            "LongStorage" => 8,
            "IntStorage" => 4,
            "ShortStorage" => 2,
            "ByteStorage" => 1,
            "CharStorage" => 1,
            "BoolStorage" => 1,
            _ => throw new NotSupportedException($"Unknown storage type: {TypeName}")
        };
    }

    /// <summary>一个 torch.Tensor 的元数据</summary>
    public sealed class TorchTensor
    {
        public StorageRef Storage { get; set; } = null!;
        public long StorageOffset { get; set; }
        public long[] Size { get; set; } = Array.Empty<long>();
        public long[] Stride { get; set; } = Array.Empty<long>();
        public bool RequiresGrad { get; set; }

        public long ElementCount
        {
            get { long n = 1; foreach (var s in Size) n *= s; return n; }
        }
    }

    /// <summary>通用 Python 对象占位（Module / Sequential / DetectionModel 等）</summary>
    public sealed class PyObject
    {
        public string ClassName { get; set; } = "";
        public Dictionary<string, object?> State { get; } = new Dictionary<string, object?>();
        public object?[]? Args { get; set; }
        public override string ToString() => $"<PyObject {ClassName}>";
    }
}