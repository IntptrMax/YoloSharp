// ONNX protobuf minimal writer (pure C#, zero dependency) — hand-written serialization subset per onnx.proto3
// Minimal pure-C# zero-dependency ONNX protobuf writer — hand-written serialization subset per onnx.proto3 / 纯C#零依赖的ONNX protobuf最小写入器
// Covers / 覆盖: ModelProto(ir_version/opset_import), GraphProto, NodeProto, TensorProto(initializer), ValueInfoProto
// Design principle / 设计原则: write-only, only fields needed by the exporter; over-design forbidden / 只写不读、只覆盖导出所需字段，过度设计禁止
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace YoloSharp.OnnxExport
{
    /// <summary>ONNX data types (onnx.proto3 TensorProto.DataType, only the used ones) / ONNX数据类型（只列用到的）</summary>
    internal enum OnnxDataType
    {
        Float = 1,
        Int64 = 7,
    }

    /// <summary>Proto wire-format writer (varint/length-delimited/tag) / proto wire格式写入器（varint/length-delimited/tag）</summary>
    internal sealed class ProtoWriter
    {
        readonly List<byte> _buf = new List<byte>(1 << 16);

        public byte[] ToArray() => _buf.ToArray();
        public int Length => _buf.Count;

        // -- basic wire primitives / 基础wire原语 --
        public void WriteVarint(ulong v)
        {
            while (v >= 0x80) { _buf.Add((byte)(v | 0x80)); v >>= 7; }
            _buf.Add((byte)v);
        }

        void Tag(int fieldNumber, int wireType) => WriteVarint((uint)(fieldNumber << 3 | wireType));

        public void WriteInt64Field(int field, long v) { Tag(field, 0); WriteVarint((ulong)v); }
        public void WriteInt32Field(int field, int v) { Tag(field, 0); WriteVarint((ulong)v); }
        public void WriteStringField(int field, string s)
        {
            Tag(field, 2);
            var b = Encoding.UTF8.GetBytes(s ?? "");
            WriteVarint((uint)b.Length);
            _buf.AddRange(b);
        }
        /// <summary>Nested message: write into child writer first, then embed as a whole (standard practice) / 嵌套消息：先写进子writer再整体嵌入（标准做法）</summary>
        public void WriteMessageField(int field, ProtoWriter child)
        {
            Tag(field, 2);
            WriteVarint((uint)child.Length);
            _buf.AddRange(child._buf);
        }
        /// <summary>Packed repeated int64 (fieldNumber, values) — used by dims/int64_data / packed repeated int64——dims/int64_data用</summary>
        public void WritePackedInt64(int field, IEnumerable<long> values)
        {
            var tmp = new ProtoWriter();
            foreach (var v in values) tmp.WriteVarint((ulong)v);
            Tag(field, 2);
            WriteVarint((uint)tmp.Length);
            _buf.AddRange(tmp._buf);
        }
        public void WritePackedInt32(int field, IEnumerable<int> values)
        {
            var tmp = new ProtoWriter();
            foreach (var v in values) tmp.WriteVarint((ulong)v);
            Tag(field, 2);
            WriteVarint((uint)tmp.Length);
            _buf.AddRange(tmp._buf);
        }
        public void WritePackedFloat(int field, IEnumerable<float> values)
        {
            Tag(field, 2);
            WriteVarint((uint)(values.Count() * 4));
            foreach (var v in values)
            {
                _buf.AddRange(BitConverter.GetBytes(v));
            }
        }
        public void WriteFixed32Field(int field, float v)
        {
            Tag(field, 5);
            _buf.AddRange(BitConverter.GetBytes(v));
        }
        public void WriteRawBytesField(int field, byte[] data)   // raw_data
        {
            Tag(field, 2);
            WriteVarint((uint)data.Length);
            _buf.AddRange(data);
        }
    }

    /// <summary>TensorProto build helper (initializer weights) / TensorProto构建便捷类（initializer权重）</summary>
    internal sealed class OnnxTensor
    {
        public string Name;
        public long[] Dims;
        public OnnxDataType DataType;
        public float[] FloatData;    // one of the two by type / 按类型二选一
        public long[] Int64Data;
        public byte[] RawData;

        public void WriteTo(ProtoWriter parent, int field)
        {
            // TensorProto: dims=1(repeated int64), data_type=2, float_data=4(packed), name=8, raw_data=9
            var t = new ProtoWriter();
            foreach (var d in Dims) t.WriteInt64Field(1, d);
            t.WriteInt64Field(2, (long)DataType);
            if (FloatData != null && FloatData.Length > 0)
                t.WritePackedFloat(4, FloatData);
            if (Int64Data != null && Int64Data.Length > 0)
                t.WritePackedInt64(7, Int64Data);   // TensorProto.int64_data = field 7 (verified by official hex: 3a=7<<3|2) / 官方hex实锤
            if (RawData != null && RawData.Length > 0)
                t.WriteRawBytesField(9, RawData);
            t.WriteStringField(8, Name);
            parent.WriteMessageField(field, t);
        }
    }

    /// <summary>NodeProto build helper / NodeProto构建便捷类</summary>
    internal sealed class OnnxNode
    {
        public string OpType;
        public List<string> Inputs = new List<string>();
        public List<string> Outputs = new List<string>();
        public List<(string key, long i)> AttrInts = new List<(string, long)>();
        public List<(string key, IEnumerable<long> ints)> AttrIntLists = new List<(string, IEnumerable<long>)>();
        public List<(string key, float f)> AttrFloats = new List<(string, float)>();
        public List<(string key, string s)> AttrStrings = new List<(string, string)>();

        public OnnxNode(string opType, params string[] outputs) { OpType = opType; Outputs.AddRange(outputs); }

        public void WriteTo(ProtoWriter parent, int field)
        {
            var n = new ProtoWriter();
            foreach (var i in Inputs) n.WriteStringField(1, i);          // input field=1 (repeated)
            foreach (var o in Outputs) n.WriteStringField(2, o);         // output field=2 (repeated)
            n.WriteStringField(4, OpType);                               // op_type field=4
            // attributes field=5
            foreach (var (k, v) in AttrInts) WriteAttr(n, k, i: v);
            foreach (var (k, v) in AttrIntLists) WriteAttr(n, k, ints: v);
            foreach (var (k, v) in AttrFloats) WriteAttr(n, k, f: v);
            foreach (var (k, v) in AttrStrings) WriteAttr(n, k, s: v);
            parent.WriteMessageField(field, n);
        }

        static void WriteAttr(ProtoWriter n, string key, long? i = null, IEnumerable<long> ints = null, float? f = null, string s = null)
        {
            // AttributeProto: name=1, f=2, i=3, s=4, ints=8, type=20(required: INT=2, INTS=7, FLOAT=1, STRING=3) / type=20必填
            var a = new ProtoWriter();
            a.WriteStringField(1, key);
            int at = 0;
            if (f.HasValue) { a.WriteFixed32Field(2, f.Value); at = 1; }
            if (i.HasValue) { a.WriteInt64Field(3, i.Value); at = 2; }
            if (s != null) { a.WriteStringField(4, s); at = 3; }
            if (ints != null) { a.WritePackedInt64(8, ints); at = 7; }
            a.WriteInt64Field(20, at);   // type (required) / type（必填）
            n.WriteMessageField(5, a);
        }
    }

    /// <summary>ValueInfoProto (graph input/output declaration) / ValueInfoProto（graph输入/输出声明）</summary>
    internal sealed class OnnxValueInfo
    {
        public string Name;
        public long[] Dims;          // includes batch dim; dynamic dim uses -1 / 含batch维；动态维用-1
        public OnnxDataType DataType = OnnxDataType.Float;
        public int? Rank;            // [Add 2026-11-12] used with Dims=null to emit a fully-dynamic ranked shape / Dims=null时的动态秩

        public void WriteTo(ProtoWriter parent, int field)
        {
            var v = new ProtoWriter();
            v.WriteStringField(1, Name);
            if (Dims == null)
            {
                // [Fix 2026-11-12] Dims=null must emit a fully-dynamic RANKED shape, not skip shape:
                // omitting the shape field declares an unknown-rank tensor that Netron/ONNX Runtime
                // read as a rank-0 scalar ("float") — reported upstream as wrong output dims.
                // torch.onnx.export convention: dynamic dims via dim_param (string) entries. / 空Dims输出全动态有秩shape
                var tt0 = new ProtoWriter();
                tt0.WriteInt64Field(1, (long)OnnxDataType.Float);   // elem_type = field 1
                var shape0 = new ProtoWriter();
                string[] names = { "batch", "dim1", "dim2", "dim3", "dim4", "dim5" };
                int rank = Rank ?? 4;
                for (int i = 0; i < rank; i++)
                {
                    var dim = new ProtoWriter();
                    dim.WriteStringField(2, names[Math.Min(i, names.Length - 1)]);   // Dim.dim_param = field 2 (string, symbolic)
                    shape0.WriteMessageField(1, dim);
                }
                tt0.WriteMessageField(2, shape0);                   // tensor_type.shape = field 2
                var type0 = new ProtoWriter();
                type0.WriteMessageField(1, tt0);
                v.WriteMessageField(2, type0);
                parent.WriteMessageField(field, v);
                return;
            }
            var tt = new ProtoWriter();
            tt.WriteInt64Field(1, (long)DataType);               // TypeProto.TensorType.elem_type = field 1
            var shape = new ProtoWriter();
            foreach (var d in Dims)
            {
                var dim = new ProtoWriter();
                dim.WriteInt64Field(1, d);            // Dim.dim_value = field 1 (official hex: 08=1<<3|0)
                shape.WriteMessageField(1, dim);      // TensorShapeProto.dim = field 1 (repeated)
            }
            tt.WriteMessageField(2, shape);                      // tensor_type.shape = field 2
            var type = new ProtoWriter();
            type.WriteMessageField(1, tt);       // TypeProto.tensor_type = field 1
            v.WriteMessageField(2, type);        // ValueInfoProto.type = field 2
            parent.WriteMessageField(field, v);
        }
    }

    /// <summary>GraphProto + ModelProto assembly and file writing / GraphProto + ModelProto组装与落盘</summary>
    internal sealed class OnnxModelWriter
    {
        readonly List<OnnxNode> _nodes = new List<OnnxNode>();
        readonly List<OnnxTensor> _initializers = new List<OnnxTensor>();
        readonly List<OnnxValueInfo> _inputs = new List<OnnxValueInfo>();
        readonly List<OnnxValueInfo> _outputs = new List<OnnxValueInfo>();
        readonly List<string> _names = new List<string>();

        public OnnxNode AddNode(OnnxNode n) { _nodes.Add(n); return n; }
        public void AddInitializer(OnnxTensor t) { _initializers.Add(t); }
        public void AddInput(OnnxValueInfo v) { _inputs.Add(v); }
        public void AddOutput(OnnxValueInfo v) { _outputs.Add(v); }

        int _tmp;
        public string TempName(string hint = "t") => $"{hint}_{_tmp++}";
        string Unique(string base_) { var n = base_; int i = 1; while (_names.Contains(n)) n = base_ + (i++); _names.Add(n); return n; }

        public void Save(string path, string modelName = "yolosharp")
        {
            // GraphProto: node=1, name=2, initializer=5, input=11, output=12
            var graph = new ProtoWriter();
            foreach (var n in _nodes) n.WriteTo(graph, 1);
            graph.WriteStringField(2, Unique("main_graph"));
            foreach (var t in _initializers) t.WriteTo(graph, 5);
            foreach (var i in _inputs) i.WriteTo(graph, 11);
            foreach (var o in _outputs) o.WriteTo(graph, 12);

            // ModelProto
            var m = new ProtoWriter();
            m.WriteInt64Field(1, 8);                              // ir_version=8 (opset17 compatible) / opset17兼容
            m.WriteStringField(2, modelName);                     // producer_name
            var opset = new ProtoWriter();
            opset.WriteStringField(1, "");                        // domain=""(ai.onnx)
            opset.WriteInt64Field(2, 17);                         // version=17
            m.WriteMessageField(8, opset);                        // opset_import field=8
            m.WriteMessageField(7, graph);                        // graph field=7

            File.WriteAllBytes(path, m.ToArray());
        }
    }
}
