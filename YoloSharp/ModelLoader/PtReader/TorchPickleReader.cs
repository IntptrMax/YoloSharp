using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace YoloSharp.ModelLoader.PtReader
{
    public sealed class TorchPickleReader
    {
        private byte[] _data = Array.Empty<byte>();
        private int _pos;
        private readonly List<object?> _stack = new List<object?>();
        private readonly Dictionary<long, object?> _memo = new Dictionary<long, object?>();
        private readonly List<int> _marks = new List<int>();

        public object? Load(byte[] data)
        {
            _data = data ?? throw new ArgumentNullException(nameof(data));
            _pos = 0; _stack.Clear(); _memo.Clear(); _marks.Clear();
            while (_pos < _data.Length)
            {
                byte op = _data[_pos++];
                if (op == 0x2E) return _stack.Count > 0 ? _stack[_stack.Count - 1] : null;
                Execute(op);
            }
            throw new InvalidDataException("Pickle stream ended before STOP");
        }

        public object? LoadFile(string path) => Load(File.ReadAllBytes(path));

        private void Execute(byte op)
        {
            switch (op)
            {
                case 0x80: _pos++; break;                // PROTO
                case 0x95: _pos += 8; break;             // FRAME

                case 0x28: _marks.Add(_stack.Count); break;                // MARK
                case 0x30: _stack.RemoveAt(_stack.Count - 1); break;       // POP
                case 0x31:                                                 // POP_MARK
                    {
                        int m = _marks[_marks.Count - 1]; _marks.RemoveAt(_marks.Count - 1);
                        _stack.RemoveRange(m, _stack.Count - m); break;
                    }
                case 0x32: _stack.Add(_stack[_stack.Count - 1]); break;    // DUP

                case 0x4E: _stack.Add(null); break;                        // NONE
                case 0x88: _stack.Add(true); break;                        // NEWTRUE
                case 0x89: _stack.Add(false); break;                       // NEWFALSE

                case 0x49:
                    {
                        var s = ReadLine();
                        if (s == "01") _stack.Add(true);
                        else if (s == "00") _stack.Add(false);
                        else _stack.Add(long.Parse(s, CultureInfo.InvariantCulture));
                        break;
                    }
                case 0x4A: _stack.Add((long)ReadI32()); break;
                case 0x4B: _stack.Add((long)_data[_pos++]); break;
                case 0x4D: _stack.Add((long)ReadU16()); break;
                case 0x4C:
                    {
                        var s = ReadLine();
                        if (s.EndsWith("L", StringComparison.Ordinal))
                            s = s.Substring(0, s.Length - 1);
                        _stack.Add(long.Parse(s, CultureInfo.InvariantCulture)); break;
                    }
                case 0x8A: _stack.Add(ReadLongN(_data[_pos++])); break;
                case 0x8B: { int n = ReadI32(); _stack.Add(ReadLongN(n)); break; }

                case 0x46: _stack.Add(double.Parse(ReadLine(), CultureInfo.InvariantCulture)); break;
                case 0x47: _stack.Add(ReadDoubleBE()); break;

                case 0x53: _stack.Add(ReadLine()); break;
                case 0x54: _stack.Add(Encoding.UTF8.GetString(ReadBytes(ReadI32()))); break;
                case 0x55: _stack.Add(Encoding.UTF8.GetString(ReadBytes(_data[_pos++]))); break;
                case 0x56: _stack.Add(ReadLine()); break;
                case 0x58: _stack.Add(ReadUtf8(ReadI32())); break;
                case 0x8C: _stack.Add(ReadUtf8(_data[_pos++])); break;
                case 0x8D: _stack.Add(ReadUtf8((int)ReadI64())); break;

                case 0x42: _stack.Add(ReadBytes(ReadI32())); break;
                case 0x43: _stack.Add(ReadBytes(_data[_pos++])); break;
                case 0x8E: _stack.Add(ReadBytes((int)ReadI64())); break;
                case 0x96: _stack.Add(ReadBytes((int)ReadI64())); break;
                case 0x97: case 0x98: break;

                case 0x70: _memo[long.Parse(ReadLine(), CultureInfo.InvariantCulture)] = _stack[_stack.Count - 1]; break;
                case 0x71: _memo[_data[_pos++]] = _stack[_stack.Count - 1]; break;
                case 0x72: _memo[ReadI32()] = _stack[_stack.Count - 1]; break;
                case 0x67: _stack.Add(_memo[long.Parse(ReadLine(), CultureInfo.InvariantCulture)]); break;
                case 0x68: _stack.Add(_memo[_data[_pos++]]); break;
                case 0x6A: _stack.Add(_memo[ReadI32()]); break;
                case 0x94: _memo[_memo.Count] = _stack[_stack.Count - 1]; break;

                case 0x5D: _stack.Add(new List<object?>()); break;
                case 0x6C: _stack.Add(new List<object?>(PopFromMark())); break;
                case 0x61:
                    {
                        var v = _stack[_stack.Count - 1]; _stack.RemoveAt(_stack.Count - 1);
                        ((List<object?>)_stack[_stack.Count - 1]!).Add(v); break;
                    }
                case 0x65:
                    {
                        var items = PopFromMark();
                        ((List<object?>)_stack[_stack.Count - 1]!).AddRange(items); break;
                    }

                case 0x7D: _stack.Add(new Dictionary<string, object?>()); break;
                case 0x64:
                    {
                        var items = PopFromMark();
                        var d = new Dictionary<string, object?>();
                        for (int i = 0; i + 1 < items.Count; i += 2)
                            d[KeyToString(items[i])] = items[i + 1];
                        _stack.Add(d); break;
                    }
                case 0x73:
                    {
                        var val = _stack[_stack.Count - 1]; _stack.RemoveAt(_stack.Count - 1);
                        var key = _stack[_stack.Count - 1]; _stack.RemoveAt(_stack.Count - 1);
                        if (_stack[_stack.Count - 1] is Dictionary<string, object?> d)
                            d[KeyToString(key)] = val;
                        break;
                    }
                case 0x75:
                    {
                        var items = PopFromMark();
                        if (_stack[_stack.Count - 1] is Dictionary<string, object?> d)
                            for (int i = 0; i + 1 < items.Count; i += 2)
                                d[KeyToString(items[i])] = items[i + 1];
                        break;
                    }

                case 0x29: _stack.Add(Array.Empty<object?>()); break;
                case 0x74: _stack.Add(MakeTupleFromMark()); break;
                case 0x85: _stack.Add(MakeTuple(1)); break;
                case 0x86: _stack.Add(MakeTuple(2)); break;
                case 0x87: _stack.Add(MakeTuple(3)); break;

                case 0x8F: _stack.Add(new HashSet<object?>()); break;
                case 0x90:
                    {
                        var items = PopFromMark();
                        if (_stack[_stack.Count - 1] is HashSet<object?> hs)
                            foreach (var it in items) hs.Add(it);
                        break;
                    }
                case 0x91:
                    {
                        var items = PopFromMark();
                        var hs = new HashSet<object?>();
                        foreach (var it in items) hs.Add(it);
                        _stack.Add(hs); break;
                    }

                case 0x63: // GLOBAL : c<module>\n<name>\n
                    {
                        string module = ReadLine();
                        string name = ReadLine();
                        _stack.Add(new GlobalRef(module, name));
                        break;
                    }
                case 0x93: // STACK_GLOBAL
                    {
                        var nameObj = _stack[_stack.Count - 1]; _stack.RemoveAt(_stack.Count - 1);
                        var moduleObj = _stack[_stack.Count - 1]; _stack.RemoveAt(_stack.Count - 1);
                        _stack.Add(new GlobalRef(moduleObj as string ?? "?", nameObj as string ?? "?"));
                        break;
                    }
                case 0x69: // INST
                    {
                        string module = ReadLine();
                        string name = ReadLine();
                        var args = PopFromMark();
                        _stack.Add(new PyObject
                        {
                            ClassName = new GlobalRef(module, name).FullName,
                            Args = args.ToArray()
                        });
                        break;
                    }
                case 0x6F: // OBJ
                    {
                        var args = PopFromMark();
                        var cls = _stack[_stack.Count - 1]; _stack.RemoveAt(_stack.Count - 1);
                        _stack.Add(new PyObject { ClassName = GetClassName(cls), Args = args.ToArray() });
                        break;
                    }
                case 0x81: // NEWOBJ
                    {
                        var argsObj = _stack[_stack.Count - 1]; _stack.RemoveAt(_stack.Count - 1);
                        var cls = _stack[_stack.Count - 1]; _stack.RemoveAt(_stack.Count - 1);
                        var args = argsObj as object?[] ?? new object?[] { argsObj };
                        _stack.Add(new PyObject { ClassName = GetClassName(cls), Args = args });
                        break;
                    }
                case 0x92: // NEWOBJ_EX
                    {
                        var kwargsObj = _stack[_stack.Count - 1]; _stack.RemoveAt(_stack.Count - 1);
                        var argsObj = _stack[_stack.Count - 1]; _stack.RemoveAt(_stack.Count - 1);
                        var cls = _stack[_stack.Count - 1]; _stack.RemoveAt(_stack.Count - 1);
                        var args = argsObj as object?[] ?? new object?[] { argsObj };
                        var kwargs = kwargsObj as Dictionary<string, object?> ?? new Dictionary<string, object?>();
                        var py = new PyObject { ClassName = GetClassName(cls), Args = args };
                        foreach (var kv in kwargs) py.State[kv.Key] = kv.Value;
                        _stack.Add(py); break;
                    }
                case 0x52: // REDUCE
                    {
                        var argsObj = _stack[_stack.Count - 1]; _stack.RemoveAt(_stack.Count - 1);
                        var callable = _stack[_stack.Count - 1]; _stack.RemoveAt(_stack.Count - 1);
                        _stack.Add(DoReduce(callable, argsObj));
                        break;
                    }
                case 0x62: // BUILD
                    {
                        var state = _stack[_stack.Count - 1]; _stack.RemoveAt(_stack.Count - 1);
                        DoBuild(_stack[_stack.Count - 1], state);
                        break;
                    }

                // ---- persistent ----
                case 0x50: _stack.Add(DoPersistentLoad(ReadLine())); break;
                case 0x51:
                    {
                        var pid = _stack[_stack.Count - 1]; _stack.RemoveAt(_stack.Count - 1);
                        _stack.Add(DoPersistentLoad(pid)); break;
                    }

                // ---- EXT ----
                case 0x82: _pos += 1; throw new NotSupportedException("EXT1 not supported");
                case 0x83: _pos += 2; throw new NotSupportedException("EXT2 not supported");
                case 0x84: _pos += 4; throw new NotSupportedException("EXT4 not supported");

                default:
                    throw new NotSupportedException($"Unsupported opcode 0x{op:X2} at {_pos - 1}");
            }
        }

        private object? DoReduce(object? callable, object? argsObj)
        {
            var args = argsObj as object?[] ?? new object?[] { argsObj };

            if (callable is GlobalRef g)
            {
                switch (g.FullName)
                {
                    case "torch._utils._rebuild_tensor_v2":
                    case "torch._utils._rebuild_tensor_v3":
                    case "torch._utils._rebuild_tensor":
                        return RebuildTensor(args);

                    case "torch._utils._rebuild_parameter":
                    case "torch._utils._rebuild_parameter_with_state":
                        return args.Length > 0 ? args[0] : null;

                    case "torch.Size":
                        return args.Length > 0 ? args[0] : Array.Empty<object?>();

                    case "torch.device":
                    case "torch.dtype":
                        return args.Length > 0 ? args[0]?.ToString() ?? "" : "";

                    case "collections.OrderedDict":
                        return new Dictionary<string, object?>();

                    case "builtins.set":
                    case "builtins.frozenset":
                        return new HashSet<object?>();

                    case "builtins.bytearray":
                        return args.Length > 0 && args[0] is byte[] b ? b : Array.Empty<byte>();

                    case "builtins.complex":
                        return args.Length >= 2
                            ? new System.Numerics.Complex(Convert.ToDouble(args[0]), Convert.ToDouble(args[1]))
                            : System.Numerics.Complex.Zero;

                    case "builtins.range":
                        if (args.Length == 1)
                            return new List<object?>(Enumerable.Range(0, Convert.ToInt32(args[0])).Select(x => (object?)(long)x));
                        return new List<object?>();
                }
            }

            return new PyObject { ClassName = GetClassName(callable), Args = args };
        }

        private TorchTensor RebuildTensor(object?[] args)
        {
            if (args.Length < 4)
                throw new InvalidDataException("_rebuild_tensor_v2 needs >= 4 args");
            if (args[0] is not StorageRef storage)
                throw new InvalidCastException($"args[0] is not StorageRef (actual: {args[0]?.GetType().Name ?? "null"})");

            long offset = Convert.ToInt64(args[1]);
            long[] size = ToLongArray(args[2]);
            long[] stride = ToLongArray(args[3]);
            bool req = args.Length > 4 && args[4] is bool b && b;

            return new TorchTensor
            {
                Storage = storage,
                StorageOffset = offset,
                Size = size,
                Stride = stride,
                RequiresGrad = req
            };
        }

        private static long[] ToLongArray(object? obj) => obj switch
        {
            null => Array.Empty<long>(),
            long[] la => la,
            int[] ia => ia.Select(x => (long)x).ToArray(),
            object?[] oa => oa.Select(x => Convert.ToInt64(x)).ToArray(),
            IEnumerable<object?> en => en.Select(x => Convert.ToInt64(x)).ToArray(),
            _ => new[] { Convert.ToInt64(obj) }
        };

        private object DoPersistentLoad(object? pid)
        {
            if (pid is object?[] arr && arr.Length >= 5 && Equals(arr[0], "storage"))
            {
                var st = arr[1] as GlobalRef;
                return new StorageRef
                {
                    TypeName = st?.Name ?? "UnknownStorage",
                    Key = arr[2]?.ToString() ?? "0",
                    Location = arr[3]?.ToString() ?? "cpu",
                    Numel = Convert.ToInt64(arr[4])
                };
            }
            if (pid is string s && s.StartsWith("storage", StringComparison.Ordinal))
            {
                var parts = s.Split(':');
                if (parts.Length >= 5)
                    return new StorageRef
                    {
                        TypeName = parts[1],
                        Key = parts[2],
                        Location = parts[3],
                        Numel = long.Parse(parts[4], CultureInfo.InvariantCulture)
                    };
            }
            throw new NotSupportedException($"Unsupported persistent id: {pid}");
        }

        private void DoBuild(object? obj, object? state)
        {
            if (obj is PyObject py && state is Dictionary<string, object?> dict)
                foreach (var kv in dict) py.State[kv.Key] = kv.Value;
        }

        private object?[] MakeTuple(int n)
        {
            var tuple = new object?[n];
            for (int i = n - 1; i >= 0; i--) { tuple[i] = _stack[_stack.Count - 1]; _stack.RemoveAt(_stack.Count - 1); }
            return tuple;
        }

        private object?[] MakeTupleFromMark()
        {
            int mark = _marks[_marks.Count - 1]; _marks.RemoveAt(_marks.Count - 1);
            int count = _stack.Count - mark;
            var tuple = new object?[count];
            for (int i = 0; i < count; i++) tuple[i] = _stack[mark + i];
            _stack.RemoveRange(mark, count);
            return tuple;
        }

        private List<object?> PopFromMark()
        {
            int mark = _marks[_marks.Count - 1]; _marks.RemoveAt(_marks.Count - 1);
            var items = _stack.GetRange(mark, _stack.Count - mark);
            _stack.RemoveRange(mark, _stack.Count - mark);
            return items;
        }

        private static string GetClassName(object? clsObj) => clsObj switch
        {
            GlobalRef g => g.FullName,
            PyObject py => py.ClassName,
            string s => s,
            null => "?",
            _ => clsObj.ToString() ?? "?"
        };

        private static string KeyToString(object? key) => key switch
        {
            null => "null",
            string s => s,
            GlobalRef g => g.FullName,
            _ => key.ToString() ?? "null"
        };

        private int ReadI32()
        {
            int v = _data[_pos] | (_data[_pos + 1] << 8) | (_data[_pos + 2] << 16) | (_data[_pos + 3] << 24);
            _pos += 4; return v;
        }
        private uint ReadU16()
        {
            uint v = (uint)(_data[_pos] | (_data[_pos + 1] << 8));
            _pos += 2; return v;
        }
        private long ReadI64()
        {
            long v = 0;
            for (int i = 0; i < 8; i++) v |= (long)_data[_pos + i] << (8 * i);
            _pos += 8; return v;
        }
        private string ReadUtf8(int len) { var s = Encoding.UTF8.GetString(_data, _pos, len); _pos += len; return s; }
        private byte[] ReadBytes(int len) { var b = new byte[len]; Array.Copy(_data, _pos, b, 0, len); _pos += len; return b; }
        private string ReadLine()
        {
            int start = _pos;
            while (_pos < _data.Length && _data[_pos] != (byte)'\n') _pos++;
            var s = Encoding.UTF8.GetString(_data, start, _pos - start);
            if (_pos < _data.Length) _pos++;
            return s;
        }
        private double ReadDoubleBE()
        {
            var tmp = new byte[8];
            Array.Copy(_data, _pos, tmp, 0, 8);
            if (BitConverter.IsLittleEndian) Array.Reverse(tmp);
            _pos += 8;
            return BitConverter.ToDouble(tmp, 0);
        }
        private long ReadLongN(int n)
        {
            if (n == 0) return 0L;
            if (n > 8) { _pos += n - 8; n = 8; }
            long v = 0;
            for (int i = 0; i < n; i++) v |= (long)_data[_pos + i] << (8 * i);
            if (n < 8 && (_data[_pos + n - 1] & 0x80) != 0) v |= -1L << (8 * n);
            _pos += n;
            return v;
        }
    }
}