//using System;
//using System.Collections.Generic;
//using System.Collections.ObjectModel;
//using System.IO;
//using System.IO.Compression;
//using System.Linq;
//using System.Text;
//using TorchSharp;

//namespace YoloSharp.ModelLoader
//{
//    // 用于模拟 Python 对象的通用类
//    internal class PyTorchObject
//    {
//        public string ClassName { get; set; }
//        public Dictionary<string, object> Attributes { get; set; } = new Dictionary<string, object>();
//    }



//    public class PickleLoader
//    {
//        private ZipArchive zip;
//        private ReadOnlyCollection<ZipArchiveEntry> entries;

//        public List<CommonTensor> ReadTensorsInfoFromFile(string fileName)
//        {
//            zip = ZipFile.OpenRead(fileName);
//            entries = zip.Entries;

//            ZipArchiveEntry headerEntry = entries.FirstOrDefault(e => e.FullName == "archive/data.pkl");
//            if (headerEntry == null) headerEntry = entries.FirstOrDefault(e => e.Name == "data.pkl");

//            if (headerEntry == null) throw new ArgumentException("Could not find data.pkl in the archive.");

//            byte[] headerBytes;
//            using (Stream stream = headerEntry.Open())
//            using (MemoryStream ms = new MemoryStream())
//            {
//                stream.CopyTo(ms);
//                headerBytes = ms.ToArray();
//            }

//            if (headerBytes[0] != 0x80) throw new ArgumentException("Not a valid pickle file.");

//            // 解析 Pickle 并返回根对象
//            var rootObject = ParsePickle(headerBytes, fileName);

//            // 从根对象递归提取所有张量
//            return ExtractTensors(rootObject);
//        }

//        // 递归提取张量的核心逻辑
//        private List<CommonTensor> ExtractTensors(object obj, string prefix = "")
//        {
//            var tensors = new List<CommonTensor>();

//            if (obj is CommonTensor tensor)
//            {
//                if (!string.IsNullOrEmpty(prefix))
//                {
//                    tensor.Name = prefix;
//                }
//                tensors.Add(tensor);
//            }
//            else if (obj is PyTorchObject pyObj)
//            {
//                // 如果是 PyTorch 对象，遍历其属性
//                foreach (var attr in pyObj.Attributes)
//                {
//                    string newPrefix = string.IsNullOrEmpty(prefix) ? attr.Key : $"{prefix}.{attr.Key}";
//                    tensors.AddRange(ExtractTensors(attr.Value, newPrefix));
//                }
//            }
//            else if (obj is List<object> list)
//            {
//                // 如果是列表，遍历元素 (对应 Python 的 model.0, model.1 ...)
//                for (int i = 0; i < list.Count; i++)
//                {
//                    if (list[i] != null)
//                    {
//                        string newPrefix = string.IsNullOrEmpty(prefix) ? i.ToString() : $"{prefix}.{i}";
//                        tensors.AddRange(ExtractTensors(list[i], newPrefix));
//                    }
//                }
//            }
//            else if (obj is Dictionary<string, object> dict)
//            {
//                // 如果是字典，遍历键值对
//                foreach (var kvp in dict)
//                {
//                    string newPrefix = string.IsNullOrEmpty(prefix) ? kvp.Key : $"{prefix}.{kvp.Key}";
//                    tensors.AddRange(ExtractTensors(kvp.Value, newPrefix));
//                }
//            }

//            return tensors;
//        }

//        private object ParsePickle(byte[] data, string fileName)
//        {
//            var stack = new Stack<object>();
//            var memo = new Dictionary<int, object>();

//            int index = 2; // Skip protocol header
//            CommonTensor currentStorage = null;

//            while (index < data.Length)
//            {
//                byte opcode = data[index];

//                switch (opcode)
//                {
//                    case (byte)'.': // STOP
//                        return stack.Count > 0 ? stack.Pop() : null;

//                    case (byte)'(': // MARK
//                        stack.Push("MARK");
//                        index++;
//                        break;

//                    case (byte)'0': // POP
//                        if (stack.Count > 0) stack.Pop();
//                        index++;
//                        break;

//                    case (byte)'q': // BINPUT
//                    case (byte)'r': // LONG_BINPUT
//                        int memoIdx = (opcode == 'q') ? data[index + 1] : BitConverter.ToInt32(data, index + 1);
//                        index += (opcode == 'q') ? 2 : 5;
//                        if (stack.Count > 0) memo[memoIdx] = stack.Peek();
//                        break;

//                    case (byte)'h': // BINGET
//                    case (byte)'j': // LONG_BINGET
//                        int getId = (opcode == 'h') ? data[index + 1] : BitConverter.ToInt32(data, index + 1);
//                        index += (opcode == 'h') ? 2 : 5;
//                        if (memo.ContainsKey(getId)) stack.Push(memo[getId]);
//                        else stack.Push(null);
//                        break;

//                    case (byte)'Q': // BINPERSID
//                        if (stack.Count > 0)
//                        {
//                            var pid = stack.Pop();
//                            var storage = LoadPersistentId(pid, fileName);
//                            stack.Push(storage);
//                        }
//                        index++;
//                        break;

//                    case (byte)'K': // BININT1
//                        stack.Push((int)data[index + 1]);
//                        index += 2;
//                        break;

//                    case (byte)'M': // BININT2
//                        stack.Push((int)BitConverter.ToUInt16(data, index + 1));
//                        index += 3;
//                        break;

//                    case (byte)'J': // BININT
//                        stack.Push(BitConverter.ToInt32(data, index + 1));
//                        index += 5;
//                        break;

//                    case 0x8C: // SHORT_BINUNICODE
//                        int len8C = data[index + 1];
//                        stack.Push(Encoding.UTF8.GetString(data, index + 2, len8C));
//                        index += 2 + len8C;
//                        break;

//                    case (byte)'X': // BINUNICODE
//                        int lenX = BitConverter.ToInt32(data, index + 1);
//                        stack.Push(Encoding.UTF8.GetString(data, index + 5, lenX));
//                        index += 5 + lenX;
//                        break;

//                    case (byte)'c': // GLOBAL
//                        {
//                            int start = index + 1;
//                            int end = start;
//                            // 查找第一个换行符
//                            while (end < data.Length && data[end] != '\n') end++;
//                            string module = Encoding.UTF8.GetString(data, start, end - start);

//                            start = end + 1;
//                            end = start;
//                            // 查找第二个换行符
//                            while (end < data.Length && data[end] != '\n') end++;
//                            string name = Encoding.UTF8.GetString(data, start, end - start);

//                            // 将类名压栈，用于 REDUCE
//                            stack.Push(new PyTorchObject { ClassName = $"{module}.{name}" });
//                            index = end + 1;
//                        }
//                        break;

//                    case (byte)'R': // REDUCE
//                        {
//                            var args = stack.Pop() as List<object>;
//                            var func = stack.Pop();

//                            if (func is PyTorchObject pyObj)
//                            {
//                                // 检查是否是 Tensor 重建函数
//                                if (pyObj.ClassName.Contains("_rebuild_tensor"))
//                                {
//                                    var tensor = RebuildTensor(args);
//                                    stack.Push(tensor);
//                                }
//                                else
//                                {
//                                    // 普通对象构建，将参数存入属性以便后续 BUILD 使用
//                                    // 对于 YOLOv8, 构造函数参数通常不重要，重要的是 __dict__
//                                    // 这里我们创建一个空对象占位
//                                    stack.Push(new PyTorchObject { ClassName = pyObj.ClassName });
//                                }
//                            }
//                            else
//                            {
//                                stack.Push(null); // 未知类型
//                            }
//                            index++;
//                        }
//                        break;

//                    case (byte)'b': // BUILD
//                        {
//                            var state = stack.Pop(); // 通常是 Dict 或 List
//                            var obj = stack.Pop();   // 要修改的对象

//                            if (obj is PyTorchObject pyObj && state is Dictionary<string, object> stateDict)
//                            {
//                                // 将 state 中的属性合并到对象中
//                                foreach (var kvp in stateDict)
//                                {
//                                    pyObj.Attributes[kvp.Key] = kvp.Value;
//                                }
//                            }
//                            else if (obj is PyTorchObject pyObjList && state is List<object> stateList)
//                            {
//                                // 某些对象可能用 list 初始化，暂忽略
//                            }

//                            stack.Push(obj); // 将对象放回栈顶
//                            index++;
//                        }
//                        break;

//                    case (byte)'t': // TUPLE
//                        {
//                            var list = new List<object>();
//                            while (stack.Count > 0 && !(stack.Peek() is string s && s == "MARK"))
//                            {
//                                list.Insert(0, stack.Pop());
//                            }
//                            if (stack.Count > 0) stack.Pop(); // Pop MARK
//                            stack.Push(list);
//                        }
//                        index++;
//                        break;

//                    case (byte)'s': // SETITEM
//                        {
//                            var value = stack.Pop();
//                            var key = stack.Pop();
//                            var dict = stack.Pop();
//                            if (dict is Dictionary<string, object> d && key is string k)
//                            {
//                                d[k] = value;
//                            }
//                            stack.Push(dict); // 字典放回栈
//                        }
//                        index++;
//                        break;

//                    case (byte)'}': // EMPTY_DICT
//                        stack.Push(new Dictionary<string, object>());
//                        index++;
//                        break;

//                    case (byte)']': // EMPTY_LIST
//                        stack.Push(new List<object>());
//                        index++;
//                        break;

//                    case (byte)'e': // APPENDS
//                        {
//                            var items = new List<object>();
//                            while (stack.Count > 0 && !(stack.Peek() is string s && s == "MARK"))
//                            {
//                                items.Insert(0, stack.Pop());
//                            }
//                            if (stack.Count > 0) stack.Pop(); // Pop MARK

//                            var list = stack.Peek() as List<object>;
//                            if (list != null) list.AddRange(items);
//                        }
//                        index++;
//                        break;

//                    default:
//                        index++;
//                        break;
//                }
//            }
//            return null;
//        }

//        // 解析 Persistent ID，提取存储信息
//        private object LoadPersistentId(object pid, string fileName)
//        {
//            if (pid is List<object> pidList)
//            {
//                if (pidList.Count > 0 && pidList[0] as string == "storage")
//                {
//                    // 格式: ('storage', storage_type, key, device, numel)
//                    if (pidList.Count >= 5)
//                    {
//                        string storageType = pidList[1]?.ToString();
//                        string key = pidList[2]?.ToString();

//                        torch.ScalarType sType = torch.ScalarType.Float32;
//                        if (storageType.Contains("HalfStorage")) sType = torch.ScalarType.Float16;
//                        else if (storageType.Contains("BFloat16Storage")) sType = torch.ScalarType.BFloat16;
//                        else if (storageType.Contains("IntStorage")) sType = torch.ScalarType.Int32;
//                        else if (storageType.Contains("LongStorage")) sType = torch.ScalarType.Int64;

//                        return new CommonTensor
//                        {
//                            DataNameInZipFile = key,
//                            Type = sType,
//                            FileName = fileName
//                        };
//                    }
//                }
//            }
//            return null;
//        }

//        // 重建张量对象
//        private CommonTensor RebuildTensor(List<object> args)
//        {
//            // args: [storage, offset, shape, stride, ...]
//            if (args == null || args.Count < 4) return null;

//            var storage = args[0] as CommonTensor;
//            if (storage == null) return null;

//            CommonTensor tensor = new CommonTensor
//            {
//                DataNameInZipFile = storage.DataNameInZipFile,
//                Type = storage.Type,
//                FileName = storage.FileName,
//                Offset = new List<ulong>()
//            };

//            if (args[1] is int offsetInt)
//            {
//                tensor.Offset.Add((ulong)offsetInt);
//            }

//            if (args[2] is List<object> shapeList)
//            {
//                foreach (var dim in shapeList) tensor.Shape.Add((int)dim);
//            }

//            if (args[3] is List<object> strideList)
//            {
//                foreach (var s in strideList) tensor.Stride.Add(Convert.ToUInt64(s));
//            }

//            return tensor;
//        }

//        private byte[] ReadByteFromFile(CommonTensor tensor)
//        {
//            // ... (保持之前的 ReadByteFromFile 实现，支持 archive/data/ 路径查找)
//            if (entries is null) throw new ArgumentNullException(nameof(entries));

//            ZipArchiveEntry dataEntry = entries.FirstOrDefault(e =>
//                e.FullName == tensor.DataNameInZipFile ||
//                e.Name == tensor.DataNameInZipFile ||
//                e.FullName.EndsWith("/" + tensor.DataNameInZipFile));

//            if (dataEntry == null)
//            {
//                // 尝试查找 numeric key (YOLOv8 format is usually 'archive/data/X')
//                // 这里做一些容错处理
//                string altName = "archive/data/" + tensor.DataNameInZipFile;
//                dataEntry = entries.FirstOrDefault(e => e.FullName == altName);
//            }

//            if (dataEntry == null) throw new FileNotFoundException($"Could not find data file '{tensor.DataNameInZipFile}' in archive.");

//            long totalElements = 1;
//            foreach (var dim in tensor.Shape) totalElements *= dim;
//            long length = totalElements * tensor.Type.ElementSize();
//            long offset = (tensor.Offset != null && tensor.Offset.Count > 0) ? (long)tensor.Offset[0] : 0;

//            byte[] data = new byte[length];
//            using (Stream stream = dataEntry.Open())
//            {
//                stream.Seek(offset, SeekOrigin.Begin);
//                stream.Read(data, 0, (int)length);
//            }
//            return data;
//        }

//        public Dictionary<string, torch.Tensor> Load(string fileName, string addString = "")
//        {
//            Dictionary<string, torch.Tensor> result = new Dictionary<string, torch.Tensor>();
//            List<CommonTensor> tensorInfos = ReadTensorsInfoFromFile(fileName);

//            foreach (CommonTensor tensorInfo in tensorInfos)
//            {
//                if (string.IsNullOrEmpty(tensorInfo.Name)) continue;

//                try
//                {
//                    byte[] bytes = ReadByteFromFile(tensorInfo);
//                    torch.Tensor tensor = torch.empty(tensorInfo.Shape.ToArray(), dtype: tensorInfo.Type);
//                    tensor.bytes = bytes;

//                    // YOLOv8 的名字通常很长，可以在这里做一下清理，比如去掉开头的前缀
//                    // 比如解析出来是 "model.model.0.conv.weight"，你可能只想要 "model.0.conv.weight"
//                    // 这里保留全名
//                    string finalName = addString + tensorInfo.Name;

//                    if (!result.ContainsKey(finalName))
//                    {
//                        result.Add(finalName, tensor);
//                    }
//                }
//                catch (Exception ex)
//                {
//                    Console.WriteLine($"Failed to load tensor {tensorInfo.Name}: {ex.Message}");
//                }
//            }
//            return result;
//        }
//    }
//}


using System.Collections.ObjectModel;
using System.IO.Compression;
using System.Text;
using TorchSharp;

namespace YoloSharp.ModelLoader
{
    // 用于模拟 Python 函数/类引用
    internal class PyFunction
    {
        public string Name { get; set; }
    }

    // 用于模拟 Python 对象
    internal class PyTorchObject
    {
        public string ClassName { get; set; }
        public Dictionary<string, object> Attributes { get; set; } = new Dictionary<string, object>();
    }



    public class PickleLoader
    {
        private ZipArchive zip;
        private ReadOnlyCollection<ZipArchiveEntry> entries;

        public List<CommonTensor> ReadTensorsInfoFromFile(string fileName)
        {
            zip = ZipFile.OpenRead(fileName);
            entries = zip.Entries;

            // 查找 data.pkl，支持带路径和不带路径的情况
            ZipArchiveEntry headerEntry = entries.FirstOrDefault(e => e.FullName == "archive/data.pkl");
            if (headerEntry == null) headerEntry = entries.FirstOrDefault(e => e.Name == "data.pkl");

            if (headerEntry == null) throw new ArgumentException("Could not find data.pkl in the archive.");

            byte[] headerBytes;
            using (Stream stream = headerEntry.Open())
            using (MemoryStream ms = new MemoryStream())
            {
                stream.CopyTo(ms);
                headerBytes = ms.ToArray();
            }

            if (headerBytes[0] != 0x80) throw new ArgumentException("Not a valid pickle file.");

            var (stack, memo) = ParsePickle(headerBytes, fileName);
            return ExtractTensors(null);
        }

        private List<CommonTensor> ExtractTensors(object obj, string prefix = "")
        {
            var tensors = new List<CommonTensor>();

            if (obj is CommonTensor tensor)
            {
                if (!string.IsNullOrEmpty(prefix)) tensor.Name = prefix;
                tensors.Add(tensor);
            }
            else if (obj is PyTorchObject pyObj)
            {
                foreach (var attr in pyObj.Attributes)
                {
                    string newPrefix = string.IsNullOrEmpty(prefix) ? attr.Key : $"{prefix}.{attr.Key}";
                    tensors.AddRange(ExtractTensors(attr.Value, newPrefix));
                }
            }
            else if (obj is List<object> list)
            {
                for (int i = 0; i < list.Count; i++)
                {
                    if (list[i] != null)
                    {
                        string newPrefix = string.IsNullOrEmpty(prefix) ? i.ToString() : $"{prefix}.{i}";
                        tensors.AddRange(ExtractTensors(list[i], newPrefix));
                    }
                }
            }
            else if (obj is Dictionary<string, object> dict)
            {
                foreach (var kvp in dict)
                {
                    string newPrefix = string.IsNullOrEmpty(prefix) ? kvp.Key : $"{prefix}.{kvp.Key}";
                    tensors.AddRange(ExtractTensors(kvp.Value, newPrefix));
                }
            }
            return tensors;
        }

        private (Stack<object> stack, Dictionary<int, object> memo) ParsePickle(byte[] data, string fileName)
        {
            var stack = new Stack<object>();
            var memo = new Dictionary<int, object>();
            int index = 2; // Skip 0x80 + protocol version

            while (index < data.Length)
            {
                byte opcode = data[index];

                switch (opcode)
                {
                    case (byte)'.': // STOP
                        return (stack, memo);

                    case (byte)'(': // MARK
                        stack.Push("MARK");
                        index++;
                        break;

                    case (byte)'0': // POP
                        if (stack.Count > 0) stack.Pop();
                        index++;
                        break;

                    case (byte)'q': // BINPUT (1-byte index)
                    case (byte)'r': // LONG_BINPUT (4-byte index)
                        int memoIdx = (opcode == 'q') ? data[index + 1] : BitConverter.ToInt32(data, index + 1);
                        index += (opcode == 'q') ? 2 : 5;
                        if (stack.Count > 0) memo[memoIdx] = stack.Peek();
                        break;

                    case (byte)'h': // BINGET (1-byte index)
                    case (byte)'j': // LONG_BINGET (4-byte index)
                        int getId = (opcode == 'h') ? data[index + 1] : BitConverter.ToInt32(data, index + 1);
                        index += (opcode == 'h') ? 2 : 5;
                        stack.Push(memo.ContainsKey(getId) ? memo[getId] : null);
                        break;

                    case 0x94: // MEMOIZE (Protocol 4+, uses top of stack)
                        if (stack.Count > 0) memo[memo.Count] = stack.Peek();
                        index++;
                        break;

                    case (byte)'Q': // BINPERSID
                        if (stack.Count > 0)
                        {
                            var pid = stack.Pop();
                            var storage = LoadPersistentId(pid, fileName);
                            stack.Push(storage);
                        }
                        index++;
                        break;

                    case (byte)'K': // BININT1
                        stack.Push((int)data[index + 1]);
                        index += 2;
                        break;

                    case (byte)'M': // BININT2
                        stack.Push((int)BitConverter.ToUInt16(data, index + 1));
                        index += 3;
                        break;

                    case (byte)'J': // BININT
                        stack.Push(BitConverter.ToInt32(data, index + 1));
                        index += 5;
                        break;

                    case 0x8C: // SHORT_BINUNICODE
                        int len8C = data[index + 1];
                        stack.Push(Encoding.UTF8.GetString(data, index + 2, len8C));
                        index += 2 + len8C;
                        break;

                    case (byte)'X': // BINUNICODE
                        int lenX = BitConverter.ToInt32(data, index + 1);
                        stack.Push(Encoding.UTF8.GetString(data, index + 5, lenX));
                        index += 5 + lenX;
                        break;

                    case (byte)'c': // GLOBAL
                        {
                            int start = index + 1;
                            int end = start;
                            while (end < data.Length && data[end] != '\n') end++;
                            string module = Encoding.UTF8.GetString(data, start, end - start);

                            start = end + 1;
                            end = start;
                            while (end < data.Length && data[end] != '\n') end++;
                            string name = Encoding.UTF8.GetString(data, start, end - start);

                            stack.Push(new PyFunction { Name = $"{module}.{name}" });
                            index = end + 1;
                        }
                        break;

                    case 0x93: // STACK_GLOBAL
                        {
                            var nameObj = stack.Pop();
                            var moduleObj = stack.Pop();
                            stack.Push(new PyFunction { Name = $"{moduleObj}.{nameObj}" });
                            index++;
                        }
                        break;

                    case (byte)'t': // TUPLE
                        {
                            var list = new List<object>();
                            while (stack.Count > 0 && !(stack.Peek() is string s && s == "MARK"))
                            {
                                list.Insert(0, stack.Pop());
                            }
                            if (stack.Count > 0) stack.Pop(); // Pop MARK
                            stack.Push(list);
                            index++;
                        }
                        break;

                    case 0x85: // TUPLE1
                        {
                            var item1 = stack.Pop();
                            stack.Push(new List<object> { item1 });
                            index++;
                        }
                        break;

                    case 0x86: // TUPLE2
                        {
                            var item2 = stack.Pop();
                            var item1 = stack.Pop();
                            stack.Push(new List<object> { item1, item2 });
                            index++;
                        }
                        break;

                    case 0x87: // TUPLE3
                        {
                            var item3 = stack.Pop();
                            var item2 = stack.Pop();
                            var item1 = stack.Pop();
                            stack.Push(new List<object> { item1, item2, item3 });
                            index++;
                        }
                        break;

                    case (byte)')': // EMPTY_TUPLE
                        stack.Push(new List<object>());
                        index++;
                        break;

                    case (byte)'R': // REDUCE
                        {
                            var args = stack.Pop() as List<object>;
                            var func = stack.Pop();

                            if (func is PyFunction pyFunc)
                            {
                                if (pyFunc.Name.Contains("_rebuild_tensor"))
                                {
                                    stack.Push(RebuildTensor(args));
                                }
                                else
                                {
                                    // 对于普通类，我们构造一个对象占位
                                    stack.Push(new PyTorchObject { ClassName = pyFunc.Name });
                                }
                            }
                            else
                            {
                                stack.Push(null);
                            }
                            index++;
                        }
                        break;

                    case 0x81: // NEWOBJ (类似 REDUCE，通常用于 __new__)
                        {
                            var args = stack.Pop() as List<object>;
                            var cls = stack.Pop();
                            if (cls is PyFunction pyFunc)
                            {
                                stack.Push(new PyTorchObject { ClassName = pyFunc.Name });
                            }
                            else
                            {
                                stack.Push(new PyTorchObject());
                            }
                            index++;
                        }
                        break;

                    case (byte)'b': // BUILD
                        {
                            var state = stack.Pop();
                            var obj = stack.Pop();

                            if (obj is PyTorchObject pyObj)
                            {
                                if (state is Dictionary<string, object> stateDict)
                                {
                                    foreach (var kvp in stateDict) pyObj.Attributes[kvp.Key] = kvp.Value;
                                }
                                else if (state is List<object> stateList)
                                {
                                    // 处理 list 类型的 state，通常对应 __setstate__ 的列表参数
                                    // 这里简化处理，如果需要可以映射到属性
                                }
                            }

                            stack.Push(obj);
                            index++;
                        }
                        break;

                    case (byte)'s': // SETITEM
                        {
                            var value = stack.Pop();
                            var key = stack.Pop();
                            var dict = stack.Pop();
                            if (dict is Dictionary<string, object> d && key is string k) d[k] = value;
                            stack.Push(dict);
                            index++;
                        }
                        break;

                    case (byte)'}': // EMPTY_DICT
                        stack.Push(new Dictionary<string, object>());
                        index++;
                        break;

                    case (byte)']': // EMPTY_LIST
                        stack.Push(new List<object>());
                        index++;
                        break;

                    case (byte)'e': // APPENDS
                        {
                            var items = new List<object>();
                            while (stack.Count > 0 && !(stack.Peek() is string s && s == "MARK"))
                            {
                                items.Insert(0, stack.Pop());
                            }
                            if (stack.Count > 0) stack.Pop();
                            var list = stack.Peek() as List<object>;
                            if (list != null) list.AddRange(items);
                            index++;
                        }
                        break;

                    case 0x95: // FRAME (Protocol 4+)
                        index += 9; // Skip opcode + 8 bytes frame size
                        break;

                    default:
                        // 未知操作码直接跳过可能会导致解析错误，但在未知情况下只能前进
                        index++;
                        break;
                }
            }
            return (stack, memo);
        }

        private object LoadPersistentId(object pid, string fileName)
        {
            if (pid is List<object> pidList && pidList.Count > 0)
            {
                if (pidList[0] as string == "storage" && pidList.Count >= 5)
                {
                    // 结构: ('storage', storage_type, key, device, numel)
                    string key = pidList[2]?.ToString();
                    object storageTypeObj = pidList[1];
                    string storageTypeStr = "";

                    if (storageTypeObj is PyFunction pyFunc) storageTypeStr = pyFunc.Name;
                    else if (storageTypeObj is PyTorchObject pyObj) storageTypeStr = pyObj.ClassName;
                    else storageTypeStr = storageTypeObj?.ToString();

                    torch.ScalarType sType = torch.ScalarType.Float32;
                    if (storageTypeStr.Contains("HalfStorage")) sType = torch.ScalarType.Float16;
                    else if (storageTypeStr.Contains("BFloat16Storage")) sType = torch.ScalarType.BFloat16;
                    else if (storageTypeStr.Contains("IntStorage")) sType = torch.ScalarType.Int32;
                    else if (storageTypeStr.Contains("LongStorage")) sType = torch.ScalarType.Int64;

                    return new CommonTensor
                    {
                        DataNameInZipFile = key,
                        Type = sType,
                        FileName = fileName
                    };
                }
            }
            return null;
        }

        private CommonTensor RebuildTensor(List<object> args)
        {
            // args: [storage, offset, shape, stride, ...]
            if (args == null || args.Count < 4) return null;

            var storage = args[0] as CommonTensor;
            if (storage == null) return null;

            CommonTensor tensor = new CommonTensor
            {
                DataNameInZipFile = storage.DataNameInZipFile,
                Type = storage.Type,
                FileName = storage.FileName,
                Offset = new List<ulong>()
            };

            if (args[1] is int offsetInt) tensor.Offset.Add((ulong)offsetInt);

            if (args[2] is List<object> shapeList)
                foreach (var dim in shapeList) tensor.Shape.Add((int)dim);

            if (args[3] is List<object> strideList)
                foreach (var s in strideList) tensor.Stride.Add(Convert.ToUInt64(s));

            return tensor;
        }

        private byte[] ReadByteFromFile(CommonTensor tensor)
        {
            if (entries is null) throw new ArgumentNullException(nameof(entries));

            // 尝试多种路径匹配
            ZipArchiveEntry dataEntry = entries.FirstOrDefault(e =>
                e.FullName == tensor.DataNameInZipFile ||
                e.Name == tensor.DataNameInZipFile ||
                e.FullName.EndsWith("/" + tensor.DataNameInZipFile));

            // YOLOv8 special case: keys are numeric strings like "0", files are "archive/data/0"
            if (dataEntry == null && int.TryParse(tensor.DataNameInZipFile, out _))
            {
                dataEntry = entries.FirstOrDefault(e => e.FullName.EndsWith("data/" + tensor.DataNameInZipFile));
            }

            if (dataEntry == null) throw new FileNotFoundException($"Could not find data file '{tensor.DataNameInZipFile}' in archive.");

            long totalElements = 1;
            foreach (var dim in tensor.Shape) totalElements *= dim;
            long length = totalElements * tensor.Type.ElementSize();
            long offset = (tensor.Offset != null && tensor.Offset.Count > 0) ? (long)tensor.Offset[0] : 0;

            byte[] data = new byte[length];
            using (Stream stream = dataEntry.Open())
            {
                stream.Seek(offset, SeekOrigin.Begin);
                stream.Read(data, 0, (int)length);
            }
            return data;
        }

        public Dictionary<string, torch.Tensor> Load(string fileName, string addString = "")
        {
            Dictionary<string, torch.Tensor> result = new Dictionary<string, torch.Tensor>();
            List<CommonTensor> tensorInfos = ReadTensorsInfoFromFile(fileName);

            foreach (CommonTensor tensorInfo in tensorInfos)
            {
                if (string.IsNullOrEmpty(tensorInfo.Name)) continue;

                try
                {
                    byte[] bytes = ReadByteFromFile(tensorInfo);
                    torch.Tensor tensor = torch.empty(tensorInfo.Shape.ToArray(), dtype: tensorInfo.Type);
                    tensor.bytes = bytes;

                    string finalName = addString + tensorInfo.Name;
                    if (!result.ContainsKey(finalName)) result.Add(finalName, tensor);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to load tensor {tensorInfo.Name}: {ex.Message}");
                }
            }
            return result;
        }
    }
}
