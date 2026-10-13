using System.Collections;
using System.Collections.Generic;

namespace YoloSharp.ModelLoader.PtReader
{
    public static class ModelWalker
    {
        public static Dictionary<string, TorchTensor> CollectTensors(object? root, string rootName = "")
        {
            var result = new Dictionary<string, TorchTensor>();
            Walk(root, rootName, result);
            return result;
        }

        private static readonly HashSet<string> SkipStateKeys = new HashSet<string>
        {
            "training", "inplace",
            "yaml", "yaml_file", "nc", "args",
            "anchors", "strides", "shape", "names",
            "i", "f", "type",
        };

        private static void Walk(object? obj, string prefix, Dictionary<string, TorchTensor> result)
        {
            switch (obj)
            {
                case null: return;
                case TorchTensor t:
                    if (!string.IsNullOrEmpty(prefix))
                        result[prefix.TrimEnd('.')] = t;
                    return;

                case PyObject py:
                    foreach (var key in new[] { "_modules", "_parameters", "_buffers" })
                        if (py.State.TryGetValue(key, out var sub) && sub != null)
                            WalkDict(sub, prefix, result);

                    foreach (var kv in py.State)
                    {
                        if (kv.Key is "_modules" or "_parameters" or "_buffers") continue;
                        if (SkipStateKeys.Contains(kv.Key)) continue;
                        if (kv.Key.StartsWith("_forward") ||
                            kv.Key.StartsWith("_backward") ||
                            kv.Key.StartsWith("_load_state") ||
                            kv.Key.StartsWith("_non_persistent")) continue;
                        Walk(kv.Value, Join(prefix, kv.Key), result);
                    }
                    return;

                case Dictionary<string, object?> dict:
                    WalkDict(dict, prefix, result);
                    return;

                case IEnumerable en when obj is not string:
                    int i = 0;
                    foreach (var item in en) { Walk(item, Join(prefix, i.ToString()), result); i++; }
                    return;
            }
        }

        private static void WalkDict(object dictObj, string prefix, Dictionary<string, TorchTensor> result)
        {
            if (dictObj is Dictionary<string, object?> d)
                foreach (var kv in d)
                    Walk(kv.Value, Join(prefix, kv.Key), result);
        }

        private static string Join(string prefix, string key)
            => string.IsNullOrEmpty(prefix) ? key : $"{prefix}.{key}";
    }
}