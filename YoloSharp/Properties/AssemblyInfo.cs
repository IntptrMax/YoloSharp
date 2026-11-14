using System.Runtime.CompilerServices;

// Demo parity tests need access to internal export entries (GetYoloModule/Export are public but the Yolov8 type is internal)
// Demo对拍测试需要访问internal导出入口（GetYoloModule/Export为public但Yolov8类型为internal）
[assembly: InternalsVisibleTo("YoloSharpDemo")]
[assembly: InternalsVisibleTo("YoloSharpNet48Demo")]
