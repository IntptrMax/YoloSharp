# YoloSharp

**Train and run YOLO models in pure C# with TorchSharp.**  
No Python required — from training to inference, everything stays inside .NET.

## ✨ Features

- **100% C# implementation** – No Python environment, no extra dependencies.
- **Full pipeline support** – Train, validate, and predict with your own custom models.
- **Multiple YOLO versions** – Supports YOLOv5, YOLOv5u, YOLOv8, YOLOv11, and YOLOv12.
- **All task types** – Object detection, segmentation, oriented bounding boxes (OBB), pose estimation (keypoints), and classification.
- **Model sizes** – n/s/m/l/x variants available for every architecture.
- **Advanced preprocessing** – Built-in LetterBox and Mosaic4 data augmentation.
- **GPU‑accelerated NMS** – Non‑maximum suppression runs directly on GPU.
- **Pretrained models** – Load models from Ultralytics YOLO (v5/v8/v11) and converted YOLOv12 checkpoints.
- **Cross‑platform** – Works with .NET 6 and later.

## 🤔 Why YoloSharp?

- **No Python environment** – Say goodbye to Conda, pip, dependency hell, and version conflicts. Everything runs inside your existing .NET ecosystem.
- **Seamless integration** – Directly use C# data structures, LibTorch bindings (TorchSharp),OpenCV bindings (OpenCvSharp), and your .NET logging/tracing infrastructure.
- **Simplified deployment** – Package your trained model and inference logic into a single .NET application or container. No separate Python microservice needed.
- **Performance** – Harness GPU acceleration via LibTorch and TorchSharp with full control over memory and execution.
- **Productivity** – Train and validate models using the same language you use for the rest of your backend, desktop, or game logic. One language, one toolchain.
- **Cost‑effective** – Reduce operational overhead by eliminating Python runtimes in production.

Whether you're building a desktop application, a cloud service, or an edge device solution, YoloSharp keeps your stack consistent and maintainable.

## 🔥 Recent Updates

**2026/05/07**  
🚀 Added data augmentation: horizontal flip, vertical flip, RandomPerspective.  
🐛 Fixed Mosaic4 implementation.

**2026/03/26**  
🚀 Added training metrics curves.

**2026/03/06**  
🚀 Configurable training & prediction.  
🚀 More metrics for validation.

**2026/02/03**  
🚀 Early stopping.  
🚀 HSV transform.  
🚀 Training logs.

**2026/01/20**  
🚀 Mixed precision trainer (simple AMP).  
🚀 Tqdm support.  
🚀 BF16 precision.

## 📦 Download Pretrained Models

Get the official YOLO checkpoints below.

### Prediction Checkpoints

| model | n | s | m | l | x |
| --- | --- | --- | --- | --- | --- |
| yolov5 | [yolov5n](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov5n.bin) | [yolov5s](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov5s.bin) | [yolov5m](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov5m.bin) | [yolov5l](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov5l.bin) | [yolov5x](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov5x.bin) |
| yolov5u | [yolov5nu](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov5nu.bin) | [yolov5su](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov5su.bin) | [yolov5mu](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov5mu.bin) | [yolov5lu](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov5lu.bin) | [yolov5xu](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov5xu.bin) |
| yolov8 | [yolov8n](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov8n.bin) | [yolov8s](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov8s.bin) | [yolov8m](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov8m.bin) | [yolov8l](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov8l.bin) | [yolov8x](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov8x.bin) |
| yolov11 | [yolov11n](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov11n.bin) | [yolov11s](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov11s.bin) | [yolov11m](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/yolov11m.bin) | [yolov11l](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov11l.bin) | [yolov11x](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov11x.bin) |

### Segmentation Checkpoints

| model | n | s | m | l | x |
| --- | --- | --- | --- | --- | --- |
| yolov8 | [yolov8n-seg](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov8n-seg.bin) | [yolov8s-seg](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov8s-seg.bin) | [yolov8m-seg](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov8m-seg.bin) | [yolov8l-seg](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov8l-seg.bin) | [yolov8x-seg](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov8x-seg.bin) |
| yolov11 | [yolov11n-seg](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov11n-seg.bin) | [yolov11s-seg](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov11s-seg.bin) | [yolov11m-seg](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov11m-seg.bin) | [yolov11l-seg](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov11l-seg.bin) | [yolov11x-seg](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov11x-seg.bin) |

## 🚀 Getting Started

### Install from NuGet

```bash
dotnet add package IntptrMax.YoloSharp
```

> [!NOTE]
> You also need to add one of the LibTorch packages (version 2.5.1.0) and `OpenCvSharp4.runtime`:
> - `libtorch-cpu`
> - `libtorch-cuda-12.1`
> - `libtorch-cuda-12.1-win-x64`
> - `libtorch-cuda-12.1-linux-x64`

### Basic Usage

```csharp
string preTrainedModelPath = @"..\..\..\Assets\PreTrainedModels\yolov8n-obb.bin"; // Pretrained model path.
string predictImagePath = @"..\..\..\Assets\TestImage\trucks.jpg";
string dataRootPath = @"..\..\..\Assets\datasets\dotav1";

string trainDataPath = @"train.txt";
string valDataPath = @"val.txt";

Mat predictImage = Cv2.ImRead(predictImagePath);

// Create a Yolo config
Config config = new Config
{
    DeviceType = DeviceType.CUDA,
    ScalarType = ScalarType.BFloat16,
    RootPath = dataRootPath,
    TrainDataPath = trainDataPath,
    ValDataPath = valDataPath,
    YoloType = YoloType.Yolov8,
    YoloSize = YoloSize.n,
    TaskType = TaskType.Obb,
    ImageProcessType = ImageProcessType.Mosiac,
    ImageSize = 640,
    BatchSize = 16,
    NumberClass = 15,
    PredictThreshold = 0.3f,
    IouThreshold = 0.7f,
    Workers = 4,
    Epochs = 100,
    LearningRate = 1e-4f,
    Patience = 50,
    KeyPoint_Num = 21,
    KeyPoint_Dim = 3,
};

// Create a yolo task.
YoloTask yoloTask = new YoloTask(config);

// Load pre-trained model. If you don't want to use pre-trained model, skip the step.
yoloTask.LoadModel(preTrainedModelPath, skipNcNotEqualLayers: true);

// Train model
yoloTask.Train();

// Predict image, if the model is not trained or loaded, it will use random weight to predict.
List<YoloResult> predictResult = yoloTask.ImagePredict(predictImage);
```

## 📸 Sample Results

| Model | Output |
|-------|--------|
| YOLOv8n (detection) | ![zidane](https://raw.githubusercontent.com/IntptrMax/YoloSharp/refs/heads/master/Assets/zidane.jpg) |
| YOLOv8n‑seg | ![bus](https://raw.githubusercontent.com/IntptrMax/YoloSharp/refs/heads/master/Assets/bus.jpg) |
| YOLOv8n‑obb | ![trucks](https://raw.githubusercontent.com/IntptrMax/YoloSharp/refs/heads/master/Assets/trucks.jpg) |
| YOLOv8n‑pose | ![tennis](https://raw.githubusercontent.com/IntptrMax/YoloSharp/refs/heads/master/Assets/tennis.jpg) |

---

**Enjoy YOLO entirely in .NET – no Python needed!**  
Contributions and feedback are welcome!