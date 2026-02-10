# YoloSharp

Train Yolo model in C# with TorchSharp. <br/>
With the help of this project you won't have to transform .pt model to onnx, and can train your own model in C# and don't have to install python.

## Feature

- Written in C# only, don't have to install python.
- Train and predict your own model.
- Support Yolov5, Yolov5u, Yolov8, Yolov11 and Yolov12 now.
- Support Predict, Segment, Obb, Pose(Key Points) and Classification now.
- Support n/s/m/l/x size.
- Support LetterBox and Mosaic4 method for preprocessing images.
- Support NMS with GPU.
- Support Load PreTrained models from ultralytics yolov5/yolov8/yolo11 and yolov12(converted).
- Support .Net6 or higher.

## 🔥Important News  

**2026/02/03**  
  🚀 Add **Early Stop**  
  🚀 Add **HSV transform**  
  🚀 Add **Train Logs**

**2026/01/20**  
  🚀 YoloSharp support **Mixed Precision Trainer**  (simple amp)  
  🚀 **Tqdm** supported.  
  🚀 Add BF16 Precision.

## Models

You can download yolo pre-trained models here.  
<details>
  <summary>Prediction Checkpoints</summary>

| model | n| s | m | l | x |
| --- | ----------- | ----------- | ----------- | ----------- | ----------- |
| yolov5 | [yolov5n](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov5n.bin) | [yolov5s](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov5s.bin) | [yolov5m](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov5m.bin) | [yolov5l](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov5l.bin) | [yolov5x](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov5x.bin) |
| yolov5 | [yolov5nu](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov5nu.bin) | [yolov5su](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov5su.bin) | [yolov5mu](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov5mu.bin) | [yolov5lu](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov5lu.bin) | [yolov5xu](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov5xu.bin) |
| yolov8 | [yolov8n](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov8n.bin) | [yolov8s](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov8s.bin) | [yolov8m](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov8m.bin) | [yolov8l](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov8l.bin) | [yolov8x](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov8x.bin) |
| yolov11 | [yolov11n](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov11n.bin) | [yolov11s](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov11s.bin) | [yolov11m](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/yolov11m.bin) | [yolov11l](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov11l.bin) | [yolov11x](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov11x.bin) |

</details>

<details>
  <summary>Segmention Checkpoints</summary>

| model | n| s | m | l | x |
| --- | ----------- | ----------- | ----------- | ----------- | ----------- |
| yolov8 | [yolov8n](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov8n-seg.bin) | [yolov8s](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov8s-seg.bin) | [yolov8m](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov8m-seg.bin) | [yolov8l](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov8l-seg.bin) | [yolov8x](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov8x-seg.bin) |
| yolov11 | [yolov11n](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov11n-seg.bin) | [yolov11s](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov11s-seg.bin) | [yolov11m](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov11m-seg.bin) | [yolov11l](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov11l-seg.bin) | [yolov11x](https://github.com/IntptrMax/YoloSharp/releases/download/1.0.6/Yolov11x-seg.bin) |

</details>

## How to use

You can download the code or add it from nuget.

    dotnet add package IntptrMax.YoloSharp


> [!NOTE]
> Please add one of libtorch-cpu, libtorch-cuda-12.1, libtorch-cuda-12.1-win-x64 or libtorch-cuda-12.1-linux-x64 version 2.5.1.0 and OpenCvSharp4.runtime to execute.

You can use it with the code below:

### Yolo Task

```CSharp

string preTrainedModelPath = @"..\..\..\Assets\PreTrainedModels\yolov8n-obb.bin"; // Pretrained model path.
string predictImagePath = @"..\..\..\Assets\TestImage\trucks.jpg";

Mat predictImage = Cv2.ImRead(predictImagePath);

// Create a Yolo config
Config config = new Config
{
	DeviceType = DeviceType.CUDA,
	ScalarType = ScalarType.Float16,
	RootPath = @"..\..\..\Assets\DataSets\dotav1",
	TrainDataPath = "train.txt",
	ValDataPath = "val.txt",
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

<br/>

Use yolov8n pre-trained model to detect.

![image](https://raw.githubusercontent.com/IntptrMax/YoloSharp/refs/heads/master/Assets/zidane.jpg)

Use yolov8n-seg pre-trained model to detect.

![pred_seg](https://raw.githubusercontent.com/IntptrMax/YoloSharp/refs/heads/master/Assets/bus.jpg)

Use yolov8n-obb pre-trained model to detect.

![pred_seg](https://raw.githubusercontent.com/IntptrMax/YoloSharp/refs/heads/master/Assets/trucks.jpg)

Use yolov8n-pose pre-trained model to detect.

![pred_seg](https://raw.githubusercontent.com/IntptrMax/YoloSharp/refs/heads/master/Assets/tennis.jpg)
