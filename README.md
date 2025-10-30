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
// Create a yolo task.
YoloTask yoloTask = new YoloTask(taskType, numberClass, yoloType: yoloType, deviceType: deviceType, yoloSize: yoloSize, dtype: dtype, keyPointShape: keyPointShape);

// Load pre-trained model, if you don't want to load the model, you can skip this step.
yoloTask.LoadModel(preTrainedModelPath, skipNcNotEqualLayers: true);

// Train model
yoloTask.Train(rootPath, trainDataPath, valDataPath, outputPath: outputPath, imageSize: imageSize, batchSize: batchSize, epochs: epochs, imageProcessType: imageProcessType);

// Predict image, if the model is not trained or loaded, it will use random weight to predict.
List<YoloResult> predictResult = yoloTask.ImagePredict(predictImage, predictThreshold, iouThreshold);

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
