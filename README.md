# YoloSharp

Train Yolo model in C# with TorchSharp. </br>
With the help of this project you won't have to transform .pt model to onnx, and can train your own model in C#.

## Feature

- Written in C# only.
- Train and predict your own model.
- Support Yolov5, Yolov5u, Yolov8, Yolov11 and Yolov12 now.
- Support Predict and Segment now.
- Support n/s/m/l/x size.
- Support LetterBox and Mosaic4 method for preprocessing images.
- Support NMS with GPU.
- Support Load PreTrained models from ultralytics/yolov5/yolov8/yolo11 and yolov12(converted).
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
> Please add one of libtorch-cpu, libtorch-cuda-12.1, libtorch-cuda-12.1-win-x64 or libtorch-cuda-12.1-linux-x64 version 2.5.1.0 to execute.

You can use it with the code below:

### Predict



```CSharp
	SKBitmap predictImage = SKBitmap.Decode(predictImagePath);
	
	// Create predictor
	Predictor predictor = new Predictor(sortCount, yoloType: yoloType, deviceType: deviceType, yoloSize: yoloSize, dtype: dtype);
	predictor.LoadModel(preTrainedModelPath, skipNcNotEqualLayers: true);

	// Train model
	predictor.Train(trainDataPath, valDataPath, outputPath: outputPath, batchSize: batchSize, epochs: epochs, useMosaic: true);
	predictor.LoadModel(Path.Combine(outputPath, "best.bin"));

	// ImagePredict image
	List<Predictor.PredictResult> predictResult = predictor.ImagePredict(predictImage, predictThreshold, iouThreshold);
```
</br>
Use yolov5n pre-trained model to detect.

![image](https://raw.githubusercontent.com/IntptrMax/YoloSharp/refs/heads/master/Assets/zidane.jpg)

### Segment

```CSharp
	SKBitmap predictImage = SKBitmap.Decode(predictImagePath);

    //Create segmenter
	Segmenter segmenter = new Segmenter(sortCount, yoloType: yoloType, deviceType: deviceType, yoloSize: yoloSize, dtype: dtype);
	segmenter.LoadModel(preTrainedModelPath, skipNcNotEqualLayers: true);

	// Train model
	segmenter.Train(trainDataPath, valDataPath, outputPath: outputPath, batchSize: batchSize, epochs: epochs, useMosaic: false);
	segmenter.LoadModel(Path.Combine(outputPath, "best.bin"));

	// ImagePredict image
	var (predictResult, resultImage) = segmenter.ImagePredict(predictImage, predictThreshold, iouThreshold);

```

Use yolov8n-seg pre-trained model to detect.

![pred_seg](https://raw.githubusercontent.com/IntptrMax/YoloSharp/refs/heads/master/Assets/bus.jpg)
