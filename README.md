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

## Models

You can download yolov5/yolov8 pre-trained models here.

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

In your code you can use it as below.

### Predict

You can use it with the code below:

    Bitmap inputBitmap = new Bitmap(predictImagePath);

    // Create predictor
    Predictor predictor = new Predictor(sortCount, yoloType: yoloType, deviceType: deviceType, yoloSize: yoloSize, dtype: dtype);

    // Train model
    predictor.LoadModel(preTraindModelPath);
    predictor.Train(trainDataPath, valDataPath, outputPath: outputPath, batchSize: batchSize, epochs: epochs);

    // Predict image
    predictor.LoadModel(Path.Combine(outputPath, "best.bin"));

    var results = predictor.ImagePredict(inputBitmap, predictThreshold, iouThreshold);

Use yolov5n pre-trained model to detect.

![image](https://github.com/user-attachments/assets/d32f7805-9f98-4530-bda6-43630c765159)

### Segment

You can use it with the code below:

    // Create segmenter
    Segmenter segmenter = new Segmenter(sortCount, yoloType: yoloType, deviceType: deviceType, yoloSize: yoloSize, dtype: dtype);

    // Train model
    segmenter.LoadModel(preTraindModelPath);
    segmenter.Train(trainDataPath, valDataPath, outputPath: outputPath, batchSize: batchSize, epochs: epochs, useMosaic: false);

    // Segment image
    segmenter.LoadModel("output/best.bin");
    Bitmap testBitmap = new Bitmap(predictImagePath);
    var (predictResult, bitmap) = segmenter.ImagePredict(testBitmap, predictThreshold, iouThreshold);

Use yolov8n-seg pre-trained model to detect.

![pred_seg](https://github.com/user-attachments/assets/898f4e75-e99d-434a-b910-1d87aabe4cb0)
