# YoloSharp ONNX Export — Output Format Guide / 输出格式使用指南

This document explains the exported ONNX models: what each output is, what every
dimension means, and how to post-process them in your own runtime.
本文档说明导出的 ONNX 模型：每个输出是什么、每个维度对应什么、以及如何在您自己的运行时里做后处理。

---

## 1. Export Modes / 导出模式

```csharp
// C# (library):
OnnxExporter.Export(yolo, new long[] { 1, 3, 640, 640 }, "best.onnx",
                    dynamicBatch: true);   // input batch dim declared -1
```

Training auto-emits the deployment ONNX beside `best.bin` / 训练自动产出部署onnx:

| File | Input | Works on / 可直接用于 |
|---|---|---|
| `best.onnx` | `[-1,3,H,W]` dynamic | **all backends**: OpenVINO (load-time reshape), ONNX Runtime / ML.NET (any N), TensorRT (convert4 builds a profiled dynamic engine, batch 1..128), libtorch |

> ONE file, all backends — no variant juggling. / 一个文件通吃全后端，旧变体已废除。

**nc must match your dataset** — the demo brings bundled-dataset presets:
`YoloSharpDemo Yolov8 Pose 640 0.45 10 coco8-pose` (nc=1) / `... coco128` (nc=80)
/ `... coco128-seg` / `...` (default dotav1, nc=15). The `onnxexport` parity mode
accepts nc directly: `onnxexport <yolo> <task> [imgsz] [nc] [--pt|--bin <path>]`.
/ **nc必须与数据集匹配**——demo内置数据集预设带出nc；onnxexport第4参直传nc。

---

## 2. Single-Output Layout (ultralytics convention) / 单输出布局（ultralytics惯例）

Input `images`: `[-1, 3, 640, 640]` (batch symbolic), float32, RGB, **/255
normalized** (do it before feeding; the graph does NOT normalize). Letterbox:
centered padding.

Single output `[batch, C, A]` — **only batch symbolic; C/A concrete dim_values**,
exactly what Netron shows for an official ultralytics export. **The DFL decode
(softmax + expectation), grid and stride composition are baked into the graph**:
channels 0..3 are decoded xyxy **in input pixels** — no softmax/decode needed in
your runtime. / 单输出：**DFL解码/网格/stride已烘焙进图**，ch0-3就是解码后的像素xyxy。

### Channel layout of `output0` / output0 通道布局

| Task | `output0` dims | dim1 channel order / 通道顺序 |
|---|---|---|
| Detection | `[batch, 4+nc, 8400]` | xyxy (4, decoded, pixels) ‖ class scores (nc, sigmoid applied) |
| OBB | `[batch, 4+nc+1, 8400]` | … ‖ angle (1, `(sigmoid−0.25)·π` decoded) / 追加角度通道 |
| Pose | `[batch, 4+nc+nk·nd, 8400]` | … ‖ keypoints (nk×nd, still head raw values — decode like python: `(v*2 + anchor − 0.5) * stride`, visibility via sigmoid when nd=3) / 追加关键点通道（原始值，按python公式解码） |
| Segmentation | `[batch, 4+nc+32, 8400]` + `output1` proto | … ‖ mask coefficients (32)；proto 单独走 `output1` `[1, 32, H/4, W/4]` |

Measured examples / 实测示例: dotav1-obb (nc=15) `[batch, 20, 8400]`,
coco128-detect (nc=80) `[batch, 84, 8400]`, coco128-seg `[batch, 116, 8400]`,
coco8-pose (nc=1, kpt 17×3) `[batch, 56, 8400]`, v9 6-branch `[batch, 4+nc, 16800]`.

### What each dim is / 各维含义

- **dim0 = batch** — symbolic (`batch` dim_param); any N at runtime. / 帧数，符号维
- **dim1 = channels** — see the table above. / 通道，见上表
- **dim2 = 8400 anchors** — the P3/P4/P5 grid: `80²+40²+20² = 6400+1600+400`
  (stride 8/16/32 at 640 input). Anchor order: P3 rows first, then P4, P5.
  / anchor 网格，P3 在前

---

## 3. Post-Processing Recipe / 后处理步骤

Decode is already in the graph — your runtime only filters and NMSes:
解码已在图内，运行时只需过滤与NMS：

```
1) Filter:   score = output0[n, 4+c, a] > conf          (already a probability)
2) Box:      xyxy = output0[n, 0..3, a]                 (already input-pixel coords)
3) Un-letterbox: subtract the centered pad, divide by the scale ratio
                (pad_l = (640 − new_w)/2, ratio = min(640/w, 640/h))
4) NMS per class (IoU 0.45 typical)
```

> Angle (OBB): channel `4+nc` is the decoded angle (`(sigmoid−0.25)·π`).
> Pose keypoints: decode `(v*2 + anchor − 0.5) * stride` per point (same as
> python's kpts_decode); visibility = sigmoid(v) when nd=3.
> / OBB角度已解码；pose关键点按python公式解码

Reference implementation: `Txm.AI/Yolo/OnnxPredictCore.cs` (C#, used by all
three backends) — feel free to copy. / 参考实现可直接抄

---

## 4. Equivalence with ultralytics exports / 与ultralytics导出等价性

| Output | Dims | Layout |
|---|---|---|
| `output0` | `[batch, 4 + nc (+1 obb) (+nk·nd pose) (+32 seg), 8400]` | xyxy(4, decoded) ‖ scores(nc) ‖ extras — same channel semantics as a genuine ultralytics export / 与官方导出同通道语义 |
| `output1` (seg only) | `[1, 32, H/4, W/4]` | proto / 原型 |

Verified point-wise against the TorchSharp forward on every architecture ×
task (MaxDiff 6e-5 random weights / ~1e-3 real BF16 weights), plus
ORT / OpenVINO / TensorRT round-trips. / 与TorchSharp逐点对拍+三后端往返验证。

> Historical note: an earlier iteration kept raw DFL (64ch) for FP16 accuracy.
> Reviewers preferred full ultralytics parity — the decode moved into the graph
> (commit 2bc1c36). / 历史注记：早期版本保留64通道原始DFL，应作者反馈改为图内解码

---

## 5. Quick-Sanity Values / 快速自检数值

Feed any exported model this way and expect / 按此喂图，应得：

```
input : RGB image letterboxed to 640×640 (centered pad 114,114,114), /255
scores: max ≈ your trained confidence on a known positive (e.g. 0.937)
boxes : pixel coords inside [−margin, 640+margin]; with a zeros input every
        anchor's box centers on its grid point (P3 first ≈ (4,4), P5 last ≈ (624,640))
output: frame-contiguous [N, C, 8400] — slice per frame: a-th anchor at
        [n, :, a]; N frames do NOT fold into dim2 / 帧连续布局，不会折进 anchor 维
```

## 6. FAQ

**Q: The model declares `['batch', 20, 8400]` — which dims are symbolic?**
A: Only dim0 (`batch`, a dim_param). Channel and anchor dims are concrete
dim_values (torch.onnx.export `dynamic_axes` convention) — same as what Netron
shows for an official ultralytics export. / 仅batch是符号维，其余为具体值。

**Q: Why do the boxes look decoded already?**
A: They are — the DFL softmax/expectation, grid and stride multiply are part of
the exported graph (ultralytics parity). Just un-letterbox and NMS. / 解码已进图，直接NMS即可

**Q: Can I feed dynamic batch?**
A: Yes, on every backend: OV (load-time reshape), ORT/ML.NET (any N), TRT
(convert4 profiles 1..128). / 全后端支持动态batch。

**Q: nc mismatched my dataset — shapes look wrong.**
A: nc drives the channel count (4+nc+extras). The demo takes the dataset into
account (coco8-pose→56, coco128→84/116); make sure your Config.NumberClass
matches your labels. / 通道数由nc决定——确保nc与数据集一致
