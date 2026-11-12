# YoloSharp ONNX Export — Output Format Guide / 输出格式使用指南

This document explains the exported ONNX models: what each output is, what every
dimension means, and how to post-process them in your own runtime.
本文档说明导出的 ONNX 模型：每个输出是什么、每个维度对应什么、以及如何在您自己的运行时里做后处理。

---

## 1. Export Modes / 导出模式

```csharp
// C# (library):
OnnxExporter.Export(yolo, new long[] { 1, 3, 640, 640 }, "best.onnx",
                    dynamicBatch: false,   // true = input batch dim declared -1 (TRT profile)
                    pythonStyle: false);   // true = single "output0" (ultralytics layout)

// Platform CLI:
//   Txm.AIConsoleTest.exe convert best.bin onnx           → multi-head (default)
//   Txm.AIConsoleTest.exe convert best.bin onnx pystyle   → python-style single output
```

Training auto-emits one onnx beside `best.bin` / 训练自动产出一个onnx:

| File | Input | Intended backend / 用途 |
|---|---|---|
| `best.onnx` | `[1,3,H,W]` static | OpenVINO (runtime reshape → dynamic batch), ONNX Runtime, libtorch |
| `best_deploy_dyn.onnx` | `[-1,3,H,W]` dynamic | TensorRT dynamic engine (profile 1..128) |

> NOTE `best_deploy_dyn.onnx` is the ONLY variant TensorRT can build into a
> dynamic-batch engine — python-exported static-trace graphs cannot be profiled
> (inner reshapes/constants are traced to batch=1). / 动态变体是 TRT 动态
> engine 的唯一可用形态。

---

## 2. Single-Output Layout (both files) / 单输出布局(两个文件通用)

Input `images`: `[N, 3, 640, 640]`, float32, RGB, **/255 normalized** (do it
before feeding; the graph does NOT normalize). Letterbox: centered padding.

Single output (dims symbolic; actual at N=1, nc=1 in brackets) / 单输出(维度为符号维):

### Detection

| Output | Dims | Meaning |
|---|---|---|
| `output0` | `[N, 64+nc, 8400]` | channels `0..63` = **raw DFL distributions — NOT coordinates**; channels `64..` = class probabilities (**sigmoid already applied**) / ch0-63=DFL原始分布(不是坐标), ch64+=已sigmoid类别概率 |

### OBB (rotated boxes) adds / OBB 任务额外输出

| Output | Dims | Meaning |
|---|---|---|
| `output_angle` | `[N, 1, 8400]` | Per-anchor angle logits / 每 anchor 一个角度 logit |

### Segmentation adds / 分割任务额外输出

| Output | Dims | Meaning |
|---|---|---|
| `output_masks` | `[N, 32, 8400]` | Mask coefficients (32 per anchor) / mask 系数 |
| `output_proto` | `[N, 32, H/4, W/4]` | Mask prototypes (P3-level) / mask 原型 |

### Pose adds / 姿态任务额外输出

| Output | Dims | Meaning |
|---|---|---|
| `output_kpts` | `[N, nk*nd, 8400]` | Keypoints (nk points × nd dims: 2=x,y / 3=x,y,visible) / 关键点 |

### What each dim is / 各维含义

- **dim0 = N** — batch (frames). Static export declares 1; OV reshapes to -1 at
  load; TRT dyn variant declares it symbolic. / 帧数
- **dim1 = channels** — see per-output table above (`output_scores` dim1 = nc,
  `output_angle` dim1 = 1, `output_boxes` dim1 = 4×regMax=64). / 通道
- **dim2 = 8400 anchors** — the P3/P4/P5 grid: `80²+40²+20² = 6400+1600+400`
  (stride 8/16/32 at 640 input). Anchor order: P3 rows first, then P4, P5.
  / anchor 网格，P3 在前
- **dim3 (proto only)** — spatial H/4 × W/4. / proto 的空间维

---

## 3. Post-Processing Recipe / 后处理步骤

Scores threshold → DFL decode → grid + stride → xywh → map to original image → NMS.

```
1) Filter:   score = output_scores[n, c, a] > conf        (already a probability)
2) DFL:      for each anchor, take 64 channels = 4 edges × 16 bins
             each 16-bin group: softmax → dot([0..15]) → expected distance d_i
             box (ltrb in stride units): [−d0, −d1, d2, d3]  (order: l,t,r,b)
3) Grid:     anchor a → (gx, gy) on its level; stride s ∈ {8,16,32}
             xyxy = ((gx + 0.5 − d0·?) ...)  — equivalently
             x1,y1 = (gx − d0, gy − d1)·s ; x2,y2 = (gy + d2, gy + d3)·s
4) Un-letterbox: subtract the centered pad, divide by the scale ratio
                (pad_l = (640 − new_w)/2, ratio = min(640/w, 640/h))
5) NMS per class (IoU 0.45 typical)
```

> Angle (OBB): `output_angle` is a raw logit (multi-class angle bins when
> angle_bins > 1). Decode: softmax over bins → expectation × π / bins, or
> `sigmoid × π` for the continuous single-channel case — match your training
> config. / 角度按训练配置的 bins 解码

Reference implementation: `Txm.AI/Yolo/OnnxPredictCore.cs` (C#, used by all
three backends) — feel free to copy. / 参考实现可直接抄

---

## 4. Python-Style Single Output / python 风格单输出（pystyle）

`pythonStyle: true` (or CLI `convert <bin> onnx pystyle`) merges the per-anchor
heads into one tensor, ultralytics layout / 将各头沿通道拼接成单输出：

| Output | Dims | Layout |
|---|---|---|
| `output0` | `[N, 64 + nc (+1 obb) (+nk·nd pose), 8400]` | boxes(64) ‖ scores(nc) ‖ angle/kpts — dim1 channel order, NOT decoded boxes / 通道顺序拼接，boxes 仍是 DFL 原始 64 通道 |
| `output1` (seg only) | `[N, 32, H/4, W/4]` | proto (same as `output_proto`) |

Example (detection, nc=1): `output0 = [1, 65, 8400]` → channels `0..63` =
boxes DFL, channel `64` = score. Verified identical values to the multi-head
export / 与多头版数值完全一致。

> Difference from a genuine ultralytics export: their `output0` carries
> **decoded xywh** (4 channels, decode baked into the graph). Ours keeps raw
> DFL (64 channels) — better FP16/quantization accuracy, same anchor layout,
> one extra softmax step in your post-processing. / 与 ultralytics 原版差异：
> 原版已解码(4通道)，我们保留原始 DFL(64通道)，后处理多一步 softmax

---

## 5. Quick-Sanity Values / 快速自检数值

Feed any exported model this way and expect / 按此喂图，应得：

```
input : RGB image letterboxed to 640×640 (centered pad 114,114,114), /255
scores: max ≈ your trained confidence on a known positive (e.g. 0.937)
boxes : max < regMax (=16), because they are pre-softmax distributions
output: frame-contiguous [N, C, 8400] — slice per frame: a-th anchor at
        [n, :, a]; N frames do NOT fold into dim2 / 帧连续布局，不会折进 anchor 维
```

## 6. FAQ

**Q: Why is `output_boxes` 64 channels instead of 4?**
A: It's DFL: 4 edges × 16 distribution bins. Decode with softmax+expectation
(step 2 above). Keeping this out of the graph preserves FP16 accuracy under
TensorRT and lets all runtimes share one exact decoder. / DFL 分布，图外解码保精度

**Q: The model file declares `['batch','dim1','dim2']` — is that an error?**
A: No — fully-dynamic ranked shape (torch.onnx.export convention). Actual
shapes are resolved at inference. / 符号维声明，推理时解析

**Q: Can I feed dynamic batch?**
A: OV: yes (we reshape at load). TRT: use the `_deploy_dyn.onnx` variant with
a profile. ORT: any N works with the dyn variant. / 见第 1 节表格
