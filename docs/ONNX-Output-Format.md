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

> ONE file, all backends — no variant juggling. The earlier multi-head /
> `best_deploy_dyn.onnx` variants are retired. / 一个文件通吃全后端，旧变体已废除。

---

## 2. Single-Output Layout / 单输出布局

Input `images`: `[-1, 3, 640, 640]` (batch symbolic), float32, RGB, **/255
normalized** (do it before feeding; the graph does NOT normalize). Letterbox:
centered padding.

Single output, python-convention declaration `[batch, C, A]` (only batch
symbolic; C/A concrete, e.g. `[batch, 65, 8400]` at nc=1). **Every task packs
into ONE `output0`** — per-task channel layout below / 各任务全部打包进唯一
`output0`，按任务区分通道布局：

### Channel layout of `output0` / output0 通道布局

| Task | `output0` dims | dim1 channel order / 通道顺序 |
|---|---|---|
| Detection | `[batch, 64+nc, 8400]` | boxes DFL (64, raw distribution) ‖ class scores (nc, sigmoid applied) |
| OBB | `[batch, 64+nc+1, 8400]` | … ‖ angle (1) / 追加角度通道 |
| Pose | `[batch, 64+nc+nk·nd, 8400]` | … ‖ keypoints (nk×nd) / 追加关键点通道 |
| Segmentation | `[batch, 64+nc+32, 8400]` + `output1` proto | … ‖ mask coefficients (32)；proto 单独走 `output1` `[batch, 32, H/4, W/4]` |

> channels `0..63` are the **raw DFL distributions — NOT coordinates**; decode
> at runtime (see §3). Class scores already went through sigmoid.
> / ch0-63=DFL原始分布(不是坐标，运行时解码)；类别概率已过sigmoid。

### What each dim is / 各维含义

- **dim0 = batch** — symbolic (`batch` dim_param); any N at runtime. / 帧数，符号维
- **dim1 = channels** — see the table above. / 通道，见上表
- **dim2 = 8400 anchors** — the P3/P4/P5 grid: `80²+40²+20² = 6400+1600+400`
  (stride 8/16/32 at 640 input). Anchor order: P3 rows first, then P4, P5.
  / anchor 网格，P3 在前

---

## 3. Post-Processing Recipe / 后处理步骤

Scores threshold → DFL decode → grid + stride → xywh → map to original image → NMS.

```
1) Filter:   score = output0[n, 64+c, a] > conf         (already a probability)
2) DFL:      for each anchor, take channels 0..63 = 4 edges × 16 bins
             each 16-bin group: softmax → dot([0..15]) → expected distance d_i
             box (ltrb in stride units): [−d0, −d1, d2, d3]  (order: l,t,r,b)
3) Grid:     anchor a → (gx, gy) on its level; stride s ∈ {8,16,32}
             xyxy = ((gx + 0.5 − d0·?) ...)  — equivalently
             x1,y1 = (gx − d0, gy − d1)·s ; x2,y2 = (gy + d2, gy + d3)·s
4) Un-letterbox: subtract the centered pad, divide by the scale ratio
                (pad_l = (640 − new_w)/2, ratio = min(640/w, 640/h))
5) NMS per class (IoU 0.45 typical)
```

> Angle (OBB): channel `64+nc` of `output0` is the decoded angle
> (`(sigmoid−0.25)·π`, single channel). / OBB角度通道已解码((sigmoid−0.25)·π)

Reference implementation: `Txm.AI/Yolo/OnnxPredictCore.cs` (C#, used by all
three backends) — feel free to copy. / 参考实现可直接抄

---

## 4. Layout vs. genuine ultralytics export / 与ultralytics原版导出的布局差异

All heads are merged into one tensor, ultralytics-style channel order /
各头沿通道拼接成单输出（与ultralytics同布局语义）：

| Output | Dims | Layout |
|---|---|---|
| `output0` | `[batch, 64 + nc (+1 obb) (+nk·nd pose), 8400]` | boxes(64) ‖ scores(nc) ‖ angle/kpts — dim1 channel order, NOT decoded boxes / 通道顺序拼接，boxes 仍是 DFL 原始 64 通道 |
| `output1` (seg only) | `[batch, 32, H/4, W/4]` | proto (same as before) / proto 原型 |

Example (detection, nc=1): `output0 = [batch, 65, 8400]` → channels `0..63` =
boxes DFL, channel `64` = score. Verified identical values across
ORT / OV / TRT. / 三后端数值一致。

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

**Q: Why are the box channels 64 wide instead of 4?**
A: It's DFL: 4 edges × 16 distribution bins. Decode with softmax+expectation
(step 2 above). Keeping this out of the graph preserves FP16 accuracy under
TensorRT and lets all runtimes share one exact decoder. / DFL 分布，图外解码保精度

**Q: The model declares `['batch', 65, 8400]` — which dims are symbolic?**
A: Only dim0 (`batch`, a dim_param). Channel and anchor dims are concrete
dim_values (torch.onnx.export `dynamic_axes` convention) — same as what Netron
shows for an official ultralytics export. / 仅batch是符号维，其余为具体值（与
官方ultralytics导出显示一致）。

**Q: Can I feed dynamic batch?**
A: Yes, on every backend: OV (load-time reshape), ORT/ML.NET (any N), TRT
(convert4 profiles 1..128). / 全后端支持动态batch。
