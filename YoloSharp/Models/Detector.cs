using Data;
using System.Text;
using TorchSharp;
using Utils;
using YoloSharp.Types;
using YoloSharp.Utils;

namespace YoloSharp.Models
{
    internal class Detector : YoloBaseTaskModel
    {
        internal Detector(Config config)
        {
            this.config = config;
            yolo = config.YoloType switch
            {
                YoloType.Yolov5u => new Yolo.Yolov5u(this.config.NumberClass, end2end: config.End2End, yoloSize: this.config.YoloSize, device: this.config.Device, dtype: this.config.Dtype),
                YoloType.Yolov8 => new Yolo.Yolov8(this.config.NumberClass, end2end: config.End2End, yoloSize: this.config.YoloSize, device: this.config.Device, dtype: this.config.Dtype),
                YoloType.Yolov9 => new Yolo.Yolov9(this.config.NumberClass, end2end: config.End2End, yoloSize: this.config.YoloSize, device: this.config.Device, dtype: this.config.Dtype),
                YoloType.Yolov11 => new Yolo.Yolov11(this.config.NumberClass, end2end: config.End2End, yoloSize: this.config.YoloSize, device: this.config.Device, dtype: this.config.Dtype),
                // [Fix 2026-11-06] Pass YoloSize through — the default parameter (YoloSize.n) silently
                // discarded the UI's size choice: a "s" training built an n skeleton (first conv 16ch
                // vs pretrained 32ch), 504/741 keys mismatched, A2C2f cores stayed random → garbage
                // bottom-edge boxes + scores stuck at ~0.81.
                // 传递YoloSize——默认参数(n)把UI选的s档丢了：s训练建成n骨架，504键错配，
                // A2C2f全随机→图像下方16个假框+分数0.81。
                YoloType.Yolov12 => new Yolo.Yolov12(this.config.NumberClass, end2end: config.End2End, yoloSize: this.config.YoloSize, device: this.config.Device, dtype: this.config.Dtype),
                _ => throw new NotImplementedException(),
            };
            // [Fix 2026-11-04] v9 training yaml uses the v8-style Detect head (6 groups in ONE
            // head, single TAL loss) — same loss as v8, no dual weighting. The earlier dual-head
            // experiment targeted DualDDetect (wrong reference; that's converted.pt's inference
            // checkpoint, not the training yaml). Stride must cover all 6 feature maps
            // (aux P3-P5 + main P3-P5).
            // v9训练yaml用v8式Detect单头六组+单TAL损失；stride覆盖6层特征图。
            int[] lossStride = config.YoloType == YoloType.Yolov9 ? new int[] { 8, 16, 32, 8, 16, 32 } : null;
            loss = config.End2End
                ? new Loss.E2EDetectLoss(this.config.NumberClass, device: this.config.Device, dtype: this.config.Dtype)
                : new Utils.Loss.v8DetectionLoss(this.config.NumberClass, stride: lossStride, device: this.config.Device, dtype: this.config.Dtype);
        }

        internal override List<YoloResult> ImagePredict(torch.Tensor orgImage, float predictThreshold, float iouThreshold)
        {
            using (torch.no_grad())
            {
                // [Fix 2026-10-27] Inference preprocessing aligned with training/val (aspect-ratio letterbox).
                // [Fix 2026-10-29] Padding must be CENTERED to match the training/val pipeline (Augment.LetterboxImage/
                // RectangleImage pad_l=(w-new_w)/2, pad_u=(h-new_h)/2). The 10-27 version padded to bottom-right
                // (0,padW,0,padH): for extreme aspect-ratio images (groove bar 1190x256) the content band sits
                // ~250px off vertically vs training distribution → confidence collapse → 0 detections.
                // 推理预处理与训练/val对齐（等比缩放letterbox）。
                // [Fix 2026-10-29] 填充必须居中才与训练/val管线一致（Augment.LetterboxImage/RectangleImage
                // 的pad_l=(w-new_w)/2、pad_u=(h-new_h)/2）。10-27版按右下角填充(0,padW,0,padH)：
                // 宽高比悬殊图（沟槽棒1190x256）内容带与训练分布垂直错位约250px→置信度崩塌→0目标。
                orgImage = orgImage.to(config.Dtype, config.Device);
                // 兼容CHW(3维)与NCHW(4维)入参：read_image产物是CHW——统一按最后两维取H/W
                int dimC = (int)orgImage.dim();
                int orgH = (int)orgImage.shape[dimC - 2];
                int orgW = (int)orgImage.shape[dimC - 1];
                float ratio = Math.Min(config.ImageSize / (float)orgH, config.ImageSize / (float)orgW);
                int rsH = (int)(ratio * orgH);
                int rsW = (int)(ratio * orgW);
                var batched = orgImage.dim() == 3 ? orgImage.unsqueeze(0) : orgImage;
                using (var resized = torch.nn.functional.interpolate(batched,
                    new long[] { rsH, rsW }, mode: torch.InterpolationMode.Bilinear, align_corners: false))
                {
                    // Centered padding to 32-multiples (same as training Letterbox) / 居中填充到32倍数（与训练Letterbox一致）
                    int padH = (32 - rsH % 32) % 32;
                    int padW = (32 - rsW % 32) % 32;
                    int padU = padH / 2;
                    int padD = padH - padU;
                    int padL = padW / 2;
                    int padR = padW - padL;
                    torch.Tensor input = torch.nn.functional.pad(resized, new long[] { padL, padR, padU, padD }, PaddingModes.Zeros, 114) / 255.0f;
                    yolo.eval();

                    Dictionary<string, torch.Tensor> inference = yolo.forward(input)?.inference;

                    torch.Tensor nms_results = Ops.non_max_suppression(inference["boxes"], predictThreshold, iouThreshold, end2end: config.End2End).output[0];
                    List<YoloResult> results = new List<YoloResult>();
                for (int i = 0; i < nms_results.shape[0]; i++)
                {
                    int x = nms_results[i][0].ToInt32();
                    int y = nms_results[i][1].ToInt32();
                    int rw = nms_results[i][2].ToInt32() - x;
                    int rh = nms_results[i][3].ToInt32() - y;

                    float score = nms_results[i][4].ToSingle();
                    int sort = nms_results[i][5].ToInt32();

                    // [Fix] 坐标反映射：网络输入(居中letterbox)→原图坐标系——先减去居中偏移(padL/padU)再除以ratio还原尺度
                    // [Fix] Coordinate back-mapping: network input (centered letterbox) → original image coords —
                    // subtract the centered offset (padL/padU) first, then divide by ratio to restore scale.
                    // [Fix 2026-11-08] Pad-region filter (python scale_boxes semantics): the gray border
                    // fires phantom high scores (unseen in training); drop boxes whose center falls
                    // outside the valid content area instead of emitting them below/beside the image.
                    // pad区框过滤：灰边高分框回映射后落在原图区域外——按框中心丢弃。
                    float cX = x + rw / 2f, cY = y + rh / 2f;   // 输入域中心
                    if (cX < padL || cY < padU || cX > padL + rsW || cY > padU + rsH) continue;
                    results.Add(new YoloResult()
                    {
                        ClassID = sort,
                        Score = score,
                        CenterX = (int)((x + rw / 2 - padL) / ratio),
                        CenterY = (int)((y + rh / 2 - padU) / ratio),
                        Width = (int)(rw / ratio),
                        Height = (int)(rh / ratio)
                    });
                    }
                    return results;
                }
            }
        }

        internal override (float[] loss, float[] metrics) Val(YoloDataLoader valDataLoader, AMPWrapper amp, int epoch)
        {
            string desc = GetValDescription();
            using (Tqdm<Dictionary<string, torch.Tensor>> pbar = new Tqdm<Dictionary<string, torch.Tensor>>(valDataLoader, desc: desc, total: (int)valDataLoader.Count, barStyle: Tqdm.BarStyle.Classic, barColor: Tqdm.BarColor.White, barWidth: 10, showPartialChar: true))
            using (torch.no_grad())
            {
                yolo.eval();
                torch.Tensor loss_items = torch.empty(0);
                long count = 0;
                List<torch.Tensor> tpList = new List<torch.Tensor>();
                List<torch.Tensor> pred_scoresList = new List<torch.Tensor>();
                List<torch.Tensor> pred_classesList = new List<torch.Tensor>();
                List<torch.Tensor> true_classesList = new List<torch.Tensor>();
                foreach (Dictionary<string, torch.Tensor> data in pbar)
                {
                    using (torch.NewDisposeScope())
                    {
                        if (data["batch_idx"].NumberOfElements < 1)
                        {
                            continue;
                        }
                        yolo.eval();
                        (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds) = amp.Evaluate(data["images"].to(config.Dtype)).Value;
                        torch.Tensor loss_detach = loss.forward(preds, data).loss_detach;

                        // [Fix 2026-11-05] Val conf 0.001 (python val.py): early epochs score low and
                        // the old 0.1 threshold filtered everything → mAP50 stayed 0.000 for the
                        // first ~9 epochs (looked stalled). python's 0.001 sees them from epoch 1.
                        // Val阈值对齐python(0.001)：早期低分全被0.1滤掉→前9轮mAP恒0。
                        List<torch.Tensor> nms_results = Ops.non_max_suppression(inference["boxes"], nc: config.NumberClass, conf_thres: 0.001f, iou_thres: 0.7f, end2end: config.End2End).output;
                        float w = data["images"].shape[data["images"].shape.Length - 1];
                        float h = data["images"].shape[data["images"].shape.Length - 2];
                        torch.Tensor scale = torch.tensor(new float[] { w, h, w, h }, device: new torch.Device(data["images"].device_type));
                        for (int i = 0; i < nms_results.Count; i++)
                        {
                            torch.Tensor pred_bboxes = nms_results[i][torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(0, 4)];
                            torch.Tensor pred_scores = nms_results[i][torch.TensorIndex.Ellipsis, 4];
                            torch.Tensor pred_classes = nms_results[i][torch.TensorIndex.Ellipsis, 5];

                            torch.Tensor batch_idx = data["batch_idx"].squeeze(-1) == i;
                            torch.Tensor turn_classes = data["cls"][batch_idx].squeeze(-1);
                            torch.Tensor batch_bbox = data["bboxes"][batch_idx] * scale;

                            batch_bbox = Ops.xywh2xyxy(batch_bbox);
                            torch.Tensor iou = Metrics.box_iou(batch_bbox, pred_bboxes);
                            torch.Tensor tp_epoch = match_predictions(pred_classes, turn_classes, iou);
                            tpList.Add(tp_epoch.MoveToOuterDisposeScope());
                            pred_scoresList.Add(pred_scores.MoveToOuterDisposeScope());
                            pred_classesList.Add(pred_classes.MoveToOuterDisposeScope());
                            true_classesList.Add(turn_classes.MoveToOuterDisposeScope());
                        }

                        if (loss_items.NumberOfElements < 1)
                        {
                            loss_items = torch.zeros_like(loss_detach);
                        }
                        loss_items = loss_items + loss_detach.to(loss_items.dtype, loss_items.device);
                        loss_items = loss_items.MoveToOuterDisposeScope();
                        count += data["images"].shape[0];
                    }
                }

                torch.Tensor tp_total = torch.cat(tpList);
                torch.Tensor scores_total = torch.cat(pred_scoresList);
                torch.Tensor pred_classes_total = torch.cat(pred_classesList);
                torch.Tensor true_classes_total = torch.cat(true_classesList);
                (torch.Tensor tp, torch.Tensor fp, torch.Tensor p, torch.Tensor r, torch.Tensor f1, torch.Tensor ap, torch.Tensor unique_class, torch.Tensor p_curve, torch.Tensor r_curve, torch.Tensor f1_curve, torch.Tensor x, torch.Tensor prec_values) = Metrics.ap_per_class(tp_total, scores_total, pred_classes_total, true_classes_total);

                float R = r.mean().ToSingle();
                float P = p.mean().ToSingle();
                float mAP50 = ap[torch.TensorIndex.Ellipsis, 0].mean().ToSingle();
                float mAP50_95 = ap[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(1)].mean().ToSingle();

                StringBuilder resultBuilder = new StringBuilder();
                resultBuilder.AppendFormat("{0,10}", "All");
                resultBuilder.AppendFormat("{0,10}", count);
                resultBuilder.AppendFormat("{0,10}", true_classes_total.shape[0]);
                resultBuilder.AppendFormat("{0,10}", P.ToString("0.000"));
                resultBuilder.AppendFormat("{0,10}", R.ToString("0.000"));
                resultBuilder.AppendFormat("{0,10}", mAP50.ToString("0.000"));
                resultBuilder.AppendFormat("{0,10}", mAP50_95.ToString("0.000"));

                Console.WriteLine(resultBuilder.ToString());

                return (loss_items.@float().data<float>().ToArray(), new float[] { P, R, mAP50, mAP50_95 });
            }
        }

        internal override float GetFitness(float[] metrics)
        {
            // metrics = [P, R, mAP50, mAP50-95]
            return 0.1f * metrics[2] + 0.9f * metrics[3];
        }

        internal override string GetTrainDescription()
        {
            string[] strs = new string[] { "Epoch", "box_loss", "cls_loss", "dfl_loss", "Instances", "Size" };
            StringBuilder stringBuilder = new StringBuilder();
            foreach (string str in strs)
            {
                stringBuilder.AppendFormat("{0,10}", str);
            }
            return stringBuilder.ToString();
        }

        internal override string GetValDescription()
        {
            string[] strs = new string[] { "Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-90)" };
            StringBuilder stringBuilder = new StringBuilder();
            foreach (string str in strs)
            {
                stringBuilder.AppendFormat("{0,10}", str);
            }
            return stringBuilder.ToString();
        }

        internal override string GetSeperatLogHeaders()
        {
            return "Epoch, Time, train/box_loss, train/cls_loss, train/dfl_loss, val/box_loss, val/cls_oss, val/dfl_loss, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B), train/loss, val/loss";
        }

        internal void DrawPlots(torch.Tensor p_curve, torch.Tensor r_curve, torch.Tensor f1_curve, torch.Tensor x, torch.Tensor prec_values)
        {

        }
    }
}
