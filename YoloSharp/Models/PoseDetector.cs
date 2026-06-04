using Data;
using System.Text;
using TorchSharp;
using Utils;
using YoloSharp.Types;
using YoloSharp.Utils;

namespace YoloSharp.Models
{
    internal class PoseDetector : YoloBaseTaskModel
    {
        private static float[] OKS_SIGMA
        {
            get
            {
                float[] v = new float[] { 0.26f, 0.25f, 0.25f, 0.35f, 0.35f, 0.79f, 0.79f, 0.72f, 0.72f, 0.62f, 0.62f, 1.07f, 1.07f, 0.87f, 0.87f, 0.89f, 0.89f };
                return v.Select(x => x / 10.0f).ToArray();
            }
        }

        internal PoseDetector(Config config)
        {
            this.config = config;

            yolo = config.YoloType switch
            {
                YoloType.Yolov5u => new Yolo.Yolov5uPose(config.NumberClass, kpt_num: config.KeyPoint_Num, kpt_dim: config.KeyPoint_Dim, yoloSize: config.YoloSize, end2end: config.End2End, device: config.Device, dtype: config.Dtype),
                YoloType.Yolov8 => new Yolo.Yolov8Pose(config.NumberClass, kpt_num: config.KeyPoint_Num, kpt_dim: config.KeyPoint_Dim, yoloSize: config.YoloSize, end2end: config.End2End, device: config.Device, dtype: config.Dtype),
                YoloType.Yolov11 => new Yolo.Yolov11Pose(config.NumberClass, kpt_num: config.KeyPoint_Num, kpt_dim: config.KeyPoint_Dim, yoloSize: config.YoloSize, end2end: config.End2End, device: config.Device, dtype: config.Dtype),
                YoloType.Yolov12 => new Yolo.Yolov12Pose(config.NumberClass, kpt_num: config.KeyPoint_Num, kpt_dim: config.KeyPoint_Dim, yoloSize: config.YoloSize, end2end: config.End2End, device: config.Device, dtype: config.Dtype),
                _ => throw new NotImplementedException("Yolo type not supported."),
            };
            loss = config.End2End ? new Loss.E2EPoseLoss(config.NumberClass, config.KeyPoint_Num, config.KeyPoint_Dim, device: config.Device, dtype: config.Dtype) : new Loss.v8PoseLoss(config.NumberClass, config.KeyPoint_Num, config.KeyPoint_Dim, device: config.Device, dtype: config.Dtype);

            //Tools.TransModelFromSafetensors(yolo, @".\yolov8n-pose.safetensors", @".\PreTrainedModels\yolov8n-pose.bin");
        }


        internal override List<YoloResult> ImagePredict(torch.Tensor orgImage, float predictThreshold, float iouThreshold)
        {
            using (torch.no_grad())
            {
                yolo.eval();
                // Change RGB → BGR
                orgImage = orgImage.to(config.Dtype, config.Device).unsqueeze(0);

                int w = (int)orgImage.shape[3];
                int h = (int)orgImage.shape[2];
                int padHeight = 32 - (int)(orgImage.shape[2] % 32);
                int padWidth = 32 - (int)(orgImage.shape[3] % 32);

                padHeight = padHeight == 32 ? 0 : padHeight;
                padWidth = padWidth == 32 ? 0 : padWidth;

                torch.Tensor input = torch.nn.functional.pad(orgImage, new long[] { 0, padWidth, 0, padHeight }, PaddingModes.Zeros, 114) / 255.0f;
                Dictionary<string, torch.Tensor> inference = yolo.forward(input)?.inference;
                List<torch.Tensor> nms_result = Ops.non_max_suppression(inference["boxes"], nc: config.NumberClass, conf_thres: predictThreshold, iou_thres: iouThreshold, end2end: config.End2End).output;
                List<YoloResult> results = new List<YoloResult>();
                if (nms_result.Count > 0)
                {
                    if (nms_result[0] is not null)
                    {
                        for (int i = 0; i < nms_result[0].shape[0]; i++)
                        {
                            int x = nms_result[0][i][0].ToInt32();
                            int y = nms_result[0][i][1].ToInt32();
                            int rw = nms_result[0][i][2].ToInt32() - x;
                            int rh = nms_result[0][i][3].ToInt32() - y;

                            YoloResult result = new YoloResult();

                            result.CenterX = x + rw / 2;
                            result.CenterY = y + rh / 2;
                            result.Width = rw;
                            result.Height = rh;

                            result.Score = nms_result[0][i][4].ToSingle();
                            result.ClassID = nms_result[0][i][5].ToInt32();
                            long keyPointsCount = (nms_result[0].shape[1] - 6) / config.KeyPoint_Dim;
                            KeyPoint[] keyPoints = new Types.KeyPoint[keyPointsCount];
                            for (int j = 0; j < keyPointsCount; j++)
                            {
                                keyPoints[j] = new KeyPoint()
                                {
                                    X = nms_result[0][i][6 + j * config.KeyPoint_Dim].ToSingle(),
                                    Y = nms_result[0][i][6 + j * config.KeyPoint_Dim + 1].ToSingle(),
                                    VisibilityScore = config.KeyPoint_Dim == 3 ? nms_result[0][i][6 + j * 3 + 2].ToSingle() : 2.0f
                                };
                            }
                            result.KeyPoints = keyPoints;

                            results.Add(result);
                        }
                    }
                }
                return results;
            }
        }

        internal override (float[] loss, float[] metrics) Val(YoloDataLoader valDataLoader, AMPWrapper amp, int epoch)
        {
            yolo.eval();
            string desc = GetValDescription();
            using (Tqdm<Dictionary<string, torch.Tensor>> pbar = new Tqdm<Dictionary<string, torch.Tensor>>(valDataLoader, desc: desc, total: (int)valDataLoader.Count, barStyle: Tqdm.BarStyle.Classic, barColor: Tqdm.BarColor.White, barWidth: 10, showPartialChar: true))
            using (torch.no_grad())
            {
                torch.Tensor loss_items = torch.empty(0);
                long count = 0;
                List<torch.Tensor> tpList = new List<torch.Tensor>();
                List<torch.Tensor> pred_scoresList = new List<torch.Tensor>();
                List<torch.Tensor> pred_classesList = new List<torch.Tensor>();
                List<torch.Tensor> true_classesList = new List<torch.Tensor>();
                List<torch.Tensor> ptpList = new List<torch.Tensor>();

                foreach (Dictionary<string, torch.Tensor> data in pbar)
                {
                    using (torch.NewDisposeScope())
                    {
                        if (data["batch_idx"].NumberOfElements < 1)
                        {
                            continue;
                        }
                        (Dictionary<string, torch.Tensor> inferenct, Dictionary<string, object> preds)? pred = amp.Evaluate(data["images"].to(config.Dtype));
                        torch.Tensor ls_item = loss.forward(pred?.preds, data).loss_detach;

                        float w = data["images"].shape[data["images"].shape.Length - 1];
                        float h = data["images"].shape[data["images"].shape.Length - 2];
                        torch.Tensor scale = torch.tensor(new float[] { w, h, w, h }, device: new torch.Device(data["images"].device_type));
                        List<torch.Tensor> nms_results = Ops.non_max_suppression((torch.Tensor)pred?.inferenct["boxes"], nc: config.NumberClass, conf_thres: 0.01f, iou_thres: 0.7f, end2end: config.End2End).output;

                        for (int i = 0; i < nms_results.Count; i++)
                        {
                            torch.Tensor pred_bboxes = nms_results[i][torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(0, 4)];
                            torch.Tensor pred_scores = nms_results[i][torch.TensorIndex.Ellipsis, 4];
                            torch.Tensor pred_classes = nms_results[i][torch.TensorIndex.Ellipsis, 5];
                            torch.Tensor pred_kpt = nms_results[i][torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(6)].view(new long[] { -1, config.KeyPoint_Num, config.KeyPoint_Dim });

                            torch.Tensor batch_idx = data["batch_idx"].squeeze(-1) == i;
                            torch.Tensor turn_classes = data["cls"][batch_idx].squeeze(-1);
                            torch.Tensor batch_bbox = data["bboxes"][batch_idx] * scale;
                            torch.Tensor batch_kpt = data["keypoints"][batch_idx];
                            if (batch_kpt.shape[batch_kpt.shape.Length - 1] == 2)
                            {
                                torch.Tensor seen = torch.ones(new long[] { batch_kpt.shape[0], batch_kpt.shape[1], 1 }, device: batch_kpt.device);
                                batch_kpt = torch.cat(new torch.Tensor[] { batch_kpt, seen }, -1);
                            }

                            batch_bbox = Ops.xywh2xyxy(batch_bbox);
                            torch.Tensor iou = Metrics.box_iou(batch_bbox, pred_bboxes);
                            torch.Tensor tp_epoch = match_predictions(pred_classes, turn_classes, iou);
                            torch.Tensor kpt_scales = torch.tensor(new[] { w, h, 1.0f }, device: batch_kpt.device);

                            batch_kpt = batch_kpt * kpt_scales;

                            // `0.53` is from https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384
                            torch.Tensor area = Ops.xyxy2xywh(batch_bbox)[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(2)].prod(1) * 0.53f;
                            torch.Tensor piou = Metrics.kpt_iou(batch_kpt, pred_kpt, sigma: OKS_SIGMA, area: area);
                            torch.Tensor tp_p = match_predictions(pred_classes, turn_classes, piou);

                            tpList.Add(tp_epoch.MoveToOuterDisposeScope());
                            ptpList.Add(tp_p.MoveToOuterDisposeScope());
                            pred_scoresList.Add(pred_scores.MoveToOuterDisposeScope());
                            pred_classesList.Add(pred_classes.MoveToOuterDisposeScope());
                            true_classesList.Add(turn_classes.MoveToOuterDisposeScope());
                        }

                        if (loss_items.NumberOfElements < 1)
                        {
                            loss_items = torch.zeros_like(ls_item);
                        }
                        loss_items = loss_items + ls_item.to(loss_items.dtype, loss_items.device);
                        loss_items = loss_items.MoveToOuterDisposeScope();
                        count += data["images"].shape[0];
                        // pbar.SetPostfix(new (string key, object value)[] { ("Val Loss", $"{loss_items.sum().ToSingle() / count:f3}"), });
                    }
                }

                torch.Tensor tp_total = torch.cat(tpList);
                torch.Tensor scores_total = torch.cat(pred_scoresList);
                torch.Tensor pred_classes_total = torch.cat(pred_classesList);
                torch.Tensor true_classes_total = torch.cat(true_classesList);

                torch.Tensor tp_p_total = torch.cat(ptpList);
                (torch.Tensor tp, torch.Tensor fp, torch.Tensor p, torch.Tensor r, torch.Tensor f1, torch.Tensor ap, torch.Tensor unique_class, torch.Tensor p_curve, torch.Tensor r_curve, torch.Tensor f1_curve, torch.Tensor x, torch.Tensor prec_values) = Metrics.ap_per_class(tp_total, scores_total, pred_classes_total, true_classes_total);
                (torch.Tensor p_tp, torch.Tensor p_fp, torch.Tensor p_p, torch.Tensor p_r, torch.Tensor p_f1, torch.Tensor p_ap, torch.Tensor p_unique_class, torch.Tensor p_p_curve, torch.Tensor p_r_curve, torch.Tensor p_f1_curve, torch.Tensor p_x, torch.Tensor p_prec_values) = Metrics.ap_per_class(tp_p_total, scores_total, pred_classes_total, true_classes_total);

                float R = r.mean().ToSingle();
                float P = p.mean().ToSingle();
                float mAP50 = ap[torch.TensorIndex.Ellipsis, 0].mean().ToSingle();
                float mAP50_95 = ap[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(1)].mean().ToSingle();

                float P_p = p_p.mean().ToSingle();
                float R_p = r.mean().ToSingle();
                float mAP50_p = p_ap[torch.TensorIndex.Ellipsis, 0].mean().ToSingle();
                float mAP50_95_p = p_ap[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(1)].mean().ToSingle();

                StringBuilder resultBuilder = new StringBuilder();
                resultBuilder.AppendFormat("{0,10}", "All");
                resultBuilder.AppendFormat("{0,10}", count);
                resultBuilder.AppendFormat("{0,10}", true_classes_total.shape[0]);
                resultBuilder.AppendFormat("{0,10}", P.ToString("0.000"));
                resultBuilder.AppendFormat("{0,10}", R.ToString("0.000"));
                resultBuilder.AppendFormat("{0,10}", mAP50.ToString("0.000"));
                resultBuilder.AppendFormat("{0,10}", mAP50_95.ToString("0.000"));
                resultBuilder.AppendFormat("{0,10}", P_p.ToString("0.000"));
                resultBuilder.AppendFormat("{0,10}", R_p.ToString("0.000"));
                resultBuilder.AppendFormat("{0,10}", mAP50_p.ToString("0.000"));
                resultBuilder.AppendFormat("{0,10}", mAP50_95_p.ToString("0.000"));

                Console.WriteLine(resultBuilder.ToString());

                return (loss_items.@float().data<float>().ToArray(), new float[] { P, R, mAP50, mAP50_95, P_p, R_p, mAP50_p, mAP50_95_p });
            }
        }

        internal override string GetTrainDescription()
        {
            string[] strs = new string[] { "Epoch", "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss", "Instances", "Size" };
            StringBuilder stringBuilder = new StringBuilder();
            foreach (string str in strs)
            {
                stringBuilder.AppendFormat("{0,10}", str);
            }
            return stringBuilder.ToString();
        }


        internal override string GetValDescription()
        {
            string[] strs = new string[] { "Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)", "Pose(P", "R", "mAP50", "mAP50-95)" };
            StringBuilder stringBuilder = new StringBuilder();
            foreach (string str in strs)
            {
                stringBuilder.AppendFormat("{0,10}", str);
            }
            return stringBuilder.ToString();
        }

        internal override string GetSeperatLogHeaders()
        {
            return "Epoch, Time, train/box_loss, train/pose_loss, train/kobj_loss, train/cls_loss, train/dfl_loss, val/box_loss, val/pose_loss, val/kobj_loss, val/cls_loss, val/dfl_loss, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B), metrics/precision(P), metrics/recall(P), metrics/mAP50(P), metrics/mAP50-95(P), train/loss, val/loss";
        }

    }
}
