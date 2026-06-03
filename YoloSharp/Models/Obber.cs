using Data;
using System.Text;
using TorchSharp;
using Utils;
using YoloSharp.Types;
using YoloSharp.Utils;

namespace YoloSharp.Models
{
    internal class Obber : YoloBaseTaskModel
    {
        internal Obber(Config config)
        {
            this.config = config;

            yolo = config.YoloType switch
            {
                YoloType.Yolov5u => new Yolo.Yolov5uObb(config.NumberClass, end2end: config.End2End, yoloSize: config.YoloSize, device: config.Device, dtype: config.Dtype),
                YoloType.Yolov8 => new Yolo.Yolov8Obb(config.NumberClass, end2end: config.End2End, yoloSize: config.YoloSize, device: config.Device, dtype: config.Dtype),
                YoloType.Yolov11 => new Yolo.Yolov11Obb(config.NumberClass, end2end: config.End2End, yoloSize: config.YoloSize, device: config.Device, dtype: config.Dtype),
                YoloType.Yolov12 => new Yolo.Yolov12Obb(config.NumberClass, end2end: config.End2End, yoloSize: config.YoloSize, device: config.Device, dtype: config.Dtype),
                _ => throw new NotImplementedException("Yolo type not supported."),
            };
            loss = config.End2End ? new Loss.E2EOBBLoss(config.NumberClass, epoches: config.Epochs, device: config.Device, dtype: config.Dtype) : new Loss.v8OBBLoss(config.NumberClass, device: config.Device, dtype: config.Dtype);
            //Tools.TransModelFromSafetensors(yolo, @".\yolov8n-obb.safetensors", @".\PreTrainedModels\yolov8n-obb.bin");
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
                List<torch.Tensor> nms_result = Ops.non_max_suppression(inference["boxes"], nc: config.NumberClass, conf_thres: predictThreshold, iou_thres: iouThreshold, rotated: true, end2end: config.End2End).output;
                List<YoloResult> results = new List<YoloResult>();
                if (nms_result.Count > 0)
                {
                    if (nms_result[0] is not null)
                    {
                        for (int i = 0; i < nms_result[0].shape[0]; i++)
                        {
                            YoloResult result = new YoloResult();
                            result.CenterX = nms_result[0][i][0].ToInt32();
                            result.CenterY = nms_result[0][i][1].ToInt32();
                            result.Width = nms_result[0][i][2].ToInt32();
                            result.Height = nms_result[0][i][3].ToInt32();
                            result.Score = nms_result[0][i][4].ToSingle();
                            result.ClassID = nms_result[0][i][5].ToInt32();
                            result.Radian = nms_result[0][i][6].ToSingle();

                            results.Add(result);
                        }
                    }
                }
                return results;
            }
        }

        internal override (float[] loss, float[] metrics) Val(YoloDataLoader valDataLoader, AMPWrapper amp, int epoch)
        {
            string desc = GetValDescription();

            using (Tqdm<Dictionary<string, torch.Tensor>> pbar = new Tqdm<Dictionary<string, torch.Tensor>>(valDataLoader, desc: desc.ToString(), total: (int)valDataLoader.Count, barStyle: Tqdm.BarStyle.Classic, barColor: Tqdm.BarColor.White, barWidth: 10, showPartialChar: true))
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
                        var preds = amp.Evaluate(data["images"].to(config.Dtype));
                        torch.Tensor loss_detach = loss.forward(preds?.preds, data).loss_detach;

                        float w = data["images"].shape[data["images"].shape.Length - 1];
                        float h = data["images"].shape[data["images"].shape.Length - 2];
                        torch.Tensor scale = torch.tensor(new float[] { w, h, w, h }, device: new torch.Device(data["images"].device_type));
                        List<torch.Tensor> nms_results = Ops.non_max_suppression(preds?.inference["boxes"], nc: config.NumberClass, conf_thres: 0.01f, iou_thres: 0.7f, rotated: true, end2end: config.End2End).output;

                        for (int i = 0; i < nms_results.Count; i++)
                        {
                            torch.Tensor pred_bboxes = torch.cat(new torch.Tensor[] { nms_results[i][torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(0, 4)], nms_results[i][torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(6, 7)] }, 1);
                            torch.Tensor pred_scores = nms_results[i][torch.TensorIndex.Ellipsis, 4];
                            torch.Tensor pred_classes = nms_results[i][torch.TensorIndex.Ellipsis, 5];

                            torch.Tensor batch_idx = data["batch_idx"].squeeze(-1) == i;
                            torch.Tensor turn_classes = data["cls"][batch_idx].squeeze(-1);
                            torch.Tensor batch_bbox = torch.cat(new torch.Tensor[] { data["bboxes"][batch_idx][torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(0, 4)] * scale, data["bboxes"][batch_idx][torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(4, 5)] }, 1);

                            torch.Tensor iou = Metrics.batch_probiou(batch_bbox, pred_bboxes);
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
                        // pbar.SetPostfix(new (string key, object value)[] { ("Val Loss", $"{loss_items.sum().ToSingle() / count:f3}"), });

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

        internal override string GetTrainDescription()
        {
            string[] strs = new string[] { "Epoch", "box_loss", "cls_loss", "dfl_loss", "ag_loss", "Instances", "Size" };
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
            return "Epoch, Time, train/box_loss, train/cls_loss, train/dfl_loss, train/angle_loss, val/box_loss, val/cls_oss, val/dfl_loss, val/angle_loss, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B), train/loss, val/loss";
        }

    }
}
