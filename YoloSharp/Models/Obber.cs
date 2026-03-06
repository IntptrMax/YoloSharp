using Data;
using System.Text;
using TorchSharp;
using Utils;
using YoloSharp.Types;
using YoloSharp.Utils;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace YoloSharp.Models
{
    internal class Obber : YoloBaseTaskModel
    {
        internal Obber(Config config)
        {
            this.config = config;
            if (config.YoloType == YoloType.Yolov5 || config.YoloType == YoloType.Yolov5u || config.YoloType == YoloType.Yolov12)
            {
                throw new ArgumentException("Obb not support yolov5, yolov5u or yolov12. Please use yolov8 or yolov11 instead.");
            }

            yolo = config.YoloType switch
            {
                YoloType.Yolov8 => new Yolo.Yolov8Obb(config.NumberClass, config.YoloSize, config.Device, config.Dtype),
                YoloType.Yolov11 => new Yolo.Yolov11Obb(config.NumberClass, config.YoloSize, config.Device, config.Dtype),
                _ => throw new NotImplementedException("Yolo type not supported."),
            };
            loss = config.YoloType switch
            {
                YoloType.Yolov8 => new Loss.V8OBBLoss(config.NumberClass),
                YoloType.Yolov11 => new Loss.V8OBBLoss(config.NumberClass),
                _ => throw new NotImplementedException("Yolo type not supported."),
            };

            //Tools.TransModelFromSafetensors(yolo, @".\yolov8n-obb.safetensors", @".\PreTrainedModels\yolov8n-obb.bin");
        }

        internal override List<YoloResult> ImagePredict(Tensor orgImage, float predictThreshold, float iouThreshold)
        {
            using (no_grad())
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

                Tensor input = functional.pad(orgImage, new long[] { 0, padWidth, 0, padHeight }, PaddingModes.Zeros, 114) / 255.0f;
                Tensor[] tensors = yolo.forward(input);
                (List<Tensor> nms_result, var _) = Ops.non_max_suppression(tensors[0], nc: config.NumberClass, conf_thres: predictThreshold, iou_thres: iouThreshold, rotated: true);
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
                            result.Radian = nms_result[0][i][6].ToSingle();
                            result.ClassID = nms_result[0][i][5].ToInt32();
                            result.Score = nms_result[0][i][4].ToSingle();
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

            using (Tqdm<Dictionary<string, Tensor>> pbar = new Tqdm<Dictionary<string, Tensor>>(valDataLoader, desc: desc.ToString(), total: (int)valDataLoader.Count, barStyle: Tqdm.BarStyle.Classic, barColor: Tqdm.BarColor.White, barWidth: 10, showPartialChar: true))
            using (no_grad())
            {
                yolo.eval();
                Tensor loss_items = torch.empty(0);
                long count = 0;
                List<Tensor> tpList = new List<Tensor>();
                List<Tensor> pred_scoresList = new List<Tensor>();
                List<Tensor> pred_classesList = new List<Tensor>();
                List<Tensor> true_classesList = new List<Tensor>();

                foreach (Dictionary<string, Tensor> data in pbar)
                {
                    using (NewDisposeScope())
                    {
                        if (data["batch_idx"].NumberOfElements < 1)
                        {
                            continue;
                        }
                        List<Tensor> preds = amp.Evaluate(data["images"].to(config.Dtype)).ToList();
                        Tensor pred = preds[0];
                        Tensor[] list = preds.Take(new Range(1, preds.Count)).ToArray();
                        var (ls, ls_item) = loss.forward(list, data);

                        bool is_obb = (config.TaskType == TaskType.Obb);
                        float conf_thres = is_obb ? 0.01f : 0.001f;
                        (List<Tensor> nms_results, _) = Ops.non_max_suppression(pred, nc: config.NumberClass, conf_thres: conf_thres, iou_thres: 0.7f, rotated: is_obb);

                        for (int i = 0; i < nms_results.Count; i++)
                        {
                            Tensor pred_bboxes = torch.cat(new Tensor[] { nms_results[i][.., 0..4], nms_results[i][.., 6..7] }, 1);
                            Tensor pred_scores = nms_results[i][.., 4];
                            Tensor pred_classes = nms_results[i][.., 5];

                            Tensor batch_idx = data["batch_idx"].squeeze(-1) == i;
                            Tensor turn_classes = data["cls"][batch_idx].squeeze(-1);
                            Tensor batch_bbox = torch.cat(new Tensor[] { data["bboxes"][batch_idx][.., 0..4] * config.ImageSize, data["bboxes"][batch_idx][.., 4..5] }, 1);

                            Tensor iou = Metrics.batch_probiou(batch_bbox, pred_bboxes);
                            Tensor tp_epoch = match_predictions(pred_classes, turn_classes, iou);
                            tpList.Add(tp_epoch.MoveToOuterDisposeScope());
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

                Tensor tp_total = torch.cat(tpList);
                Tensor scores_total = torch.cat(pred_scoresList);
                Tensor pred_classes_total = torch.cat(pred_classesList);
                Tensor true_classes_total = torch.cat(true_classesList);
                (Tensor tp, Tensor fp, Tensor p, Tensor r, Tensor f1, Tensor ap, Tensor unique_class, Tensor p_curve, Tensor r_curve, Tensor f1_curve, Tensor x, Tensor prec_values) = Metrics.ap_per_class(tp_total, scores_total, pred_classes_total, true_classes_total);

                float R = r.mean().ToSingle();
                float P = p.mean().ToSingle();
                float mAP50 = ap[.., 0].mean().ToSingle();
                float mAP50_95 = ap[.., 1..].mean().ToSingle();

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

    }
}
