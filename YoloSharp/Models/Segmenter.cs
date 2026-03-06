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
    internal class Segmenter : YoloBaseTaskModel
    {
        internal Segmenter(Config config)
        {
            this.config = config;
            if (config.YoloType == YoloType.Yolov5 || config.YoloType == YoloType.Yolov5u || config.YoloType == YoloType.Yolov12)
            {
                throw new ArgumentException("Segmenter not support yolov5, yolov5u or yolov12. Please use yolov8 or yolov11 instead.");
            }

            yolo = config.YoloType switch
            {
                YoloType.Yolov8 => new Yolo.Yolov8Segment(config.NumberClass, config.YoloSize, config.Device, config.Dtype),
                YoloType.Yolov11 => new Yolo.Yolov11Segment(config.NumberClass, config.YoloSize, config.Device, config.Dtype),
                _ => throw new NotImplementedException("Yolo type not supported."),
            };
            loss = config.YoloType switch
            {
                YoloType.Yolov8 => new Loss.V8SegmentationLoss(config.NumberClass),
                YoloType.Yolov11 => new Loss.V8SegmentationLoss(config.NumberClass),
                _ => throw new NotImplementedException("Yolo type not supported."),
            };
            //Tools.TransModelFromSafetensors(yolo, @".\yolov8n-seg.safetensors", @".\PreTrainedModels\yolov11x-seg.bin");
        }

        internal override List<YoloResult> ImagePredict(Tensor orgImage, float predictThreshold, float iouThreshold)
        {
            // Change RGB → BGR
            orgImage = orgImage.to(config.Dtype, config.Device).unsqueeze(0);

            int w = (int)orgImage.shape[3];
            int h = (int)orgImage.shape[2];
            int padHeight = 32 - (int)(orgImage.shape[2] % 32);
            int padWidth = 32 - (int)(orgImage.shape[3] % 32);

            padHeight = padHeight == 32 ? 0 : padHeight;
            padWidth = padWidth == 32 ? 0 : padWidth;

            Tensor input = functional.pad(orgImage, new long[] { 0, padWidth, 0, padHeight }, PaddingModes.Zeros, 114) / 255.0f;
            yolo.eval();

            Tensor[] outputs = yolo.forward(input);

            (List<Tensor> preds, var _) = Ops.non_max_suppression(outputs[0], nc: this.config.NumberClass, conf_thres: predictThreshold, iou_thres: iouThreshold);
            Tensor proto = outputs[4];

            List<YoloResult> results = new List<YoloResult>();
            if (proto.shape[0] > 0)
            {
                if (!Equals(preds[0], null))
                {
                    int i = 0;
                    Tensor masks = Ops.process_mask(proto[i], preds[i][.., 6..], preds[i][.., 0..4], new long[] { input.shape[2], input.shape[3] }, upsample: true);
                    preds[i][.., ..4] = Ops.clip_boxes(preds[i][.., ..4], new float[] { orgImage.shape[2], orgImage.shape[3] });
                    masks = torchvision.transforms.functional.crop(masks, 0, 0, (int)input.shape[2], (int)input.shape[3]);
                    masks = torchvision.transforms.functional.resize(masks, (int)orgImage.shape[2], (int)orgImage.shape[3]);

                    for (int j = 0; j < masks.shape[0]; j++)
                    {
                        byte[,] mask = new byte[masks.shape[2], masks.shape[1]];
                        Buffer.BlockCopy(masks[j].transpose(0, 1).@byte().data<byte>().ToArray(), 0, mask, 0, mask.Length);

                        int x = preds[i][j, 0].ToInt32();
                        int y = preds[i][j, 1].ToInt32();

                        int ww = preds[i][j, 2].ToInt32() - x;
                        int hh = preds[i][j, 3].ToInt32() - y;

                        results.Add(new YoloResult()
                        {
                            ClassID = preds[i][j, 5].ToInt32(),
                            Score = preds[i][j, 4].ToSingle(),
                            CenterX = x + ww / 2,
                            CenterY = y + hh / 2,
                            Width = ww,
                            Height = hh,
                            Mask = mask
                        });
                    }
                }
            }
            return results;
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
                List<Tensor> tpmList = new List<Tensor>();
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
                        preds = new List<Tensor>() { preds[1], preds[2], preds[3], preds[0][.., (int)(config.NumberClass + 4).., TensorIndex.Colon], preds[4] };
                        var (ls, ls_item) = loss.forward(preds.ToArray(), data);
                        Tensor protos = preds[4];
                        bool is_obb = (config.TaskType == TaskType.Obb);
                        float conf_thres = is_obb ? 0.01f : 0.001f;
                        (List<Tensor> nms_results, _) = Ops.non_max_suppression(pred, nc: config.NumberClass, conf_thres: conf_thres, iou_thres: 0.7f, rotated: is_obb);

                        for (int i = 0; i < nms_results.Count; i++)
                        {
                            Tensor pred_bboxes = nms_results[i][.., 0..4];
                            Tensor pred_scores = nms_results[i][.., 4];
                            Tensor pred_classes = nms_results[i][.., 5];
                            Tensor coefficient = nms_results[i][.., 6..];
                            long[] size = new long[] { protos.shape[protos.shape.Length - 2], protos.shape[protos.shape.Length - 1] };
                            Tensor masks = Ops.process_mask(protos[i], coefficient, pred_bboxes, size);

                            Tensor batch_idx = data["batch_idx"].squeeze(-1) == i;
                            Tensor turn_classes = data["cls"][batch_idx].squeeze(-1);
                            Tensor batch_bbox = data["bboxes"][batch_idx] * config.ImageSize;
                            Tensor batch_mask = data["masks"];
                            batch_mask = batch_mask[i].squeeze(0);

                            long nl = turn_classes.shape[0];
                            Tensor index = torch.arange(1, nl + 1, device: batch_mask.device).view(nl, 1, 1);
                            batch_mask = (batch_mask == index).@float();

                            batch_bbox = Ops.xywh2xyxy(batch_bbox);
                            Tensor iou = Metrics.box_iou(batch_bbox, pred_bboxes);
                            Tensor tp_epoch = match_predictions(pred_classes, turn_classes, iou);

                            Tensor miou = Metrics.mask_iou(batch_mask.flatten(1).@float(), masks.flatten(1));
                            Tensor tp_m_epoch = match_predictions(pred_classes, turn_classes, miou);

                            tpList.Add(tp_epoch.MoveToOuterDisposeScope());
                            tpmList.Add(tp_m_epoch.MoveToOuterDisposeScope());
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
                Tensor tp_m_total = torch.cat(tpmList);
                Tensor scores_total = torch.cat(pred_scoresList);
                Tensor pred_classes_total = torch.cat(pred_classesList);
                Tensor true_classes_total = torch.cat(true_classesList);
                (Tensor tp, Tensor fp, Tensor p, Tensor r, Tensor f1, Tensor ap, Tensor unique_class, Tensor p_curve, Tensor r_curve, Tensor f1_curve, Tensor x, Tensor prec_values) = Metrics.ap_per_class(tp_total, scores_total, pred_classes_total, true_classes_total);

                (Tensor tpm, Tensor fpm, Tensor pm, Tensor rm, Tensor f1m, Tensor apm, Tensor unique_classm, Tensor p_curvem, Tensor r_curvem, Tensor f1_curvem, Tensor xm, Tensor prec_valuesm) = Metrics.ap_per_class(tp_m_total, scores_total, pred_classes_total, true_classes_total);

                float R = r.mean().ToSingle();
                float P = p.mean().ToSingle();
                float mAP50 = ap[.., 0].mean().ToSingle();
                float mAP50_95 = ap[.., 1..].mean().ToSingle();

                float mR = rm.mean().ToSingle();
                float mP = pm.mean().ToSingle();
                float mAP50m = apm[.., 0].mean().ToSingle();
                float mAP50_95m = apm[.., 1..].mean().ToSingle();

                StringBuilder resultBuilder = new StringBuilder();
                resultBuilder.AppendFormat("{0,10}", "All");
                resultBuilder.AppendFormat("{0,10}", count);
                resultBuilder.AppendFormat("{0,10}", true_classes_total.shape[0]);
                resultBuilder.AppendFormat("{0,10}", P.ToString("0.000"));
                resultBuilder.AppendFormat("{0,10}", R.ToString("0.000"));
                resultBuilder.AppendFormat("{0,10}", mAP50.ToString("0.000"));
                resultBuilder.AppendFormat("{0,10}", mAP50_95.ToString("0.000"));

                resultBuilder.AppendFormat("{0,10}", mP.ToString("0.000"));
                resultBuilder.AppendFormat("{0,10}", mR.ToString("0.000"));
                resultBuilder.AppendFormat("{0,10}", mAP50m.ToString("0.000"));
                resultBuilder.AppendFormat("{0,10}", mAP50_95m.ToString("0.000"));

                Console.WriteLine(resultBuilder.ToString());
                return  (loss_items.@float().data<float>().ToArray(), new float[] { P, R, mAP50, mAP50_95, mP, mR, mAP50m, mAP50_95m });
            }
        }

        internal override string GetTrainDescription()
        {
            string[] strs = new string[] { "Epoch", "box_loss", "seg_loss", "cls_loss", "dfl_loss", "Instances", "Size" };
            StringBuilder stringBuilder = new StringBuilder();
            foreach (string str in strs)
            {
                stringBuilder.AppendFormat("{0,10}", str);
            }
            return stringBuilder.ToString();
        }

        internal override string GetValDescription()
        {
            string[] strs = new string[] { "Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)", "Mask(P", "R", "mAP50", "mAP50-95)" };
            StringBuilder stringBuilder = new StringBuilder();
            foreach (string str in strs)
            {
                stringBuilder.AppendFormat("{0,10}", str);
            }
            return stringBuilder.ToString();
        }

    }
}
