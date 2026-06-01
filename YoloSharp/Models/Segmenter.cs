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

            yolo = config.YoloType switch
            {
                YoloType.Yolov5u => new Yolo.Yolov5uSegment(config.NumberClass, yoloSize: config.YoloSize, end2end: config.End2End, device: config.Device, dtype: config.Dtype),
                YoloType.Yolov8 => new Yolo.Yolov8Segment(config.NumberClass, yoloSize: config.YoloSize, end2end: config.End2End, device: config.Device, dtype: config.Dtype),
                YoloType.Yolov11 => new Yolo.Yolov11Segment(config.NumberClass, yoloSize: config.YoloSize, end2end: config.End2End, device: config.Device, dtype: config.Dtype),
                YoloType.Yolov12 => new Yolo.Yolov12Segment(config.NumberClass, yoloSize: config.YoloSize, end2end: config.End2End, device: config.Device, dtype: config.Dtype),
                _ => throw new NotImplementedException("Yolo type not supported."),
            };
            loss = config.End2End ? new Loss.E2ESegmentLoss(config.NumberClass, epoches: config.Epochs, device: config.Device, dtype: config.Dtype) : new Loss.v8SegmentationLoss(config.NumberClass, device: config.Device, dtype: config.Dtype);
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

            Dictionary<string, Tensor> inference = yolo.forward(input)?.inference;
            List<Tensor> nms_results = Ops.non_max_suppression((torch.Tensor)(inference["boxes"]), nc: this.config.NumberClass, conf_thres: predictThreshold, iou_thres: iouThreshold, end2end: config.End2End).output;
            Tensor proto = inference["proto"];

            List<YoloResult> results = new List<YoloResult>();
            if (proto.shape[0] > 0)
            {
                if (!Equals(nms_results[0], null))
                {
                    int i = 0;
                    Tensor masks = Ops.process_mask(proto[i], nms_results[i][torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(6)], nms_results[i][torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(0, 4)], new long[] { input.shape[2], input.shape[3] }, upsample: true);
                    nms_results[i][torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(0, 4)] = Ops.clip_boxes(nms_results[i][torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(0, 4)], new float[] { orgImage.shape[2], orgImage.shape[3] });
                    masks = torchvision.transforms.functional.crop(masks, 0, 0, (int)input.shape[2], (int)input.shape[3]);
                    masks = torchvision.transforms.functional.resize(masks, (int)orgImage.shape[2], (int)orgImage.shape[3]);

                    for (int j = 0; j < masks.shape[0]; j++)
                    {
                        byte[,] mask = new byte[masks.shape[2], masks.shape[1]];
                        Buffer.BlockCopy(masks[j].transpose(0, 1).@byte().data<byte>().ToArray(), 0, mask, 0, mask.Length);

                        int x = nms_results[i][j, 0].ToInt32();
                        int y = nms_results[i][j, 1].ToInt32();

                        int ww = nms_results[i][j, 2].ToInt32() - x;
                        int hh = nms_results[i][j, 3].ToInt32() - y;

                        results.Add(new YoloResult()
                        {
                            ClassID = nms_results[i][j, 5].ToInt32(),
                            Score = nms_results[i][j, 4].ToSingle(),
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
                        var preds = amp.Evaluate(data["images"].to(config.Dtype));
                        var (ls, ls_item) = loss.forward(preds?.preds, data);

                        torch.Tensor proto = preds?.inference["proto"];

                        float w = data["images"].shape[data["images"].shape.Length - 1];
                        float h = data["images"].shape[data["images"].shape.Length - 2];
                        torch.Tensor scale = torch.tensor(new float[] { w, h, w, h }, device: new Device(data["images"].device_type));
                        (List<Tensor> nms_results, _) = Ops.non_max_suppression(preds?.inference["boxes"], nc: config.NumberClass, conf_thres: 0.01f, iou_thres: 0.7f, end2end: config.End2End);

                        for (int i = 0; i < nms_results.Count; i++)
                        {
                            Tensor pred_bboxes = nms_results[i][torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(0, 4)];
                            Tensor pred_scores = nms_results[i][torch.TensorIndex.Ellipsis, 4];
                            Tensor pred_classes = nms_results[i][torch.TensorIndex.Ellipsis, 5];
                            Tensor coefficient = nms_results[i][torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(6)];
                            long[] size = new long[] { proto.shape[proto.shape.Length - 1], proto.shape[proto.shape.Length - 2] };
                            Tensor masks = Ops.process_mask(proto[i], coefficient, pred_bboxes, size);

                            Tensor batch_idx = data["batch_idx"].squeeze(-1) == i;
                            Tensor turn_classes = data["cls"][batch_idx].squeeze(-1);
                            Tensor batch_bbox = data["bboxes"][batch_idx] * scale;
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
                float mAP50 = ap[torch.TensorIndex.Ellipsis, 0].mean().ToSingle();
                float mAP50_95 = ap[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(1)].mean().ToSingle();

                float mR = rm.mean().ToSingle();
                float mP = pm.mean().ToSingle();
                float mAP50m = apm[torch.TensorIndex.Ellipsis, 0].mean().ToSingle();
                float mAP50_95m = apm[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(1)].mean().ToSingle();

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
                return (loss_items.@float().data<float>().ToArray(), new float[] { P, R, mAP50, mAP50_95, mP, mR, mAP50m, mAP50_95m });
            }
        }

        internal override string GetTrainDescription()
        {
            string[] strs = new string[] { "Epoch", "box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss", "Instances", "Size" };
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

        internal override string GetSeperatLogHeaders()
        {
            return "Epoch, Time, train/box_loss, train/seg_loss, train/cls_loss, train/dfl_loss, train/sem_loss, val/box_loss, val/seg_loss, val/cls_loss, val/dfl_loss, val/sem_loss, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B), metrics/precision(M), metrics/recall(M), metrics/mAP50(M), metrics/mAP50-95(M), train/loss, val/loss";
        }

    }
}
