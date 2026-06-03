using Newtonsoft.Json.Converters;
using ScottPlot.TickGenerators.Financial;
using System.Diagnostics;
using TorchSharp;
using TorchSharp.Modules;
using static Tensorboard.TensorShapeProto.Types;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace YoloSharp.Utils
{
    internal class Loss
    {
        private static float[] OKS_SIGMA
        {
            get
            {
                float[] v = new float[] { 0.26f, 0.25f, 0.25f, 0.35f, 0.35f, 0.79f, 0.79f, 0.72f, 0.72f, 0.62f, 0.62f, 1.07f, 1.07f, 0.87f, 0.87f, 0.89f, 0.89f };
                return v.Select(x => x / 10.0f).ToArray();
            }
        }

        /// <summary>
        /// Returns label smoothing BCE targets for reducing overfitting
        /// </summary>
        /// <param name="eps"></param>
        /// <returns>pos: `1.0 - 0.5*eps`, neg: `0.5*eps`.</returns>
        private static (float pos, float neg) Smooth_BCE(float eps = 0.1f)
        {
            // For details see https://github.com/ultralytics/yolov3/issues/238;  //issuecomment-598028441"""
            return (1.0f - 0.5f * eps, 0.5f * eps);
        }

        private class BCEBlurWithLogitsLoss : Module<Tensor, Tensor, Tensor>
        {
            private readonly BCEWithLogitsLoss loss_fcn;
            private readonly float alpha;
            public BCEBlurWithLogitsLoss(float alpha = 0.05f, Reduction reduction = Reduction.None) : base(nameof(BCEBlurWithLogitsLoss))
            {
                this.loss_fcn = BCEWithLogitsLoss(reduction: reduction);  // must be nn.BCEWithLogitsLoss()
                this.alpha = alpha;
            }

            public override Tensor forward(Tensor pred, Tensor t)
            {
                using (NewDisposeScope())
                {
                    Tensor loss = loss_fcn.forward(pred, t);
                    pred = sigmoid(pred);  // prob from logits
                    Tensor dx = pred - t;// ;  // reduce only missing label effects
                                         // dx = (pred - true).abs()  ;  // reduce missing label and false label effects
                    Tensor alpha_factor = 1 - exp((dx - 1) / (alpha + 1e-4));
                    loss *= alpha_factor;
                    return loss.mean();
                }
            }
        }

        private class FocalLoss : Module<Tensor, Tensor, Tensor>
        {
            private readonly BCEWithLogitsLoss loss_fcn;
            private readonly float alpha;
            private readonly float gamma;
            private Reduction reduction;
            public FocalLoss(BCEWithLogitsLoss loss_fcn, float gamma = 1.5f, float alpha = 0.25f) : base(nameof(FocalLoss))
            {
                this.loss_fcn = loss_fcn;  // must be nn.BCEWithLogitsLoss()
                this.gamma = gamma;
                this.alpha = alpha;
                reduction = loss_fcn.reduction;
            }

            public override Tensor forward(Tensor pred, Tensor t)
            {
                using (NewDisposeScope())
                {
                    Tensor loss = loss_fcn.forward(pred, t);
                    Tensor pred_prob = sigmoid(pred);  // prob from logits
                    Tensor p_t = true * pred_prob + (1 - t) * (1 - pred_prob);
                    Tensor alpha_factor = t * alpha + (1 - t) * (1 - alpha);
                    Tensor modulating_factor = (1.0 - p_t).pow(gamma);

                    loss *= alpha_factor * modulating_factor;

                    loss = reduction switch
                    {
                        Reduction.Mean => loss.mean(),
                        Reduction.Sum => loss.sum(),
                        Reduction.None => loss,
                        _ => loss
                    };
                    return loss.MoveToOuterDisposeScope();
                }
            }
        }

        internal class DFLoss : Module<Tensor, Tensor, Tensor>
        {
            private readonly int reg_max;
            public int regMax => reg_max;

            public DFLoss(int reg_max = 16) : base(nameof(DFLoss))
            {
                this.reg_max = reg_max;
            }

            public override Tensor forward(Tensor pred_dist, Tensor target)
            {
                using (NewDisposeScope())
                {
                    target = target.clamp_(0, reg_max - 1 - 0.01);

                    Tensor tl = target.@long(); // target left
                    Tensor tr = tl + 1; //target right
                    Tensor wl = tr - target; //weight left
                    Tensor wr = 1 - wl; //weight right
                    return (
                        functional.cross_entropy(pred_dist, tl.view(-1), reduction: Reduction.None).view(tl.shape) * wl
                        + functional.cross_entropy(pred_dist, tr.view(-1), reduction: Reduction.None).view(tl.shape) * wr
                    ).mean(new long[] { -1 }, keepdim: true).MoveToOuterDisposeScope();
                }
            }
        }

        internal class BboxLoss : Module
        {
            protected readonly DFLoss? dfl_loss;
            protected readonly int reg_max;

            public BboxLoss(int regMax = 16) : base(nameof(BboxLoss))
            {
                dfl_loss = regMax > 1 ? new DFLoss(regMax) : null;
                reg_max = regMax;
            }

            public virtual (Tensor loss_iou, Tensor loss_dfl) forward(Tensor pred_dist, Tensor pred_bboxes, Tensor anchor_points, Tensor target_bboxes, Tensor target_scores, Tensor target_scores_sum, Tensor fg_mask)
            {
                using (NewDisposeScope())
                {
                    // Step 1: Compute weight
                    Tensor weight = target_scores.sum(new long[] { -1 })[fg_mask].unsqueeze(-1);

                    // Step 2: Compute IoU
                    Tensor iou = Metrics.bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], false, true);
                    Tensor lossIou = ((1.0 - iou) * weight).sum() / target_scores_sum;

                    // Step 3: Compute DFL loss
                    Tensor lossDfl;
                    if (dfl_loss is not null)
                    {
                        Tensor targetLtrb = Tal.bbox2dist(anchor_points, target_bboxes, reg_max - 1);
                        lossDfl = dfl_loss.forward(pred_dist[fg_mask].view(-1, reg_max), targetLtrb[fg_mask]) * weight;
                        lossDfl = lossDfl.sum() / target_scores_sum;
                    }
                    else
                    {
                        lossDfl = tensor(0.0, device: pred_dist.device);
                    }

                    return (lossIou.MoveToOuterDisposeScope(), lossDfl.MoveToOuterDisposeScope());
                }
            }

            public virtual (Tensor loss_iou, Tensor loss_dfl) forward(Tensor pred_dist, Tensor pred_bboxes, Tensor anchor_points, Tensor target_bboxes, Tensor target_scores, Tensor target_scores_sum, Tensor fg_mask, Tensor imgsz, Tensor stride)
            {
                using (NewDisposeScope())
                {
                    Tensor weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1);
                    Tensor iou = Metrics.bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh: false, CIoU: true);
                    Tensor loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum;

                    // Step 3: Compute DFL loss
                    Tensor loss_dfl;
                    if (dfl_loss is not null)
                    {
                        Tensor target_ltrb = Tal.bbox2dist(anchor_points, target_bboxes, reg_max - 1);
                        loss_dfl = this.dfl_loss.forward(pred_dist[fg_mask].view(-1, reg_max), target_ltrb[fg_mask]) * weight;
                        loss_dfl = loss_dfl.sum() / target_scores_sum;
                    }
                    else
                    {
                        Tensor target_ltrb = Tal.bbox2dist(anchor_points, target_bboxes);
                        // normalize ltrb by image size
                        target_ltrb = target_ltrb * stride;
                        target_ltrb[TensorIndex.Ellipsis, TensorIndex.Slice(start: 0, step: 2)] /= imgsz[1];
                        target_ltrb[TensorIndex.Ellipsis, TensorIndex.Slice(start: 1, step: 2)] /= imgsz[0];
                        pred_dist = pred_dist * stride;
                        pred_dist[TensorIndex.Ellipsis, TensorIndex.Slice(start: 0, step: 2)] /= imgsz[1];
                        pred_dist[TensorIndex.Ellipsis, TensorIndex.Slice(start: 1, step: 2)] /= imgsz[0];
                        loss_dfl = (torch.nn.functional.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction: Reduction.None).mean(new long[] { -1 }, keepdim: true) * weight);
                        loss_dfl = loss_dfl.sum() / target_scores_sum;
                    }

                    return (loss_iou.MoveToOuterDisposeScope(), loss_dfl.MoveToOuterDisposeScope());
                }
            }
        }

        private class KeypointLoss : Module<torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor>
        {
            private readonly Tensor sigmas;
            internal KeypointLoss(torch.Tensor sigmas) : base(nameof(KeypointLoss))
            {
                this.sigmas = sigmas;
            }

            public override Tensor forward(torch.Tensor pred_kpts, torch.Tensor gt_kpts, torch.Tensor kpt_mask, torch.Tensor area)
            {
                using (NewDisposeScope())
                {
                    torch.Tensor d = (pred_kpts[TensorIndex.Ellipsis, 0] - gt_kpts[TensorIndex.Ellipsis, 0]).pow(2) + (pred_kpts[TensorIndex.Ellipsis, 1] - gt_kpts[TensorIndex.Ellipsis, 1]).pow(2);
                    torch.Tensor kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim: 1) + 1e-6);
                    // e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
                    torch.Tensor e = d / ((2 * this.sigmas.to(pred_kpts.device)).pow(2) * (area + 1e-9) * 2); // from cocoeval
                    return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean().MoveToOuterDisposeScope();
                }
            }
        }

        private class RotatedBboxLoss : BboxLoss
        {
            internal RotatedBboxLoss(int regMax) : base(regMax: regMax)
            {

            }

            public override (Tensor loss_iou, Tensor loss_dfl) forward(Tensor pred_dist, Tensor pred_bboxes, Tensor anchor_points, Tensor target_bboxes, Tensor target_scores, Tensor target_scores_sum, Tensor fg_mask, Tensor imgsz, Tensor stride)
            {
                using (no_grad())
                using (NewDisposeScope())
                {
                    Tensor weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1);
                    Tensor iou = Metrics.probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask]);
                    Tensor loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum;
                    Tensor loss_dfl = torch.zeros(0);

                    // DFL loss
                    if (dfl_loss is not null)
                    {
                        Tensor target_ltrb = Tal.rbox2dist(target_bboxes[TensorIndex.Ellipsis, torch.TensorIndex.Slice(0, 4)], anchor_points, target_bboxes[TensorIndex.Ellipsis, torch.TensorIndex.Slice(4, 5)], reg_max: this.reg_max - 1);
                        loss_dfl = this.dfl_loss.forward(pred_dist[fg_mask].view(-1, this.reg_max), target_ltrb[fg_mask]) * weight;
                        loss_dfl = loss_dfl.sum() / target_scores_sum;
                    }
                    else
                    {
                        Tensor target_ltrb = Tal.rbox2dist(target_bboxes[TensorIndex.Ellipsis, torch.TensorIndex.Slice(0, 4)], anchor_points, target_bboxes[TensorIndex.Ellipsis, torch.TensorIndex.Slice(4, 5)]);
                        target_ltrb = target_ltrb * stride;
                        target_ltrb[TensorIndex.Ellipsis, TensorIndex.Slice(0, null, 2)] /= imgsz[1];
                        target_ltrb[TensorIndex.Ellipsis, TensorIndex.Slice(1, null, 2)] /= imgsz[0];
                        pred_dist = pred_dist * stride;
                        pred_dist[TensorIndex.Ellipsis, TensorIndex.Slice(0, null, 2)] /= imgsz[1];
                        pred_dist[TensorIndex.Ellipsis, TensorIndex.Slice(1, null, 2)] /= imgsz[0];
                        loss_dfl = (torch.nn.functional.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction: Reduction.None).mean(new long[] { -1 }, keepdim: true) * weight);
                        loss_dfl = loss_dfl.sum() / target_scores_sum;
                    }
                    return (loss_iou.MoveToOuterDisposeScope(), loss_dfl.MoveToOuterDisposeScope());
                }
            }
        }

        /// <summary>
        /// /Criterion class for computing multi-channel Dice losses.
        /// </summary>
        private class MultiChannelDiceLoss : Module<torch.Tensor, torch.Tensor, torch.Tensor>
        {
            private readonly float smooth;
            private readonly Reduction reduction;
            /// <summary>
            /// Initialize MultiChannelDiceLoss with smoothing and reduction options.
            /// </summary>
            /// <param name="smooth">Smoothing factor to avoid division by zero.</param>
            /// <param name="reduction">Reduction method ('mean', 'sum', or 'none').</param>
            internal MultiChannelDiceLoss(float smooth = 1e-6f, Reduction reduction = Reduction.Mean) : base(nameof(MultiChannelDiceLoss))
            {
                this.smooth = smooth;
                this.reduction = reduction;
            }

            /// <summary>
            /// Calculate multi-channel Dice loss between predictions and targets.
            /// </summary>
            /// <param name="pred"></param>
            /// <param name="target"></param>
            /// <returns></returns>
            public override Tensor forward(torch.Tensor pred, torch.Tensor target)
            {
                Debug.Assert(pred.shape == target.shape, "the size of predict and target must be equal.");

                pred = pred.sigmoid();
                torch.Tensor intersection = (pred * target).sum(dim: new long[] { 2, 3 });
                torch.Tensor union = pred.sum(dim: new long[] { 2, 3 }) + target.sum(dim: new long[] { 2, 3 });
                torch.Tensor dice = (2.0 * intersection + this.smooth) / (union + this.smooth);
                torch.Tensor dice_loss = 1.0 - dice;
                dice_loss = dice_loss.mean(dimensions: new long[] { 1 });

                if (this.reduction == Reduction.Mean)
                {
                    return dice_loss.mean();
                }
                else if (this.reduction == Reduction.Sum)
                {
                    return dice_loss.sum();
                }
                else
                {
                    return dice_loss;
                }
            }
        }

        /// <summary>
        /// Criterion class for computing combined BCE and Dice losses.
        /// </summary>
        private class BCEDiceLoss : Module<torch.Tensor, torch.Tensor, torch.Tensor>
        {
            private readonly float weight_bce;
            private readonly float weight_dice;

            private readonly BCEWithLogitsLoss bce;
            private readonly MultiChannelDiceLoss dice;

            /// <summary>
            /// Initialize BCEDiceLoss with BCE and Dice weight factors.
            /// </summary>
            /// <param name="weight_bce">Weight factor for BCE loss component.</param>
            /// <param name="weight_dice">Weight factor for Dice loss component.</param>
            internal BCEDiceLoss(float weight_bce = 0.5f, float weight_dice = 0.5f) : base(nameof(BCEDiceLoss))
            {
                this.weight_bce = weight_bce;
                this.weight_dice = weight_dice;
                this.bce = nn.BCEWithLogitsLoss();
                this.dice = new MultiChannelDiceLoss(smooth: 1);

                RegisterComponents();
            }

            /// <summary>
            /// Calculate combined BCE and Dice loss between predictions and targets.
            /// </summary>
            /// <param name="pred"></param>
            /// <param name="target"></param>
            /// <returns></returns>
            /// <exception cref="NotImplementedException"></exception>
            public override Tensor forward(Tensor pred, Tensor target)
            {
                long mask_h = pred.shape[2];
                long mask_w = pred.shape[3];

                // downsample to the same size as pred
                if (target.shape[target.shape.Length - 2] != mask_h || target.shape[target.shape.Length - 1] != mask_w)
                {
                    target = torch.nn.functional.interpolate(target, new long[] { mask_h, mask_w }, mode: InterpolationMode.Nearest);
                }
                return this.weight_bce * this.bce.forward(pred, target) + this.weight_dice * this.dice.forward(pred, target);
            }
        }


        internal class v8DetectionLoss : Module<Dictionary<string, object>, Dictionary<string, Tensor>, (Tensor loss, Tensor loss_detach)>
        {
            protected BCEWithLogitsLoss bce;
            protected int[] stride;
            protected int nc;
            protected int no;
            protected int reg_max;
            protected Device device;
            protected bool use_dfl;
            protected Tal.TaskAlignedAssigner assigner;
            protected BboxLoss bbox_loss;
            protected readonly Tensor proj;
            protected readonly float hyp_box;
            protected readonly float hyp_cls;
            protected readonly float hyp_dfl;

            internal v8DetectionLoss(int nc, int reg_max = 16, int[]? stride = null, int tal_topk = 10, int? tal_topk2 = null, float hyp_box = 7.5f, float hyp_cls = 0.5f, float hyp_dfl = 1.5f, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(v8DetectionLoss))
            {
                this.bce = nn.BCEWithLogitsLoss(reduction: Reduction.None);
                this.stride = stride ?? new int[] { 8, 16, 32 };  // model strides
                this.nc = nc;  // number of classes
                this.no = nc + reg_max * 4;
                this.reg_max = reg_max;
                this.device = device;
                this.hyp_box = hyp_box;
                this.hyp_cls = hyp_cls;
                this.hyp_dfl = hyp_dfl;

                this.use_dfl = reg_max > 1;
                this.assigner = new Tal.TaskAlignedAssigner(topk: tal_topk, num_classes: this.nc, alpha: 0.5f, beta: 6.0f, stride = this.stride, topk2: tal_topk2);
                this.bbox_loss = new BboxLoss(reg_max).to(device);
                this.proj = torch.arange(reg_max, dtype: torch.float32, device: device);
                RegisterComponents();
            }

            private Tensor preprocess(torch.Tensor targets, int batch_size, torch.Tensor scale_tensor)
            {
                using (NewDisposeScope())
                {
                    long nl = targets.shape[0];
                    long ne = targets.shape[1];

                    Tensor @out;
                    if (nl == 0)
                    {
                        @out = torch.zeros(batch_size, 0, ne - 1, device: this.device);
                    }
                    else
                    {
                        Tensor batch_idx = targets[TensorIndex.Colon, 0].@long();  // image index
                        Tensor counts = batch_idx.unique(return_counts: true).counts;
                        counts = counts.to(torch.int32);
                        @out = torch.zeros(batch_size, counts.max().ToInt32(), ne - 1, device: this.device);
                        Tensor offsets = torch.zeros(batch_size + 1, dtype: torch.int64, device: this.device);
                        offsets = offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx));
                        offsets = offsets.cumsum(0);
                        Tensor within_idx = torch.arange(nl, device: this.device) - offsets[batch_idx];
                        @out[batch_idx, within_idx] = targets[TensorIndex.Colon, torch.TensorIndex.Slice(1)];
                        @out[TensorIndex.Ellipsis, torch.TensorIndex.Slice(1, 5)] = Ops.xywh2xyxy(@out[TensorIndex.Ellipsis, torch.TensorIndex.Slice(1, 5)].mul_(scale_tensor));
                    }
                    return @out.MoveToOuterDisposeScope();
                }
            }

            /// <summary>
            /// Decode predicted object bounding box coordinates from anchor points and distribution.
            /// </summary>
            /// <param name="anchor_points"></param>
            /// <param name="pred_dist"></param>
            /// <returns></returns>
            private Tensor bbox_decode(Tensor anchor_points, Tensor pred_dist)
            {
                if (this.use_dfl)
                {
                    long b = pred_dist.shape[0];   // batch
                    long a = pred_dist.shape[1];   // anchors
                    long c = pred_dist.shape[2];   // channels
                    pred_dist = pred_dist.view(b, a, 4, c / 4).softmax(3).matmul(this.proj.type(pred_dist.dtype));
                }

                return Tal.dist2bbox(pred_dist, anchor_points, xywh: false);
            }

            protected ((Tensor fg_mask, Tensor target_gt_idx, Tensor target_bboxes, Tensor anchor_points, Tensor stride_tensor) target, Tensor loss, Tensor loss_detach) get_assigned_targets_and_loss(Dictionary<string, object> preds, Dictionary<string, Tensor> batch)
            {
                using (NewDisposeScope())
                {
                    Tensor loss = torch.zeros(3, device: this.device);  // box, cls, dfl
                    Tensor pred_distri = ((Tensor)preds["boxes"]).permute(0, 2, 1).contiguous();
                    Tensor pred_scores = ((Tensor)preds["scores"]).permute(0, 2, 1).contiguous();
                    (Tensor anchor_points, Tensor stride_tensor) = Tal.make_anchors((Tensor[])preds["feats"], this.stride, 0.5f);
                    ScalarType dtype = pred_scores.dtype;
                    int batch_size = (int)pred_scores.shape[0];
                    Tensor imgsz = torch.tensor(new long[] { ((Tensor[])preds["feats"])[0].shape[2], ((Tensor[])preds["feats"])[0].shape[3] }, device: this.device, dtype: dtype) * this.stride[0];

                    // Targets
                    Tensor targets = torch.cat(new Tensor[] { ((Tensor)batch["batch_idx"]).view(-1, 1), ((Tensor)batch["cls"]).view(-1, 1), (Tensor)batch["bboxes"] }, 1);
                    var indices = torch.tensor(new long[] { 1, 0, 1, 0 }, device: device);
                    targets = this.preprocess(targets.to(this.device), batch_size, scale_tensor: imgsz[indices]);

                    Tensor[] gt_labels_bboxes = targets.split(new long[] { 1, 4 }, 2);  // cls, xyxy
                    Tensor gt_labels = gt_labels_bboxes[0];
                    Tensor gt_bboxes = gt_labels_bboxes[1];
                    Tensor mask_gt = gt_bboxes.sum(2, keepdim: true).gt_(0.0);

                    // Pboxes
                    Tensor pred_bboxes = this.bbox_decode(anchor_points, pred_distri);  // xyxy, (b, h*w, 4)

                    (_, Tensor target_bboxes, Tensor target_scores, Tensor fg_mask, Tensor target_gt_idx) = this.assigner.forward(
                           pred_scores.detach().sigmoid(),
                           (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                           anchor_points * stride_tensor,
                           gt_labels,
                           gt_bboxes,
                           mask_gt);

                    float target_scores_sum = Math.Max(target_scores.sum().ToSingle(), 1);

                    // Cls loss
                    loss[1] = this.bce.forward(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum; //BCE

                    // Bbox loss
                    if (fg_mask.sum().ToSingle() > 0)
                    {
                        (loss[0], loss[2]) = this.bbox_loss.forward(
                            pred_distri,
                            pred_bboxes,
                            anchor_points,
                            target_bboxes / stride_tensor,
                            target_scores,
                            target_scores_sum,
                            fg_mask,
                            imgsz,
                            stride_tensor);
                    }
                    loss[0] *= this.hyp_box;  // box gain
                    loss[1] *= this.hyp_cls;  // cls gain
                    loss[2] *= this.hyp_dfl;  // dfl gain
                    return ((fg_mask.MoveToOuterDisposeScope(), target_gt_idx.MoveToOuterDisposeScope(), target_bboxes.MoveToOuterDisposeScope(), anchor_points.MoveToOuterDisposeScope(), stride_tensor.MoveToOuterDisposeScope()), loss.MoveToOuterDisposeScope(), loss.detach());
                }
            }

            (Tensor loss, Tensor loss_items) loss(Dictionary<string, object> preds, Dictionary<string, Tensor> batch)
            {
                long batch_size = ((Tensor)preds["boxes"]).shape[0];
                (_, Tensor loss, Tensor loss_detach) = this.get_assigned_targets_and_loss(preds, batch);
                //Tensor loss_detach = l.loss_detach;

                return (loss * batch_size, loss.detach());
            }

            public override (Tensor loss, Tensor loss_detach) forward(Dictionary<string, object> preds, Dictionary<string, Tensor> batch)
            {
                return this.loss(preds, batch);
            }

        }

        internal class v8OBBLoss : v8DetectionLoss
        {
            private readonly float hyp_angle;
            internal v8OBBLoss(int nc, int reg_max = 16, int[]? stride = null, int tal_topk = 10, int? tal_topk2 = null, float hyp_box = 7.5f, float hyp_cls = 0.5f, float hyp_dfl = 1.5f, float hyp_angle = 1.0f, Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, reg_max: reg_max, stride: stride, tal_topk: tal_topk, tal_topk2: tal_topk2, hyp_box: hyp_box, hyp_cls: hyp_cls, hyp_dfl: hyp_dfl, device: device, dtype: dtype)
            {
                this.hyp_angle = hyp_angle;
                this.assigner = new Tal.RotatedTaskAlignedAssigner(
                                    topk: tal_topk,
                                    num_classes: nc,
                                    alpha: 0.5f,
                                    beta: 6.0f,
                                    stride: stride,
                                    topk2: tal_topk2);
                this.bbox_loss = new RotatedBboxLoss(this.reg_max).to(this.device);
                RegisterComponents();
            }

            /// <summary>
            /// Preprocess targets for oriented bounding box detection.
            /// </summary>
            /// <param name="targets"></param>
            /// <param name="batch_size"></param>
            /// <param name="scale_tensor"></param>
            /// <returns></returns>
            private Tensor preprocess(Tensor targets, int batch_size, Tensor scale_tensor)
            {
                using (torch.NewDisposeScope())
                {
                    if (targets.shape[0] == 0)
                    {
                        torch.Tensor @out = torch.zeros(new long[] { batch_size, 0, 6 }, device: this.device);
                        return @out.MoveToOuterDisposeScope();
                    }
                    else
                    {
                        torch.Tensor batch_idx = targets[TensorIndex.Colon, 0].@long();  // image index
                        torch.Tensor counts = batch_idx.unique(return_counts: true).counts;
                        counts = counts.to(torch.int32);
                        torch.Tensor @out = torch.zeros(new long[] { batch_size, counts.max().ToInt32(), 6 }, device: this.device);
                        torch.Tensor packed_targets = targets[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(1)].clone();
                        packed_targets[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(1, 5)] = packed_targets[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(1, 5)].mul_(scale_tensor);
                        torch.Tensor offsets = torch.zeros(batch_size + 1, dtype: torch.int64, device: this.device);
                        offsets = offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx));
                        offsets = offsets.cumsum(0);
                        torch.Tensor within_idx = torch.arange(targets.shape[0], device: this.device) - offsets[batch_idx];
                        @out[batch_idx, within_idx] = packed_targets;
                        return @out.MoveToOuterDisposeScope();
                    }
                }
            }

            /// <summary>
            /// Calculate and return the loss for oriented bounding box detection.
            /// </summary>
            /// <param name="preds"></param>
            /// <param name="batch"></param>
            /// <returns></returns>
            private (Tensor loss, Tensor loss_items) loss(Dictionary<string, object> preds, Dictionary<string, Tensor> batch)
            {
                using (torch.NewDisposeScope())
                {
                    torch.Tensor loss = torch.zeros(4, device: this.device);
                    torch.Tensor pred_distri = ((torch.Tensor)preds["boxes"]).permute(0, 2, 1).contiguous();
                    torch.Tensor pred_scores = ((torch.Tensor)preds["scores"]).permute(0, 2, 1).contiguous();
                    torch.Tensor pred_angle = ((torch.Tensor)preds["angle"]).permute(0, 2, 1).contiguous();

                    (torch.Tensor anchor_points, torch.Tensor stride_tensor) = Tal.make_anchors((torch.Tensor[])preds["feats"], this.stride, 0.5f);
                    int batch_size = (int)pred_angle.shape[0];  // batch size
                    ScalarType dtype = pred_scores.dtype;
                    torch.Tensor imgsz = torch.tensor(new long[] { ((torch.Tensor[])preds["feats"])[0].shape[2], ((torch.Tensor[])preds["feats"])[0].shape[3] }, device: this.device, dtype: dtype) * this.stride[0];

                    torch.Tensor batch_idx = batch["batch_idx"].view(-1, 1);
                    torch.Tensor targets = torch.cat(new torch.Tensor[] { batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5) }, 1);
                    torch.Tensor rw = targets[TensorIndex.Colon, 4] * (imgsz[1].ToSingle());
                    torch.Tensor rh = targets[TensorIndex.Colon, 5] * (imgsz[0].ToSingle());
                    targets = targets[(rw >= 2) & (rh >= 2)];  // filter rboxes of tiny size to stabilize training
                    Tensor indices = tensor(new long[] { 1, 0, 1, 0 }, device: device);

                    targets = this.preprocess(targets.to(this.device), batch_size, scale_tensor: imgsz[indices]);

                    torch.Tensor[] gt_labels_bboxes = targets.split(new long[] { 1, 5 }, 2);  // cls, xywhr
                    torch.Tensor gt_labels = gt_labels_bboxes[0];
                    torch.Tensor gt_bboxes = gt_labels_bboxes[1];
                    torch.Tensor mask_gt = gt_bboxes.sum(2, keepdim: true).gt_(0.0);

                    // Pboxes
                    torch.Tensor pred_bboxes = this.bbox_decode(anchor_points, pred_distri, pred_angle);  // xyxy, (b, h*w, 4)

                    torch.Tensor bboxes_for_assigner = pred_bboxes.clone().detach();

                    // Only the first four elements need to be scaled
                    bboxes_for_assigner[TensorIndex.Ellipsis, torch.TensorIndex.Slice(0, 4)] *= stride_tensor;

                    (_, torch.Tensor target_bboxes, torch.Tensor target_scores, torch.Tensor fg_mask, _) = this.assigner.forward(
                                                                               pred_scores.detach().sigmoid(),
                                                                               bboxes_for_assigner.type(gt_bboxes.dtype),
                                                                               anchor_points * stride_tensor,
                                                                               gt_labels,
                                                                               gt_bboxes,
                                                                               mask_gt
                                                                           );

                    float target_scores_sum = Math.Max(target_scores.sum().ToSingle(), 1);

                    // Cls loss
                    // loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
                    loss[1] = this.bce.forward(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum;  // BCE

                    // Bbox loss
                    if (fg_mask.sum().ToSingle() > 0)
                    {
                        target_bboxes[TensorIndex.Ellipsis, torch.TensorIndex.Slice(0, 4)] /= stride_tensor;
                        (loss[0], loss[2]) = this.bbox_loss.forward(
                                                 pred_distri,
                                                 pred_bboxes,
                                                 anchor_points,
                                                 target_bboxes,
                                                 target_scores,
                                                 target_scores_sum,
                                                 fg_mask,
                                                 imgsz,
                                                 stride_tensor
                                             );
                        torch.Tensor weight = target_scores.sum(-1)[fg_mask];
                        loss[3] = this.calculate_angle_loss(pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum);  // angle loss
                    }
                    else
                    {
                        loss[0] += (pred_angle * 0).sum();
                    }

                    loss[0] *= this.hyp_box;  // box gain
                    loss[1] *= this.hyp_cls;  // cls gain
                    loss[2] *= this.hyp_dfl;  // dfl gain
                    loss[3] *= this.hyp_angle;  // angle gain

                    return ((loss * batch_size).MoveToOuterDisposeScope(), loss.detach().MoveToOuterDisposeScope());  // loss(box, cls, dfl, angle)

                }
            }

            /// <summary>
            /// Decode predicted object bounding box coordinates from anchor points and distribution.
            /// </summary>
            /// <param name="anchor_points">Anchor points, (h*w, 2).</param>
            /// <param name="pred_dist">Predicted rotated distance, (bs, h*w, 4).</param>
            /// <param name="pred_angle">Predicted angle, (bs, h*w, 1).</param>
            /// <returns>Predicted rotated bounding boxes with angles, (bs, h*w, 5).</returns>
            private torch.Tensor bbox_decode(torch.Tensor anchor_points, torch.Tensor pred_dist, torch.Tensor pred_angle)
            {
                if (this.use_dfl)
                {
                    // batch, anchors, channels
                    long b = pred_dist.shape[0];
                    long a = pred_dist.shape[1];
                    long c = pred_dist.shape[2];
                    pred_dist = pred_dist.view(b, a, 4, c / 4).softmax(3).matmul(this.proj.type(pred_dist.dtype));
                }
                return torch.cat(new torch.Tensor[] { Tal.dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle }, dim: -1);
            }

            /// <summary>
            /// Calculate oriented angle loss.
            /// </summary>
            /// <param name="pred_bboxes">Predicted bounding boxes with shape [N, 5] (x, y, w, h, theta).</param>
            /// <param name="target_bboxes">Target bounding boxes with shape [N, 5] (x, y, w, h, theta).</param>
            /// <param name="fg_mask">Foreground mask indicating valid predictions.</param>
            /// <param name="weight">Loss weights for each prediction.</param>
            /// <param name="target_scores_sum">Sum of target scores for normalization.</param>
            /// <param name="lambda_val">Controls the sensitivity to aspect ratio.</param>
            /// <returns>The calculated angle loss.</returns>
            private torch.Tensor calculate_angle_loss(torch.Tensor pred_bboxes, torch.Tensor target_bboxes, torch.Tensor fg_mask, torch.Tensor weight, torch.Tensor target_scores_sum, int lambda_val = 3)
            {
                using (torch.NewDisposeScope())
                {
                    torch.Tensor w_gt = target_bboxes[TensorIndex.Ellipsis, 2];
                    torch.Tensor h_gt = target_bboxes[TensorIndex.Ellipsis, 3];
                    torch.Tensor pred_theta = pred_bboxes[TensorIndex.Ellipsis, 4];
                    torch.Tensor target_theta = target_bboxes[TensorIndex.Ellipsis, 4];

                    torch.Tensor log_ar = torch.log((w_gt + 1e-9) / (h_gt + 1e-9));
                    torch.Tensor scale_weight = torch.exp(-(torch.pow(log_ar, 2)) / (torch.pow(lambda_val, 2)));

                    torch.Tensor delta_theta = pred_theta - target_theta;
                    torch.Tensor delta_theta_wrapped = delta_theta - torch.round(delta_theta / Math.PI) * Math.PI;
                    torch.Tensor ang_loss = torch.pow(torch.sin(2 * delta_theta_wrapped[fg_mask]), 2);
                    ang_loss = scale_weight[fg_mask] * ang_loss;
                    ang_loss = ang_loss * weight;

                    return (ang_loss.sum() / target_scores_sum).MoveToOuterDisposeScope();
                }
            }

            public override (Tensor loss, Tensor loss_detach) forward(Dictionary<string, object> preds, Dictionary<string, Tensor> batch)
            {
                return this.loss(preds, batch);
            }
        }

        /// <summary>
        /// Criterion class for computing training losses for YOLOv8 segmentation.
        /// </summary>
        internal class v8SegmentationLoss : v8DetectionLoss
        {
            private readonly bool overlap;
            private readonly BCEDiceLoss bcedice_loss;

            internal v8SegmentationLoss(int nc, int reg_max = 16, int[]? stride = null, bool overlap_mask = true, int tal_topk = 10, int? tal_topk2 = null, float hyp_box = 7.5f, float hyp_cls = 0.5f, float hyp_dfl = 1.5f, Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, reg_max: reg_max, stride: stride, tal_topk: tal_topk, tal_topk2: tal_topk2, hyp_box: hyp_box, hyp_cls: hyp_cls, hyp_dfl: hyp_dfl, device: device, dtype: dtype)
            {
                this.overlap = overlap_mask;
                this.bcedice_loss = new BCEDiceLoss(weight_bce: 0.5f, weight_dice: 0.5f);
                RegisterComponents();
            }

            public override (Tensor loss, Tensor loss_detach) forward(Dictionary<string, object> preds, Dictionary<string, Tensor> batch)
            {
                return this.loss(preds, batch);
            }

            /// <summary>
            /// Calculate and return the combined loss for detection and segmentation.
            /// </summary>
            /// <param name="preds"></param>
            /// <param name="batch"></param>
            /// <returns></returns>
            (Tensor loss, Tensor loss_items) loss(Dictionary<string, object> preds, Dictionary<string, Tensor> batch)
            {
                using (torch.NewDisposeScope())
                {
                    Tensor pred_masks = ((Tensor)preds["mask_coefficient"]).permute(0, 2, 1).contiguous();
                    Tensor proto = (Tensor)preds["proto"];
                    preds.TryGetValue("pred_semseg", out object pred_semseg);

                    Tensor loss = torch.zeros(5, device: this.device);  // box, seg, cls, dfl, semseg

                    ((Tensor fg_mask, Tensor target_gt_idx, Tensor target_bboxes, Tensor _, Tensor __) target, torch.Tensor det_loss, torch.Tensor _) p = this.get_assigned_targets_and_loss(preds, batch);

                    Tensor fg_mask = p.target.fg_mask;
                    Tensor target_gt_idx = p.target.target_gt_idx;
                    Tensor target_bboxes = p.target.target_bboxes;
                    Tensor det_loss = p.det_loss;

                    // NOTE: re-assign index for consistency for now. Need to be removed in the future.
                    loss[0] = det_loss[0];
                    loss[2] = det_loss[1];
                    loss[3] = det_loss[2];
                    long batch_size = proto.shape[0];
                    long mask_h = proto.shape[2];
                    long mask_w = proto.shape[3];

                    if (fg_mask.sum().ToSingle() > 0)
                    {
                        torch.Tensor masks = batch["masks"].to(this.device).@float();
                        if (masks.shape[masks.shape.Length - 2] != mask_h || masks.shape[masks.shape.Length - 1] != mask_w) // downsample
                        {
                            // masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
                            proto = torch.nn.functional.interpolate(proto, new long[] { masks.shape[masks.shape.Length - 2], masks.shape[masks.shape.Length - 1] }, mode: InterpolationMode.Bilinear, align_corners: false);
                        }
                        torch.Tensor imgsz = torch.tensor(new long[] { ((torch.Tensor[])preds["feats"])[0].shape[2], ((torch.Tensor[])preds["feats"])[0].shape[3] }, device: this.device, dtype: pred_masks.dtype) * this.stride[0];
                        loss[1] = this.calculate_segmentation_loss(fg_mask, masks, target_gt_idx, target_bboxes, batch["batch_idx"].view(-1, 1), proto, pred_masks, imgsz);
                        if (pred_semseg is not null)
                        {
                            torch.Tensor sem_masks = batch["sem_masks"].to(this.device);  // NxHxW
                            sem_masks = torch.nn.functional.one_hot(sem_masks.@long(), num_classes: this.nc).permute(0, 3, 1, 2).@float();  // NxCxHxW
                            if (this.overlap)
                            {
                                torch.Tensor mask_zero = (masks == 0);  // NxHxW
                                sem_masks[mask_zero.unsqueeze(1).expand_as(sem_masks)] = 0;
                            }
                            else
                            {
                                torch.Tensor batch_idx = batch["batch_idx"].view(-1);  // [total_instances]
                                for (int i = 0; i < batch_size; i++)
                                {

                                    torch.Tensor instance_mask_i = masks[batch_idx == i];  // [num_instances_i, H, W]
                                    if (instance_mask_i.shape[0] == 0)
                                    {
                                        continue;
                                    }
                                    sem_masks[i, TensorIndex.Colon, instance_mask_i.sum(dim: 0) == 0] = 0;
                                }
                            }
                            loss[4] = this.bcedice_loss.forward((torch.Tensor)pred_semseg, sem_masks);
                            loss[4] *= this.hyp_box;  // seg gain
                        }
                    }
                    else
                    {
                        loss[1] += (proto * 0).sum() + (pred_masks * 0).sum();  // inf sums may lead to nan loss
                    }
                    loss[1] *= this.hyp_box; // seg gain
                    return ((loss * batch_size).MoveToOuterDisposeScope(), loss.detach().MoveToOuterDisposeScope()); // loss(box, seg, cls, dfl, semseg)
                }
            }

            /// <summary>
            /// Compute the instance segmentation loss for a single image.
            /// </summary>
            /// <remarks>
            /// The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the predicted masks from the prototype masks and predicted mask coefficients.
            /// </remarks>
            /// <param name="gt_mask">Ground truth mask of shape (N, H, W), where N is the number of objects.</param>
            /// <param name="pred">Predicted mask coefficients of shape (N, 32).</param>
            /// <param name="proto">Prototype masks of shape (32, H, W).</param>
            /// <param name="xyxy">Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (N, 4).</param>
            /// <param name="area">Area of each ground truth bounding box of shape (N,).</param>
            /// <returns>The calculated mask loss for a single image.</returns>
            private torch.Tensor single_mask_loss(torch.Tensor gt_mask, torch.Tensor pred, torch.Tensor proto, torch.Tensor xyxy, torch.Tensor area)
            {
                using (NewDisposeScope())
                {
                    torch.Tensor pred_mask = torch.einsum("in,nhw->ihw", pred, proto);  // (n, 32) @ (32, 80, 80) -> (n, 80, 80)
                    torch.Tensor loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction: Reduction.None);
                    return ((Ops.crop_mask(loss, xyxy).mean(dimensions: new long[] { 1, 2 }) / area).sum()).MoveToOuterDisposeScope();
                }
            }

            /// <summary>
            /// Calculate the loss for instance segmentation.
            /// </summary>
            /// <remarks>
            /// The batch loss can be computed for improved speed at higher memory usage. For example, pred_mask can be computed as follows: <br/>pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
            /// </remarks>
            /// <param name="fg_mask">A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.</param>
            /// <param name="masks">Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).</param>
            /// <param name="target_gt_idx">Indexes of ground truth objects for each anchor of shape (BS, N_anchors).</param>
            /// <param name="target_bboxes">Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).</param>
            /// <param name="batch_idx">Batch indices of shape (N_labels_in_batch, 1).</param>
            /// <param name="proto">Prototype masks of shape (BS, 32, H, W).</param>
            /// <param name="pred_masks">Predicted masks for each anchor of shape (BS, N_anchors, 32).</param>
            /// <param name="imgsz">Size of the input image as a tensor of shape (2), i.e., (H, W).</param>
            /// <returns>The calculated loss for instance segmentation.</returns>
            private torch.Tensor calculate_segmentation_loss(torch.Tensor fg_mask, torch.Tensor masks, torch.Tensor target_gt_idx, torch.Tensor target_bboxes, torch.Tensor batch_idx, torch.Tensor proto, torch.Tensor pred_masks, torch.Tensor imgsz)
            {
                using (torch.NewDisposeScope())
                {
                    long mask_h = proto.shape[2];
                    long mask_w = proto.shape[3];

                    torch.Tensor loss = torch.zeros(1, device: proto.device);

                    // Normalize to 0-1
                    Tensor indices = tensor(new long[] { 1, 0, 1, 0 }, device: device);
                    torch.Tensor target_bboxes_normalized = target_bboxes / imgsz[indices];

                    // Areas of target bboxes
                    torch.Tensor marea = Ops.xyxy2xywh(target_bboxes_normalized)[TensorIndex.Ellipsis, torch.TensorIndex.Slice(2)].prod(2);

                    // Normalize to mask size
                    torch.Tensor mxyxy = target_bboxes_normalized * torch.tensor(new long[] { mask_w, mask_h, mask_w, mask_h }, device: proto.device);
                    for (int i = 0; i < fg_mask.shape[0]; i++)
                    {
                        if (fg_mask[i].any().ToBoolean())
                        {
                            torch.Tensor mask_idx = target_gt_idx[i][fg_mask[i]];
                            torch.Tensor gt_mask;
                            if (this.overlap)
                            {
                                gt_mask = (masks[i] == (mask_idx + 1).view(-1, 1, 1));
                                gt_mask = gt_mask.@float();
                            }
                            else
                            {
                                gt_mask = masks[batch_idx.view(-1) == i][mask_idx];
                            }
                            loss += this.single_mask_loss(gt_mask, pred_masks[i][fg_mask[i]], proto[i], mxyxy[i][fg_mask[i]], marea[i][fg_mask[i]]);
                        }
                        // WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
                        else
                        {
                            loss += (proto * 0).sum() + (pred_masks * 0).sum();  // inf sums may lead to nan loss
                        }
                    }

                    return (loss.sum() / fg_mask.sum()).MoveToOuterDisposeScope();
                }
            }

        }

        /// <summary>
        /// Criterion class for computing training losses for YOLOv8 pose estimation.
        /// </summary>
        internal class v8PoseLoss : v8DetectionLoss
        {
            private readonly int kpt_num;
            private readonly int kpt_dim;
            private readonly float hyp_pose;
            private readonly float hyp_kobj;
            private readonly BCEWithLogitsLoss bce_pose;
            private readonly KeypointLoss keypoint_loss;

            /// <summary>
            /// Initialize v8PoseLoss with model parameters and keypoint-specific loss functions.
            /// </summary>
            /// <param name="nc"></param>
            /// <param name="kpt_num"></param>
            /// <param name="kpt_dim"></param>
            /// <param name="reg_max"></param>
            /// <param name="stride"></param>
            /// <param name="overlap_mask"></param>
            /// <param name="tal_topk"></param>
            /// <param name="tal_topk2"></param>
            /// <param name="hyp_box"></param>
            /// <param name="hyp_cls"></param>
            /// <param name="hyp_dfl"></param>
            /// <param name="hyp_pose"></param>
            /// <param name="device"></param>
            /// <param name="dtype"></param>
            internal v8PoseLoss(int nc, int kpt_num = 17, int kpt_dim = 3, int reg_max = 16, int[]? stride = null, int tal_topk = 10, int? tal_topk2 = 10, float hyp_box = 7.5f, float hyp_cls = 0.5f, float hyp_dfl = 1.5f, float hyp_pose = 12.0f, float hyp_kobj = 1.0f, Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, reg_max: reg_max, stride: stride, tal_topk: tal_topk, tal_topk2: tal_topk2, hyp_box: hyp_box, hyp_cls: hyp_cls, hyp_dfl: hyp_dfl, device: device, dtype: dtype)
            {
                this.kpt_num = kpt_num;
                this.kpt_dim = kpt_dim;
                this.hyp_pose = hyp_pose;
                this.hyp_kobj = hyp_kobj;
                this.bce_pose = new BCEWithLogitsLoss();
                bool is_pose = (kpt_num == 17 && kpt_dim == 3);  // COCO format
                int nkpt = kpt_num;  // number of keypoints
                torch.Tensor sigmas = is_pose ? torch.tensor(OKS_SIGMA) : torch.ones(nkpt) / nkpt;
                this.keypoint_loss = new KeypointLoss(sigmas: sigmas);
                RegisterComponents();
            }

            public override (Tensor loss, Tensor loss_detach) forward(Dictionary<string, object> preds, Dictionary<string, Tensor> batch)
            {
                return this.loss(preds, batch);
            }

            /// <summary>
            /// Calculate the total loss and detach it for pose estimation.
            /// </summary>
            /// <param name="preds"></param>
            /// <param name="batch"></param>
            /// <returns></returns>
            private (Tensor loss, Tensor loss_items) loss(Dictionary<string, object> preds, Dictionary<string, Tensor> batch)
            {
                using (torch.NewDisposeScope())
                {
                    torch.Tensor pred_kpts = ((Tensor)preds["kpts"]).permute(0, 2, 1).contiguous();
                    torch.Tensor loss = torch.zeros(5, device: this.device);  // box, kpt_location, kpt_visibility, cls, dfl
                    ((torch.Tensor fg_mask, torch.Tensor target_gt_idx, torch.Tensor target_bboxes, torch.Tensor anchor_points, torch.Tensor stride_tensor) target, torch.Tensor det_loss, _) = this.get_assigned_targets_and_loss(preds, batch);

                    torch.Tensor fg_mask = target.fg_mask;
                    torch.Tensor target_gt_idx = target.target_gt_idx;
                    torch.Tensor target_bboxes = target.target_bboxes;
                    torch.Tensor anchor_points = target.anchor_points;
                    torch.Tensor stride_tensor = target.stride_tensor;

                    // NOTE: re-assign index for consistency for now. Need to be removed in the future.
                    loss[0] = det_loss[0];
                    loss[3] = det_loss[1];
                    loss[4] = det_loss[2];

                    long batch_size = pred_kpts.shape[0];
                    torch.Tensor imgsz = torch.tensor(new long[] { ((torch.Tensor[])preds["feats"])[0].shape[2], ((torch.Tensor[])preds["feats"])[0].shape[3] }, device: this.device, dtype: pred_kpts.dtype) * this.stride[0];

                    // Pboxes
                    pred_kpts = this.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, this.kpt_num, this.kpt_dim));  // (b, h*w, 17, 3)

                    // Keypoint loss
                    if (fg_mask.sum().ToSingle() > 0)
                    {
                        torch.Tensor keypoints = batch["keypoints"].to(this.device).@float().clone();
                        keypoints[TensorIndex.Ellipsis, 0] *= imgsz[1];
                        keypoints[TensorIndex.Ellipsis, 1] *= imgsz[0];

                        (loss[1], loss[2]) = this.calculate_keypoints_loss(
                            fg_mask,
                            target_gt_idx,
                            keypoints,
                            batch["batch_idx"].view(-1, 1),
                            stride_tensor,
                            target_bboxes,
                            pred_kpts
                        );
                    }

                    loss[1] *= this.hyp_pose;  // pose gain
                    loss[2] *= this.hyp_kobj;  // kobj gain

                    return ((loss * batch_size).MoveToOuterDisposeScope(), loss.detach().MoveToOuterDisposeScope());  // loss(box, pose, kobj, cls, dfl)
                }
            }

            /// <summary>
            /// Decode predicted keypoints to image coordinates.
            /// </summary>
            /// <param name="anchor_points"></param>
            /// <param name="kpts"></param>
            /// <returns></returns>
            private torch.Tensor kpts_decode(torch.Tensor anchor_points, torch.Tensor pred_kpts)
            {
                torch.Tensor y = pred_kpts.clone();
                y[TensorIndex.Ellipsis, torch.TensorIndex.Slice(0, 2)] *= 2.0;
                y[TensorIndex.Ellipsis, 0] += anchor_points[TensorIndex.Colon, torch.TensorIndex.Slice(0, 1)] - 0.5f;
                y[TensorIndex.Ellipsis, 1] += anchor_points[TensorIndex.Colon, torch.TensorIndex.Slice(1, 2)] - 0.5f;
                return y;
            }

            /// <summary>
            /// Select target keypoints for each anchor based on batch index and target ground truth index.
            /// </summary>
            /// <param name="keypoints">Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).</param>
            /// <param name="batch_idx">Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).</param>
            /// <param name="target_gt_idx">Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).</param>
            /// <param name="masks">Binary mask tensor indicating object presence, shape (BS, N_anchors).</param>
            /// <returns>Selected keypoints tensor, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).</returns>
            private torch.Tensor _select_target_keypoints(torch.Tensor keypoints, torch.Tensor batch_idx, torch.Tensor target_gt_idx, torch.Tensor masks)
            {
                using (torch.NewDisposeScope())
                {
                    batch_idx = batch_idx.flatten();
                    long batch_size = masks.shape[0];

                    // Find the maximum number of keypoints in a single image
                    long max_kpts = torch.unique(batch_idx, return_counts: true).counts.max().ToInt64();

                    // Create a tensor to hold batched keypoints
                    torch.Tensor batched_keypoints = torch.zeros(new long[] { batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2] }, device: keypoints.device);

                    // Vectorized fill: compute within-batch position for each keypoint using cumulative offsets
                    torch.Tensor batch_idx_long = batch_idx.@long();
                    torch.Tensor offsets = torch.zeros(batch_size + 1, dtype: torch.@long, device: keypoints.device);
                    offsets.scatter_add_(0, batch_idx_long + 1, torch.ones_like(batch_idx_long));
                    offsets = offsets.cumsum(0);
                    torch.Tensor within_idx = torch.arange(batch_idx.shape[0], device: keypoints.device) - offsets[batch_idx_long];
                    batched_keypoints[batch_idx_long, within_idx] = keypoints;

                    // Expand dimensions of target_gt_idx to match the shape of batched_keypoints
                    torch.Tensor target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1);

                    // Use target_gt_idx_expanded to select keypoints from batched_keypoints
                    torch.Tensor selected_keypoints = batched_keypoints.gather(1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2]));

                    return selected_keypoints.MoveToOuterDisposeScope();
                }

            }

            /// <summary>
            /// Calculate the keypoints loss for the model.
            /// </summary>
            /// <remarks>
            /// This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is based on the difference between the predicted keypoints and ground truth keypoints.The keypoints object loss is a binary classification loss that classifies whether a keypoint is present or not.
            /// </remarks>
            /// <param name="masks">Binary mask tensor indicating object presence, shape (BS, N_anchors).</param>
            /// <param name="target_gt_idx">Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).</param>
            /// <param name="keypoints">Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).</param>
            /// <param name="batch_idx">Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).</param>
            /// <param name="stride_tensor">Stride tensor for anchors, shape (N_anchors, 1).</param>
            /// <param name="target_bboxes">Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).</param>
            /// <param name="pred_kpts">Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).</param>
            /// <returns></returns>
            private (Tensor kpts_loss, Tensor kpts_obj_loss) calculate_keypoints_loss(torch.Tensor masks, torch.Tensor target_gt_idx, torch.Tensor keypoints, torch.Tensor batch_idx, torch.Tensor stride_tensor, torch.Tensor target_bboxes, torch.Tensor pred_kpts)
            {
                using (torch.NewDisposeScope())
                {
                    // Select target keypoints using helper method
                    torch.Tensor selected_keypoints = this._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks);

                    // Divide coordinates by stride
                    selected_keypoints[TensorIndex.Ellipsis, torch.TensorIndex.Slice(0, 2)] /= stride_tensor.view(1, -1, 1, 1);

                    torch.Tensor kpts_loss = 0;
                    torch.Tensor kpts_obj_loss = 0;

                    if (masks.any().ToBoolean())
                    {
                        target_bboxes /= stride_tensor;
                        torch.Tensor gt_kpt = selected_keypoints[masks];
                        torch.Tensor area = Ops.xyxy2xywh(target_bboxes[masks])[TensorIndex.Colon, torch.TensorIndex.Slice(2)].prod(1, keepdim: true);
                        torch.Tensor pred_kpt = pred_kpts[masks];
                        torch.Tensor kpt_mask = gt_kpt.shape[gt_kpt.shape.Length - 1] == 3 ? gt_kpt[TensorIndex.Ellipsis, 2] != 0 : torch.full_like(gt_kpt[TensorIndex.Ellipsis, 0], true);
                        kpts_loss = this.keypoint_loss.forward(pred_kpt, gt_kpt, kpt_mask, area);  // pose loss

                        if (pred_kpt.shape[pred_kpt.shape.Length - 1] == 3)
                        {
                            kpts_obj_loss = this.bce_pose.forward(pred_kpt[TensorIndex.Ellipsis, 2], kpt_mask.@float());  // keypoint obj loss
                        }
                    }

                    return (kpts_loss.MoveToOuterDisposeScope(), kpts_obj_loss.MoveToOuterDisposeScope());
                }
            }
        }

        internal class v8ClassificationLoss : Module<Dictionary<string, object>, Dictionary<string, Tensor>, (Tensor loss, Tensor loss_detach)>
        {
            internal v8ClassificationLoss() : base(nameof(v8ClassificationLoss))
            {

            }

            public override (Tensor loss, Tensor loss_detach) forward(Dictionary<string, object> preds, Dictionary<string, Tensor> batch)
            {
                using (torch.NewDisposeScope())
                {
                    torch.Tensor pred = (Tensor)(preds["cls"]);
                    Tensor loss = torch.nn.functional.cross_entropy(pred, batch["cls"].view(-1), reduction: Reduction.Mean);

                    return (loss.MoveToOuterDisposeScope(), loss.detach().MoveToOuterDisposeScope());

                }
            }
        }


        internal class E2EDetectLoss : Module<Dictionary<string, object>, Dictionary<string, Tensor>, (Tensor loss, Tensor loss_detach)>
        {
            private readonly v8DetectionLoss one2many;
            private readonly v8DetectionLoss one2one;
            internal E2EDetectLoss(int nc, int reg_max = 16, int[]? stride = null, int tal_topk = 10, int? tal_topk2 = null, float hyp_box = 7.5f, float hyp_cls = 0.5f, float hyp_dfl = 1.5f, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(E2EDetectLoss))
            {
                this.one2many = new v8DetectionLoss(nc, reg_max, stride, tal_topk: 10, device: device, dtype: dtype);
                this.one2one = new v8DetectionLoss(nc, reg_max, stride, tal_topk: 1, device: device, dtype: dtype);
                RegisterComponents();
            }

            public override (Tensor loss, Tensor loss_detach) forward(Dictionary<string, object> preds, Dictionary<string, Tensor> batch)
            {
                Dictionary<string, object> local_preds = preds;

                Dictionary<string, object> one2many = (Dictionary<string, object>)local_preds["one2many"];
                var loss_one2many = this.one2many.forward(one2many, batch);

                Dictionary<string, object> one2one = (Dictionary<string, object>)local_preds["one2one"];
                var loss_one2one = this.one2one.forward(one2one, batch);

                return (loss_one2many.loss + loss_one2one.loss, loss_one2many.loss_detach + loss_one2one.loss_detach);

            }
        }

        internal class E2EOBBLoss : Module<Dictionary<string, object>, Dictionary<string, Tensor>, (Tensor loss, Tensor loss_detach)>
        {
            private readonly v8OBBLoss one2many;
            private readonly v8OBBLoss one2one;

            private int updates;
            private float total;
            private float o2m;
            private float o2o;
            private float o2m_copy;
            private float final_o2m;
            private int epochs;

            internal E2EOBBLoss(int nc, int reg_max = 16, int[]? stride = null, int epoches = 100, int tal_topk = 10, int? tal_topk2 = null, float hyp_box = 7.5f, float hyp_cls = 0.5f, float hyp_dfl = 1.5f, float hyp_angle = 1.0f, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(E2EOBBLoss))
            {
                this.one2many = new v8OBBLoss(nc, reg_max, stride, tal_topk: 10, hyp_box: hyp_box, hyp_cls: hyp_cls, hyp_dfl: hyp_dfl, hyp_angle: hyp_angle, device: device, dtype: dtype);
                this.one2one = new v8OBBLoss(nc, reg_max, stride, tal_topk: 7, tal_topk2: 1, hyp_box: hyp_box, hyp_cls: hyp_cls, hyp_dfl: hyp_dfl, hyp_angle: hyp_angle, device: device, dtype: dtype);

                this.updates = 0;
                this.epochs = epoches;
                this.total = 1.0f;

                // init gain
                this.o2m = 0.8f;
                this.o2o = this.total - this.o2m;
                this.o2m_copy = this.o2m;

                // final gain
                this.final_o2m = 0.1f;

                RegisterComponents();
            }

            public override (Tensor loss, Tensor loss_detach) forward(Dictionary<string, object> preds, Dictionary<string, Tensor> batch)
            {
                Dictionary<string, object> local_preds = preds;

                Dictionary<string, object> one2many = (Dictionary<string, object>)local_preds["one2many"];
                (torch.Tensor loss, torch.Tensor loss_detach) loss_one2many = this.one2many.forward(one2many, batch);

                Dictionary<string, object> one2one = (Dictionary<string, object>)local_preds["one2one"];
                (torch.Tensor loss, torch.Tensor loss_detach) loss_one2one = this.one2one.forward(one2one, batch);

                return (loss_one2many.loss * this.o2m + loss_one2one.loss * this.o2o, loss_one2one.loss_detach * this.o2o + loss_one2many.loss_detach * this.o2m);
            }

            public void update()
            {
                this.updates += 1;
                this.o2m = this.decay(this.updates);
                this.o2o = Math.Max(this.total - this.o2m, 0);
            }

            private float decay(float x)
            {
                return Math.Max(1 - x / Math.Max(this.epochs - 1, 1), 0) * (this.o2m_copy - this.final_o2m) + this.final_o2m;
            }
        }

        internal class E2ESegmentLoss : Module<Dictionary<string, object>, Dictionary<string, Tensor>, (Tensor loss, Tensor loss_detach)>
        {
            private readonly v8SegmentationLoss one2many;
            private readonly v8SegmentationLoss one2one;

            private int updates;
            private float total;
            private float o2m;
            private float o2o;
            private float o2m_copy;
            private float final_o2m;
            private int epochs;

            internal E2ESegmentLoss(int nc, int reg_max = 16, int[]? stride = null, int epoches = 100, int tal_topk = 10, int? tal_topk2 = null, float hyp_box = 7.5f, float hyp_cls = 0.5f, float hyp_dfl = 1.5f, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(E2EOBBLoss))
            {
                this.one2many = new v8SegmentationLoss(nc: nc, reg_max: reg_max, stride: stride, tal_topk: 10, hyp_box: hyp_box, hyp_cls: hyp_cls, hyp_dfl: hyp_dfl, device: device, dtype: dtype);
                this.one2one = new v8SegmentationLoss(nc: nc, reg_max: reg_max, stride: stride, tal_topk: 7, tal_topk2: 1, hyp_box: hyp_box, hyp_cls: hyp_cls, hyp_dfl: hyp_dfl, device: device, dtype: dtype);

                this.updates = 0;
                this.epochs = epoches;
                this.total = 1.0f;

                // init gain
                this.o2m = 0.8f;
                this.o2o = this.total - this.o2m;
                this.o2m_copy = this.o2m;

                // final gain
                this.final_o2m = 0.1f;

                RegisterComponents();
            }

            public override (Tensor loss, Tensor loss_detach) forward(Dictionary<string, object> preds, Dictionary<string, Tensor> batch)
            {
                Dictionary<string, object> local_preds = preds;

                Dictionary<string, object> one2many = (Dictionary<string, object>)local_preds["one2many"];
                (torch.Tensor loss, torch.Tensor loss_detach) loss_one2many = this.one2many.forward(one2many, batch);

                Dictionary<string, object> one2one = (Dictionary<string, object>)local_preds["one2one"];
                (torch.Tensor loss, torch.Tensor loss_detach) loss_one2one = this.one2one.forward(one2one, batch);

                return (loss_one2many.loss * this.o2m + loss_one2one.loss * this.o2o, loss_one2one.loss_detach * this.o2o + loss_one2many.loss_detach * this.o2m);
            }

            public void update()
            {
                this.updates += 1;
                this.o2m = this.decay(this.updates);
                this.o2o = Math.Max(this.total - this.o2m, 0);
            }

            private float decay(float x)
            {
                return Math.Max(1 - x / Math.Max(this.epochs - 1, 1), 0) * (this.o2m_copy - this.final_o2m) + this.final_o2m;
            }
        }

        internal class E2EPoseLoss : Module<Dictionary<string, object>, Dictionary<string, Tensor>, (Tensor loss, Tensor loss_detach)>
        {
            private readonly v8PoseLoss one2many;
            private readonly v8PoseLoss one2one;

            private int updates;
            private float total;
            private float o2m;
            private float o2o;
            private float o2m_copy;
            private float final_o2m;
            private int epochs;

            internal E2EPoseLoss(int nc, int kpt_num = 17, int kpt_dim = 3, int reg_max = 16, int[]? stride = null, int epoches = 100, int tal_topk = 10, int? tal_topk2 = 10, float hyp_box = 7.5f, float hyp_cls = 0.5f, float hyp_dfl = 1.5f, float hyp_pose = 12.0f, float hyp_kobj = 1.0f, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(E2EOBBLoss))
            {
                this.one2many = new v8PoseLoss(nc: nc, kpt_num: kpt_num, kpt_dim: kpt_dim, reg_max: reg_max, stride: stride, tal_topk: 10, hyp_box: hyp_box, hyp_cls: hyp_cls, hyp_dfl: hyp_dfl, hyp_pose: hyp_pose, hyp_kobj: hyp_kobj, device: device, dtype: dtype);
                this.one2one = new v8PoseLoss(nc: nc, kpt_num: kpt_num, kpt_dim: kpt_dim, reg_max: reg_max, stride: stride, tal_topk: 7, tal_topk2: 1, hyp_box: hyp_box, hyp_cls: hyp_cls, hyp_dfl: hyp_dfl, hyp_pose: hyp_pose, hyp_kobj: hyp_kobj, device: device, dtype: dtype);

                this.updates = 0;
                this.epochs = epoches;
                this.total = 1.0f;

                // init gain
                this.o2m = 0.8f;
                this.o2o = this.total - this.o2m;
                this.o2m_copy = this.o2m;

                // final gain
                this.final_o2m = 0.1f;

                RegisterComponents();
            }

            public override (Tensor loss, Tensor loss_detach) forward(Dictionary<string, object> preds, Dictionary<string, Tensor> batch)
            {
                Dictionary<string, object> local_preds = preds;

                Dictionary<string, object> one2many = (Dictionary<string, object>)local_preds["one2many"];
                (torch.Tensor loss, torch.Tensor loss_detach) loss_one2many = this.one2many.forward(one2many, batch);

                Dictionary<string, object> one2one = (Dictionary<string, object>)local_preds["one2one"];
                (torch.Tensor loss, torch.Tensor loss_detach) loss_one2one = this.one2one.forward(one2one, batch);

                return (loss_one2many.loss * this.o2m + loss_one2one.loss * this.o2o, loss_one2one.loss_detach * this.o2o + loss_one2many.loss_detach * this.o2m);
            }

            public void update()
            {
                this.updates += 1;
                this.o2m = this.decay(this.updates);
                this.o2o = Math.Max(this.total - this.o2m, 0);
            }

            private float decay(float x)
            {
                return Math.Max(1 - x / Math.Max(this.epochs - 1, 1), 0) * (this.o2m_copy - this.final_o2m) + this.final_o2m;
            }
        }
    }
}



