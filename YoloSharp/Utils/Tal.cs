using Microsoft.VisualBasic;
using OpenCvSharp.Dnn;
using ScottPlot.Palettes;
using ScottPlot.TickGenerators.Financial;
using TorchSharp;
using TorchSharp.Modules;
using Utils;
using static Tensorboard.TensorShapeProto.Types;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace YoloSharp.Utils
{
    internal class Tal
    {
        /// <summary>
        /// A task-aligned assigner for object detection.
        /// </summary>
        /// <remarks>
        /// This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both classification and localization information.
        /// </remarks>
        internal class TaskAlignedAssigner : Module<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, (Tensor, Tensor, Tensor, Tensor, Tensor)>
        {
            /// <summary>
            /// The number of top candidates to consider.
            /// </summary>
            private readonly int topk;

            /// <summary>
            /// Secondary topk value for additional filtering.
            /// </summary>
            private int topk2;

            /// <summary>
            /// The number of object classes.
            /// </summary>
            private readonly int num_classes;
            private readonly float alpha;
            private readonly float beta;
            private readonly float eps;
            private long bs;
            private long n_max_boxes;
            protected int[] stride;

            protected int stride_val;

            internal TaskAlignedAssigner(int topk = 13, int num_classes = 80, float alpha = 1.0f, float beta = 6.0f, int[]? stride = null, float eps = 1e-9f, int? topk2 = null) : base("TaskAlignedAssigner")
            {
                this.topk = topk;
                this.topk2 = topk2 ?? topk;
                this.num_classes = num_classes;
                this.alpha = alpha;
                this.beta = beta;
                this.eps = eps;
                this.stride = stride ?? new int[] { 8, 16, 32 };
                this.stride_val = this.stride.Length > 1 ? this.stride[1] : this.stride[0];
            }

            public override (Tensor, Tensor, Tensor, Tensor, Tensor) forward(Tensor pd_scores, Tensor pd_bboxes, Tensor anc_points, Tensor gt_labels, Tensor gt_bboxes, Tensor mask_gt)
            {
                this.bs = pd_scores.shape[0];
                this.n_max_boxes = gt_bboxes.shape[1];
                Device device = gt_bboxes.device;
                if (this.n_max_boxes == 0)
                {
                    return (
                        torch.full_like(pd_scores[TensorIndex.Ellipsis, 0], this.num_classes, device: device),
                        torch.zeros_like(pd_bboxes, device: device),
                        torch.zeros_like(pd_scores, device: device),
                        torch.zeros_like(pd_scores[TensorIndex.Ellipsis, 0], device: device),
                        torch.zeros_like(pd_scores[TensorIndex.Ellipsis, 0], device: device)
                    );
                }
                return _forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt);
            }

            private (Tensor target_labels, Tensor target_bboxes, Tensor target_scores, Tensor fg_mask, Tensor target_gt_idx) _forward(Tensor pd_scores, Tensor pd_bboxes, Tensor anc_points, Tensor gt_labels, Tensor gt_bboxes, Tensor mask_gt)
            {
                using (NewDisposeScope())
                {
                    (Tensor mask_pos, Tensor align_metric, Tensor overlaps) = this.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt);
                    (Tensor target_gt_idx, Tensor fg_mask, mask_pos) = this.select_highest_overlaps(mask_pos, overlaps, this.n_max_boxes, align_metric);

                    // Assigned target
                    (Tensor target_labels, Tensor target_bboxes, Tensor target_scores) = this.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask);

                    // Normalize
                    align_metric *= mask_pos;
                    Tensor pos_align_metrics = align_metric.amax(dims: new long[] { -1 }, keepdim: true);  // b, max_num_obj
                    Tensor pos_overlaps = (overlaps * mask_pos).amax(dims: new long[] { -1 }, keepdim: true);  // b, max_num_obj
                    Tensor norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + this.eps)).amax(-2).unsqueeze(-1);
                    target_scores = target_scores * norm_align_metric;
                    return (target_labels.MoveToOuterDisposeScope(), target_bboxes.MoveToOuterDisposeScope(), target_scores.MoveToOuterDisposeScope(), fg_mask.@bool().MoveToOuterDisposeScope(), target_gt_idx.MoveToOuterDisposeScope());
                }
            }

            private (Tensor, Tensor, Tensor) get_pos_mask(Tensor pd_scores, Tensor pd_bboxes, Tensor gt_labels, Tensor gt_bboxes, Tensor anc_points, Tensor mask_gt)
            {
                using (NewDisposeScope())
                {
                    Tensor mask_in_gts = this.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt);
                    (Tensor align_metric, Tensor overlaps) = this.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt);
                    Tensor mask_topk = select_topk_candidates(align_metric, topk_mask: mask_gt.expand(-1, -1, topk).@bool());
                    Tensor mask_pos = mask_topk * mask_in_gts * mask_gt;

                    return (mask_pos.MoveToOuterDisposeScope(), align_metric.MoveToOuterDisposeScope(), overlaps.MoveToOuterDisposeScope());
                }
            }

            /// <summary>
            /// Compute alignment metric given predicted and ground truth bounding boxes.
            /// </summary>
            /// <param name="pd_scores">Predicted classification scores with shape (bs, num_total_anchors, num_classes).</param>
            /// <param name="pd_bboxes">Predicted bounding boxes with shape (bs, num_total_anchors, 4).</param>
            /// <param name="gt_labels">Ground truth labels with shape (bs, n_max_boxes, 1).</param>
            /// <param name="gt_bboxes">Ground truth boxes with shape (bs, n_max_boxes, 4).</param>
            /// <param name="mask_gt">Mask for valid ground truth boxes with shape (bs, n_max_boxes, h*w).</param>
            /// <returns><para>align_metric: Alignment metric combining classification and localization.</para></returns>
            private (Tensor align_metric, Tensor overlaps) get_box_metrics(Tensor pd_scores, Tensor pd_bboxes, Tensor gt_labels, Tensor gt_bboxes, Tensor mask_gt)
            {
                using (NewDisposeScope())
                {
                    long na = pd_bboxes.shape[^2];
                    mask_gt = mask_gt.@bool();

                    Tensor overlaps = torch.zeros(this.bs, this.n_max_boxes, na, dtype: pd_bboxes.dtype, device: pd_bboxes.device);
                    Tensor bbox_scores = torch.zeros(this.bs, this.n_max_boxes, na, dtype: pd_scores.dtype, device: pd_scores.device);

                    Tensor ind = torch.zeros(2, this.bs, this.n_max_boxes, dtype: torch.int64, device: gt_labels.device);
                    ind[0] = torch.arange(this.bs, dtype: torch.int64, device: gt_labels.device).view(-1, 1).expand(-1, this.n_max_boxes);
                    ind[1] = gt_labels.squeeze(-1);

                    bbox_scores[mask_gt] = pd_scores[TensorIndex.Tensor(ind[0]), TensorIndex.Colon, TensorIndex.Tensor(ind[1])][mask_gt]; // b, max_num_obj, h*w

                    Tensor pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, this.n_max_boxes, -1, -1)[mask_gt];
                    Tensor gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt];
                    overlaps[mask_gt] = iou_calculation(gt_boxes, pd_boxes);

                    Tensor align_metric = bbox_scores.pow(this.alpha) * overlaps.pow(this.beta);
                    return (align_metric.MoveToOuterDisposeScope(), overlaps.MoveToOuterDisposeScope());
                }
            }

            internal virtual Tensor iou_calculation(Tensor gt_bboxes, Tensor pd_bboxes)
            {
                return Metrics.bbox_iou(gt_bboxes, pd_bboxes, xywh: false, CIoU: true).squeeze(-1).clamp(0);
            }

            private Tensor select_topk_candidates(Tensor metrics, Tensor topk_mask = null)
            {
                using (NewDisposeScope())
                {
                    (Tensor topk_metrics, Tensor topk_idxs) = torch.topk(metrics, topk, dim: -1, largest: true);

                    if (topk_mask.IsInvalid)
                    {
                        topk_mask = (topk_metrics.max(0, keepdim: true).values > eps).expand_as(topk_idxs);
                    }

                    topk_idxs.masked_fill_(~topk_mask, 0);

                    Tensor count_tensor = torch.zeros(metrics.shape, dtype: torch.int8, device: topk_idxs.device);
                    Tensor ones = torch.ones_like(topk_idxs[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(0, 1)], dtype: torch.int8, device: topk_idxs.device);

                    for (int k = 0; k < topk; k++)
                    {
                        count_tensor.scatter_add_(dim: -1, index: topk_idxs[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(k, k + 1)], src: ones);
                    }

                    count_tensor.masked_fill_(count_tensor > 1, 0);
                    return count_tensor.to_type(metrics.dtype).MoveToOuterDisposeScope();
                }
            }

            private (Tensor, Tensor, Tensor) get_targets(Tensor gt_labels, Tensor gt_bboxes, Tensor target_gt_idx, Tensor fg_mask)
            {
                using (NewDisposeScope())
                {
                    // Assigned target labels, (b, 1)
                    Tensor batch_ind = torch.arange(bs, dtype: torch.int64, device: gt_labels.device)[TensorIndex.Ellipsis, TensorIndex.None];
                    target_gt_idx = target_gt_idx + batch_ind * this.n_max_boxes;
                    Tensor target_labels = gt_labels.@long().flatten()[target_gt_idx];

                    // Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
                    Tensor target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[^1])[target_gt_idx];

                    // Assigned target scores
                    target_labels = target_labels.clamp_(0);


                    // 10x faster than F.one_hot()
                    Tensor target_scores = torch.zeros(
                        target_labels.shape[0], target_labels.shape[1], num_classes,
                        dtype: torch.int64, device: target_labels.device
                    );

                    target_scores = target_scores.scatter_(dim: 2, index: target_labels.unsqueeze(-1), src: torch.ones_like(target_labels.unsqueeze(-1), dtype: torch.int64, device: target_labels.device));
                    //target_scores = target_scores.scatter_(dim: 2, index: target_labels.unsqueeze(-1), 1);

                    Tensor fg_scores_mask = fg_mask[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.None].repeat(1, 1, this.num_classes);
                    //target_scores = torch.where(fg_scores_mask > 0, target_scores, torch.zeros_like(target_scores));
                    target_scores = torch.where(fg_scores_mask > 0, target_scores, 0);
                    return (target_labels.MoveToOuterDisposeScope(), target_bboxes.MoveToOuterDisposeScope(), target_scores.MoveToOuterDisposeScope());
                }
            }

            internal virtual Tensor select_candidates_in_gts(Tensor xy_centers, Tensor gt_bboxes, Tensor mask_gt, float eps = 1e-9f)
            {
                using (NewDisposeScope())
                {
                    //long n_anchors = xy_centers.shape[0];
                    //long bs = gt_bboxes.shape[0];
                    //long n_boxes = gt_bboxes.shape[1];
                    //Tensor[] lt_rb = gt_bboxes.view(-1, 1, 4).chunk(2, dim: 2);
                    //Tensor lt = lt_rb[0].to(xy_centers.device);
                    //Tensor rb = lt_rb[1].to(xy_centers.device);

                    //Tensor bbox_deltas = torch.cat(new Tensor[] { xy_centers[TensorIndex.None] - lt, rb - xy_centers[TensorIndex.None] }, dim: 2).view(bs, n_boxes, n_anchors, -1);
                    //return bbox_deltas.amin(dims: new long[] { 3 }).gt_(eps).MoveToOuterDisposeScope();

                    Tensor gt_bboxes_xywh = Ops.xyxy2xywh(gt_bboxes);
                    Tensor wh_mask = gt_bboxes_xywh[TensorIndex.Ellipsis, 2..] < this.stride[0];  // the smallest stride
                    gt_bboxes_xywh[TensorIndex.Ellipsis, 2..] = torch.where(
                        (wh_mask * mask_gt).@bool(),
                        torch.tensor(this.stride_val, dtype: gt_bboxes_xywh.dtype, device: gt_bboxes_xywh.device),
                        gt_bboxes_xywh[TensorIndex.Ellipsis, 2..]);
                    gt_bboxes = Ops.xywh2xyxy(gt_bboxes_xywh);
                    long n_anchors = xy_centers.shape[0];
                    long bs = gt_bboxes.shape[0];
                    long n_boxes = gt_bboxes.shape[1];
                    //lt, rb
                    Tensor[] lt_rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2);  // left-top, right-bottom
                    Tensor lt = lt_rb[0];
                    Tensor rb = lt_rb[1];
                    Tensor bbox_deltas = torch.cat(new Tensor[] { xy_centers[TensorIndex.None] - lt, rb - xy_centers[TensorIndex.None] }, dim: 2).view(bs, n_boxes, n_anchors, -1);
                    return bbox_deltas.amin(3).gt_(eps).MoveToOuterDisposeScope();
                }
            }

            private (Tensor, Tensor, Tensor) select_highest_overlaps(Tensor mask_pos, Tensor overlaps, long n_max_boxes, Tensor align_metric)
            {
                using (NewDisposeScope())
                {
                    Tensor fg_mask = mask_pos.sum(dim: -2);

                    if (fg_mask.amax().ToSingle() > 1)
                    {
                        Tensor mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(bs, n_max_boxes, -1);
                        Tensor max_overlaps_idx = overlaps.argmax(dim: 1);

                        Tensor is_max_overlaps = torch.zeros_like(mask_pos, dtype: mask_pos.dtype, device: mask_pos.device);
                        is_max_overlaps.scatter_(dim: 1, index: max_overlaps_idx.unsqueeze(1), src: torch.ones_like(max_overlaps_idx.unsqueeze(1), dtype: mask_pos.dtype, device: mask_pos.device));

                        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).to_type(torch.ScalarType.Float32);
                        fg_mask = mask_pos.sum(dim: -2);
                    }
                    if (this.topk2 != topk)
                    {
                        align_metric = align_metric * mask_pos;  // update overlaps
                        Tensor max_overlaps_idx = torch.topk(align_metric, this.topk2, dim: -1, largest: true).indices;  // (b, n_max_boxes)
                        Tensor topk_idx = torch.zeros(mask_pos.shape, dtype: mask_pos.dtype, device: mask_pos.device);  // update mask_pos
                        topk_idx.scatter_(dim: -1, max_overlaps_idx, torch.ones_like(max_overlaps_idx, dtype: mask_pos.dtype, device: mask_pos.device));
                        mask_pos *= topk_idx;
                        fg_mask = mask_pos.sum(-2);
                    }

                    Tensor target_gt_idx = mask_pos.argmax(dim: -2);
                    return (target_gt_idx.MoveToOuterDisposeScope(), fg_mask.MoveToOuterDisposeScope(), mask_pos.MoveToOuterDisposeScope());
                }
            }


        }

        internal class RotatedTaskAlignedAssigner : TaskAlignedAssigner
        {
            internal RotatedTaskAlignedAssigner(int topk = 13, int num_classes = 80, float alpha = 1.0f, float beta = 6.0f, int[]? stride = null, float eps = 1e-9f, int? topk2 = null) : base(topk: topk, num_classes: num_classes, alpha: alpha, beta: beta, stride: stride, eps: eps, topk2: topk2)
            {

            }

            internal override Tensor iou_calculation(Tensor gt_bboxes, Tensor pd_bboxes)
            {
                return Metrics.probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0);
            }

            /// <summary>
            /// Select the positive anchor center in gt for rotated bounding boxes.
            /// </summary>
            /// <param name="xy_centers">Anchor center coordinates with shape (h*w, 2).</param>
            /// <param name="gt_bboxes">Ground truth bounding boxes with shape (b, n_boxes, 5).</param>
            /// <param name="mask_gt">Mask for valid ground truth boxes with shape (b, n_boxes, 1).</param>
            /// <returns>Boolean mask of positive anchors with shape (b, n_boxes, h*w).</returns>
            internal override Tensor select_candidates_in_gts(Tensor xy_centers, Tensor gt_bboxes, Tensor mask_gt, float eps = 1e-09f)
            {
                using (NewDisposeScope())
                {
                    Tensor wh_mask = gt_bboxes[TensorIndex.Ellipsis, 2..4] < this.stride[0];
                    gt_bboxes[TensorIndex.Ellipsis, 2..4] = torch.where(
                       (wh_mask * mask_gt).@bool(),
                       torch.tensor(this.stride_val, dtype: gt_bboxes.dtype, device: gt_bboxes.device),
                       gt_bboxes[TensorIndex.Ellipsis, 2..4]);

                    // (b, n_boxes, 5) --> (b, n_boxes, 4, 2)
                    Tensor corners = Ops.xywhr2xyxyxyxy(gt_bboxes);
                    // (b, n_boxes, 1, 2)
                    Tensor[] abcd = corners.split(1, dim: -2);
                    Tensor a = abcd[0];
                    Tensor b = abcd[1];
                    Tensor d = abcd[3];

                    Tensor ab = b - a;
                    Tensor ad = d - a;

                    // (b, n_boxes, h*w, 2)
                    Tensor ap = xy_centers - a;
                    Tensor norm_ab = (ab * ab).sum(dim: -1);
                    Tensor norm_ad = (ad * ad).sum(dim: -1);
                    Tensor ap_dot_ab = (ap * ab).sum(dim: -1);
                    Tensor ap_dot_ad = (ap * ad).sum(dim: -1);
                    return ((ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)).MoveToOuterDisposeScope();  // is_in_box
                }
            }

        }

        // Generate anchors from features.
        internal static (Tensor anchor_points, Tensor stride_tensor) make_anchors(Tensor[] feats, int[] strides, float grid_cell_offset = 0.5f)
        {
            using (NewDisposeScope())
            {
                torch.ScalarType dtype = feats[0].dtype;
                Device device = feats[0].device;
                List<Tensor> anchor_points = new List<Tensor>();
                List<Tensor> stride_tensor = new List<Tensor>();
                for (int i = 0; i < strides.Length; i++)
                {
                    long h = feats[i].shape[2];
                    long w = feats[i].shape[3];
                    Tensor sx = arange(w, device: device, dtype: dtype) + grid_cell_offset;  // shift x
                    Tensor sy = arange(h, device: device, dtype: dtype) + grid_cell_offset;  // shift y
                    Tensor[] sy_sx = meshgrid(new Tensor[] { sy, sx }, indexing: "ij");
                    sy = sy_sx[0];
                    sx = sy_sx[1];
                    anchor_points.Add(stack(new Tensor[] { sx, sy }, -1).view(-1, 2));
                    stride_tensor.Add(full(new long[] { h * w, 1 }, strides[i], dtype: dtype, device: device));
                }
                return (cat(anchor_points).MoveToOuterDisposeScope(), cat(stride_tensor).MoveToOuterDisposeScope());
            }
        }

        // Transform distance(ltrb) to box(xywh or xyxy).
        internal static Tensor dist2bbox(Tensor distance, Tensor anchor_points, bool xywh = true, int dim = -1)
        {
            using (NewDisposeScope())
            {
                Tensor[] ltrb = distance.chunk(2, dim);
                Tensor lt = ltrb[0];
                Tensor rb = ltrb[1];
                Tensor x1y1 = anchor_points - lt;
                Tensor x2y2 = anchor_points + rb;

                if (xywh)
                {
                    Tensor c_xy = (x1y1 + x2y2) / 2;
                    Tensor wh = x2y2 - x1y1;
                    return cat(new Tensor[] { c_xy, wh }, dim).MoveToOuterDisposeScope();  // xywh bbox
                }
                return torch.cat(new Tensor[] { x1y1, x2y2 }, dim).MoveToOuterDisposeScope(); // xyxy bbox
            }
        }

        /// <summary>
        /// Transform bbox(xyxy) to dist(ltrb).
        /// </summary>
        /// <param name="anchor_points"></param>
        /// <param name="bbox"></param>
        /// <param name="reg_max"></param>
        /// <returns></returns>
        internal static Tensor bbox2dist(Tensor anchor_points, Tensor bbox, int? reg_max = null)
        {
            using (NewDisposeScope())
            {
                Tensor[] x1y1x2y2 = bbox.chunk(2, -1);
                Tensor x1y1 = x1y1x2y2[0];
                Tensor x2y2 = x1y1x2y2[1];
                Tensor dist = torch.cat(new Tensor[] { anchor_points - x1y1, x2y2 - anchor_points }, -1);
                if (reg_max is not null)
                {
                    dist = dist.clamp_(0, reg_max - 0.01);  // dist (lt, rb)
                }
                return dist.MoveToOuterDisposeScope();
            }
        }

        /// <summary>
        /// Decode predicted rotated bounding box coordinates from anchor points and distribution.
        /// </summary>
        /// <param name="pred_dist">Predicted rotated distance with shape (bs, h*w, 4).</param>
        /// <param name="pred_angle">Predicted angle with shape (bs, h*w, 1).</param>
        /// <param name="anchor_points">Anchor points with shape (h*w, 2).</param>
        /// <param name="dim">Dimension along which to split. Defaults to -1.</param>
        /// <returns>Predicted rotated bounding boxes with shape (bs, h*w, 4).</returns>
        internal static Tensor dist2rbox(Tensor pred_dist, Tensor pred_angle, Tensor anchor_points, int dim = -1)
        {
            using (NewDisposeScope())
            {
                Tensor[] lt_rb = pred_dist.split(2, dim: dim);
                Tensor lt = lt_rb[0]; // (bs, h*w, 2)
                Tensor rb = lt_rb[1]; // (bs, h*w, 2)

                Tensor cos = torch.cos(pred_angle);
                Tensor sin = torch.sin(pred_angle);
                // (bs, h*w, 1)
                Tensor[] xf_yf = ((rb - lt) / 2).split(1, dim: dim);
                Tensor xf = xf_yf[0]; // (bs, h*w, 1)
                Tensor yf = xf_yf[1]; // (bs, h*w, 1)
                Tensor x = xf * cos - yf * sin;
                Tensor y = xf * sin + yf * cos;
                Tensor xy = torch.cat(new Tensor[] { x, y }, dim: dim) + anchor_points;
                return torch.cat(new Tensor[] { xy, lt + rb }, dim: dim).MoveToOuterDisposeScope();
            }
        }

        /// <summary>
        /// Transform rotated bounding box (xywh) to distance (ltrb). This is the inverse of dist2rbox.
        /// </summary>
        /// <param name="target_bboxes">Target rotated bounding boxes with shape (bs, h*w, 4), format [x, y, w, h].</param>
        /// <param name="anchor_points">Anchor points with shape (h*w, 2).</param>
        /// <param name="target_angle">Target angle with shape (bs, h*w, 1).</param>
        /// <param name="dim">Dimension along which to split.</param>
        /// <param name="reg_max">Maximum regression value for clamping.</param>
        /// <returns>Rotated distance with shape (bs, h*w, 4), format [l, t, r, b].</returns>
        internal static torch.Tensor rbox2dist(torch.Tensor target_bboxes, torch.Tensor anchor_points, torch.Tensor target_angle, int dim = -1, int? reg_max = null)
        {
            using (torch.NewDisposeScope())
            {
                torch.Tensor[] xy_wh = target_bboxes.split(2, dim: dim);
                torch.Tensor xy = xy_wh[0];
                torch.Tensor wh = xy_wh[1];
                torch.Tensor offset = xy - anchor_points;  // (bs, h*w, 2)
                torch.Tensor[] offset_x_y = offset.split(1, dim: dim);

                torch.Tensor offset_x = offset_x_y[0]; // (bs, h*w, 1)
                torch.Tensor offset_y = offset_x_y[1]; // (bs, h*w, 1)
                torch.Tensor cos = torch.cos(target_angle);
                torch.Tensor sin = torch.sin(target_angle);
                torch.Tensor xf = offset_x * cos + offset_y * sin;
                torch.Tensor yf = -offset_x * sin + offset_y * cos;
                torch.Tensor[] w__h = wh.split(1, dim: dim);

                torch.Tensor w = w__h[0]; // (bs, h*w, 1)
                torch.Tensor h = w__h[1]; // (bs, h*w, 1)

                torch.Tensor target_l = w / 2 - xf;
                torch.Tensor target_t = h / 2 - yf;
                torch.Tensor target_r = w / 2 + xf;
                torch.Tensor target_b = h / 2 + yf;

                torch.Tensor dist = torch.cat(new torch.Tensor[] { target_l, target_t, target_r, target_b }, dim: dim);
                if (reg_max is not null)
                {
                    dist = dist.clamp_(0, reg_max - 0.01);
                }
                return dist.MoveToOuterDisposeScope();
            }

        }
    }

}
