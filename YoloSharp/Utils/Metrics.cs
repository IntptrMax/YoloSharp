using TorchSharp;
using static TorchSharp.torch;

namespace YoloSharp.Utils
{
    internal class Metrics
    {
        /// <summary>
        /// Calculate intersection-over-union (IoU) of boxes.
        /// </summary>
        /// <param name="box1">A tensor of shape (N, 4) representing N bounding boxes in (x1, y1, x2, y2) format.</param>
        /// <param name="box2">A tensor of shape (M, 4) representing M bounding boxes in (x1, y1, x2, y2) format.</param>
        /// <param name="eps">A small value to avoid division by zero.</param>
        /// <returns>An NxM tensor containing the pairwise IoU values for every element in box1 and box2.</returns>
        internal static Tensor box_iou(Tensor box1, Tensor box2, float eps = 1e-7f)
        {
            using (NewDisposeScope())
            using (no_grad())
            {
                // NOTE: Need .float() to get accurate iou values
                // inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
                Tensor[] a = box1.@float().unsqueeze(1).chunk(2, 2);
                Tensor a1 = a[0];
                Tensor a2 = a[1];
                Tensor[] b = box2.@float().unsqueeze(0).chunk(2, 2);
                Tensor b1 = b[0];
                Tensor b2 = b[1];

                Tensor inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2);

                // IoU = inter / (area1 + area2 - inter)
                return (inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)).MoveToOuterDisposeScope();
            }
        }

        /// <summary>
        /// Calculate masks IoU.
        /// </summary>
        /// <param name="mask1">A tensor of shape (N, n) where N is the number of ground truth objects and n is the product of image width and height.</param>
        /// <param name="mask2">A tensor of shape (M, n) where M is the number of predicted objects and n is the product of image width and height.</param>
        /// <param name="eps">A small value to avoid division by zero.</param>
        /// <returns>A tensor of shape (N, M) representing masks IoU.</returns>
        internal static Tensor mask_iou(torch.Tensor mask1, torch.Tensor mask2, float eps = 1e-7f)
        {
            Tensor intersection = torch.matmul(mask1, mask2.T).clamp_(0);
            Tensor union = (mask1.sum(1)[.., TensorIndex.None] + mask2.sum(1)[TensorIndex.None]) - intersection; // (area1 + area2) - intersection
            return intersection / (union + eps);
        }

        /// <summary>
        /// Calculate probabilistic IoU between oriented bounding boxes.
        /// <para>OBB format: [center_x, center_y, width, height, rotation_angle].</para>
        /// <para>https://arxiv.org/pdf/2106.06072v1.pdf</para>
        /// </summary>
        /// <param name="obb1">Ground truth OBBs, shape (N, 5), format xywhr.</param>
        /// <param name="obb2">Predicted OBBs, shape (N, 5), format xywhr.</param>
        /// <param name="CIoU">If True, calculate CIoU.</param>
        /// <param name="eps">Small value to avoid division by zero.</param>
        /// <returns>OBB similarities, shape (N,).</returns>
        internal static Tensor probiou(Tensor obb1, Tensor obb2, bool CIoU = false, float eps = 1e-7f)
        {
            using (NewDisposeScope())
            using (no_grad())
            {
                Tensor x1 = obb1[.., 0..1];
                Tensor y1 = obb1[.., 1..2];
                Tensor x2 = obb2[.., 0..1];
                Tensor y2 = obb2[.., 1..2];

                (Tensor a1, Tensor b1, Tensor c1) = _get_covariance_matrix(obb1);
                (Tensor a2, Tensor b2, Tensor c2) = _get_covariance_matrix(obb2);

                Tensor t1 = (((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.25;
                Tensor t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5;
                Tensor t3 = (((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2)) / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps) + eps).log() * 0.5;

                Tensor bd = (t1 + t2 + t3).clamp(eps, 100.0);
                Tensor hd = (1.0 - (-bd).exp() + eps).sqrt();
                Tensor iou = 1 - hd;

                if (CIoU)  // only include the wh aspect ratio part
                {
                    Tensor w1 = obb1[.., 2];
                    Tensor h1 = obb1[.., 3];
                    Tensor w2 = obb2[.., 2];
                    Tensor h2 = obb2[.., 3];

                    Tensor v = (4 / Math.Pow(Math.PI, 2)) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2);

                    using (torch.no_grad())
                    {
                        Tensor alpha = v / (v - iou + (1 + eps));
                        return iou - v * alpha;  // CIoU
                    }
                }
                return iou.MoveToOuterDisposeScope();
            }
        }

        /// <summary>
        /// Calculate Object Keypoint Similarity (OKS).
        /// </summary>
        /// <param name="kpt1">A tensor of shape (N, 17, 3) representing ground truth keypoints.</param>
        /// <param name="kpt2">A tensor of shape (M, 17, 3) representing predicted keypoints.</param>
        /// <param name="area">A tensor of shape (N,) representing areas from ground truth.</param>
        /// <param name="sigma">A list containing 17 values representing keypoint scales.</param>
        /// <param name="eps">A small value to avoid division by zero.</param>
        /// <returns>A tensor of shape (N, M) representing keypoint similarities.</returns>
        internal static Tensor kpt_iou(Tensor kpt1, Tensor kpt2, Tensor area, float[] sigma, float eps = 1e-7f)
        {
            using (NewDisposeScope())
            using (no_grad())
            {
                Tensor d = (kpt1[.., TensorIndex.None, .., 0] - kpt2[TensorIndex.Ellipsis, 0]).pow(2) + (kpt1[.., TensorIndex.None, .., 1] - kpt2[TensorIndex.Ellipsis, 1]).pow(2);    //   (N, M, 17)
                Tensor sigma_tensor = torch.tensor(sigma, device: kpt1.device, dtype: kpt1.dtype);  // (17, )
                Tensor kpt_mask = kpt1[TensorIndex.Ellipsis, 2] != 0;  // (N, 17)
                Tensor e = d / ((2 * sigma_tensor).pow(2) * (area[.., TensorIndex.None, TensorIndex.None] + eps) * 2);  // from cocoeval

                // e = d / ((area[None, :, None] + eps) * sigma) ** 2 / 2  # from formula
                Tensor kpt_iou = ((-e).exp() * kpt_mask[..,TensorIndex. None]).sum(-1) / (kpt_mask.sum(-1)[..,TensorIndex. None] + eps);
                return kpt_iou.MoveToOuterDisposeScope();
            }
        }

        /// <summary>
        /// Calculate the probabilistic IoU between oriented bounding boxes.
        /// <para>https://arxiv.org/pdf/2106.06072v1.pdf</para>
        /// </summary>
        /// <param name="obb1">A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.</param>
        /// <param name="obb2">A tensor of shape (M, 5) representing predicted obbs, with xywhr format.</param>
        /// <param name="eps">A small value to avoid division by zero.</param>
        /// <returns>A tensor of shape (N, M) representing obb similarities.</returns>
        internal static Tensor batch_probiou(Tensor obb1, Tensor obb2, float eps = 1e-7f)
        {
            using (NewDisposeScope())
            using (no_grad())
            {
                // Split coordinates and get covariance matrices
                Tensor x1 = obb1[.., 0].unsqueeze(-1);
                Tensor y1 = obb1[.., 1].unsqueeze(-1);
                Tensor x2 = obb2[.., 0].unsqueeze(0);
                Tensor y2 = obb2[.., 1].unsqueeze(0);

                (Tensor a1, Tensor b1, Tensor c1) = _get_covariance_matrix(obb1);
                (Tensor a2, Tensor b2, Tensor c2) = _get_covariance_matrix(obb2);

                a2 = a2.squeeze(-1)[TensorIndex.None];
                b2 = b2.squeeze(-1)[TensorIndex.None];
                c2 = c2.squeeze(-1)[TensorIndex.None];

                //Prepare tensors for broadcasting
                Tensor t1 = (((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.25;
                Tensor t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5;
                Tensor t3 = (
                       ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
                       / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
                       + eps
                   ).log() * 0.5;

                Tensor bd = (t1 + t2 + t3).clamp(eps, 100.0);
                Tensor hd = (1.0 - (-bd).exp() + eps).sqrt();

                return (1 - hd).MoveToOuterDisposeScope();
            }
        }




        /// <summary>
        /// Generate covariance matrix from oriented bounding boxes.
        /// </summary>
        /// <param name="obb"> A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.</param>
        /// <returns>Covariance matrices corresponding to original rotated bounding boxes.</returns>
        private static (Tensor a, Tensor b, Tensor c) _get_covariance_matrix(Tensor boxes)
        {
            using (NewDisposeScope())
            using (no_grad())
            {
                // Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
                Tensor gbbs = torch.cat(new Tensor[] { boxes[.., 2..4].pow(2) / 12, boxes[.., 4..] }, dim: -1);

                Tensor[] abc = gbbs.split(1, dim: -1);
                Tensor a = abc[0];
                Tensor b = abc[1];
                Tensor c = abc[2];

                Tensor cos = c.cos();
                Tensor sin = c.sin();
                Tensor cos2 = cos.pow(2);
                Tensor sin2 = sin.pow(2);

                return ((a * cos2 + b * sin2).MoveToOuterDisposeScope(), (a * sin2 + b * cos2).MoveToOuterDisposeScope(), ((a - b) * cos * sin).MoveToOuterDisposeScope());
            }
        }

        /// <summary>
        /// Calculate the Intersection over Union (IoU) between bounding boxes.
        /// </summary>
        /// <param name="box1">A tensor representing one or more bounding boxes, with the last dimension being 4.</param>
        /// <param name="box2">A tensor representing one or more bounding boxes, with the last dimension being 4.</param>
        /// <param name="xywh">If True, input boxes are in (x, y, w, h) format. If False, input boxes are in (x1, y1, x2, y2) format.</param>
        /// <param name="GIoU">If True, calculate Generalized IoU.</param>
        /// <param name="DIoU">If True, calculate Distance IoU.</param>
        /// <param name="CIoU">If True, calculate Complete IoU.</param>
        /// <param name="eps">A small value to avoid division by zero.</param>
        /// <returns>IoU, GIoU, DIoU, or CIoU values depending on the specified flags.</returns>
        internal static Tensor bbox_iou(Tensor box1, Tensor box2, bool xywh = true, bool GIoU = false, bool DIoU = false, bool CIoU = false, float eps = 1e-7f)
        {
            using (NewDisposeScope())
            using (no_grad())
            {
                Tensor b1_x1, b1_x2, b1_y1, b1_y2;
                Tensor b2_x1, b2_x2, b2_y1, b2_y2;
                Tensor w1, h1, w2, h2;

                if (xywh)  // transform from xywh to xyxy
                {
                    Tensor[] xywh1 = box1.chunk(4, -1);
                    Tensor x1 = xywh1[0];
                    Tensor y1 = xywh1[1];
                    w1 = xywh1[2];
                    h1 = xywh1[3];

                    Tensor[] xywh2 = box2.chunk(4, -1);
                    Tensor x2 = xywh2[0];
                    Tensor y2 = xywh2[1];
                    w2 = xywh2[2];
                    h2 = xywh2[3];

                    var (w1_, h1_, w2_, h2_) = (w1 / 2, h1 / 2, w2 / 2, h2 / 2);
                    (b1_x1, b1_x2, b1_y1, b1_y2) = (x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_);
                    (b2_x1, b2_x2, b2_y1, b2_y2) = (x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_);
                }

                else  // x1, y1, x2, y2 = box1
                {
                    Tensor[] b1x1y1x2y2 = box1.chunk(4, -1);
                    b1_x1 = b1x1y1x2y2[0];
                    b1_y1 = b1x1y1x2y2[1];
                    b1_x2 = b1x1y1x2y2[2];
                    b1_y2 = b1x1y1x2y2[3];

                    Tensor[] b2x1y1x2y2 = box2.chunk(4, -1);
                    b2_x1 = b2x1y1x2y2[0];
                    b2_y1 = b2x1y1x2y2[1];
                    b2_x2 = b2x1y1x2y2[2];
                    b2_y2 = b2x1y1x2y2[3];

                    (w1, h1) = (b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps));
                    (w2, h2) = (b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps));
                }

                // Intersection area
                Tensor inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0);

                // Union Area
                Tensor union = w1 * h1 + w2 * h2 - inter + eps;

                // IoU
                Tensor iou = inter / union;
                if (CIoU || DIoU || GIoU)
                {
                    Tensor cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1);  //convex (smallest enclosing box) width
                    Tensor ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1);  // convex height
                    if (CIoU || DIoU)  // Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                    {
                        Tensor c2 = cw.pow(2) + ch.pow(2) + eps;   //convex diagonal squared
                        Tensor rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4;   //center dist ** 2

                        if (CIoU)  // https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                        {
                            Tensor v = 4 / (MathF.PI * MathF.PI) * (atan(w2 / h2) - atan(w1 / h1)).pow(2);

                            {
                                Tensor alpha = v / (v - iou + (1 + eps));
                                return (iou - (rho2 / c2 + v * alpha)).MoveToOuterDisposeScope();  //CIoU
                            }
                        }
                        return (iou - rho2 / c2).MoveToOuterDisposeScope();  // DIoU
                    }
                    Tensor c_area = cw * ch + eps;    // convex area
                    return (iou - (c_area - union) / c_area).MoveToOuterDisposeScope();  // GIoU https://arxiv.org/pdf/1902.09630.pdf
                }
                return iou.MoveToOuterDisposeScope(); //IoU
            }
        }

        /// <summary>
        /// Compute the average precision per class for object detection evaluation.
        /// </summary>
        /// <param name="tp">Binary array indicating whether the detection is correct (True) or not (False).</param>
        /// <param name="conf">Array of confidence scores of the detections.</param>
        /// <param name="pred_cls">Array of predicted classes of the detections.</param>
        /// <param name="target_cls">Array of true classes of the detections.</param>
        /// <param name="eps">A small value to avoid division by zero.</param>
        /// <param name="prefix">A prefix string for saving the plot files.</param>
        /// <returns>
        /// tp: True positive counts at threshold given by max F1 metric for each class.<br/>
        /// fp: False positive counts at threshold given by max F1 metric for each class.<br/>
        /// p: Precision values at threshold given by max F1 metric for each class.<br/>
        /// r: Recall values at threshold given by max F1 metric for each class.<br/>
        /// f1: F1-score values at threshold given by max F1 metric for each class.<br/>
        ///	ap: Average precision for each class at different IoU thresholds.<br/>
        ///	unique_classes: An array of unique classes that have data.<br/>
        /// p_curve: Precision curves for each class.<br/>
        /// r_curve: Recall curves for each class.<br/>
        /// f1_curve: F1-score curves for each class.<br/>
        /// x: X-axis values for the curves.<br/>
        /// prec_values: Precision values at mAP@0.5 for each class.<br/>
        /// </returns>
        internal static (Tensor tp, Tensor fp, Tensor p, Tensor r, Tensor f1, Tensor ap, Tensor unique_classes, Tensor p_curve, Tensor r_curve, Tensor f1_curve, Tensor x, Tensor prec_values) ap_per_class(Tensor tp, Tensor conf, Tensor pred_cls, Tensor target_cls, float eps = 1e-16f)
        {
            using (NewDisposeScope())
            using (no_grad())
            {
                // Sort by objectness
                Tensor ii = torch.argsort(-conf);
                tp = tp[ii];
                conf = conf[ii];
                pred_cls = pred_cls[ii];

                // Find unique classes
                (Tensor unique_classes, _, Tensor nt) = torch.unique(target_cls, return_counts: true);
                long nc = unique_classes.shape[0];  // number of classes, number of detections

                // Create Precision-Recall curve and compute AP for each class
                Tensor x = torch.linspace(0, 1, 1000, device: conf.device);
                List<Tensor> prec_values = new List<Tensor>();
                // Average precision, precision and recall curves

                Tensor ap = torch.zeros(new long[] { nc, tp.shape[1] }, device: tp.device);
                Tensor p_curve = torch.zeros(new long[] { nc, 1000 }, device: tp.device);
                Tensor r_curve = torch.zeros(new long[] { nc, 1000 }, device: tp.device);

                for (int ci = 0; ci < nc; ci++)
                {
                    Tensor c = unique_classes[ci];
                    Tensor i = (pred_cls == c);
                    long n_l = nt[ci].ToInt64();  // number of labels
                    long n_p = i.sum().ToInt64();  // number of predictions

                    if (n_p == 0 || n_l == 0)
                    {
                        continue;
                    }

                    // Accumulate FPs and TPs
                    Tensor fpc = (~tp[i]).cumsum(0);
                    Tensor tpc = tp[i].cumsum(0);

                    // Recall
                    Tensor recall = tpc / (n_l + eps);  // recall curve
                    r_curve[ci] = interp(-x, -conf[i], recall[.., 0], left: 0);

                    // Precision
                    Tensor precision = tpc / (tpc + fpc); // precision curve
                    p_curve[ci] = interp(-x, -conf[i], precision[.., 0], left: 1);  // p at pr_score

                    // AP from recall-precision curve
                    for (int j = 0; j < tp.shape[1]; j++)
                    {
                        (ap[ci, j], Tensor mpre, Tensor mrec) = compute_ap(recall[.., j], precision[.., j]);
                        if (j == 0)
                        {
                            prec_values.Add(interp(x, mrec, mpre)); // precision at mAP@0.5
                        }
                    }
                }

                if (prec_values.Count < 1)
                {
                    prec_values = new List<Tensor> { torch.zeros(new long[] { 1, 1000 }) };
                }

                // Compute F1 (harmonic mean of precision and recall)
                Tensor f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps);
                Tensor iii = smooth(f1_curve.mean(new long[] { 0 }), 0.1f).argmax();  // max F1 index

                Tensor p = p_curve[TensorIndex.Ellipsis, TensorIndex.Tensor(iii)];
                Tensor r = r_curve[TensorIndex.Ellipsis, TensorIndex.Tensor(iii)];
                Tensor f1 = f1_curve[TensorIndex.Ellipsis, TensorIndex.Tensor(iii)];

                tp = (r * nt).round();  // true positives
                Tensor fp = (tp / (p + eps) - tp).round();  // false positives
                return (tp.MoveToOuterDisposeScope(), fp.MoveToOuterDisposeScope(), p.MoveToOuterDisposeScope(), r.MoveToOuterDisposeScope(), f1.MoveToOuterDisposeScope(), ap.MoveToOuterDisposeScope(), unique_classes.@int().MoveToOuterDisposeScope(), p_curve.MoveToOuterDisposeScope(), r_curve.MoveToOuterDisposeScope(), f1_curve.MoveToOuterDisposeScope(), x.MoveToOuterDisposeScope(), torch.stack(prec_values).MoveToOuterDisposeScope());
            }
        }

        /// <summary>
        /// Compute the average precision (AP) given the recall and precision curves.
        /// </summary>
        /// <param name="recall">The recall curve.</param>
        /// <param name="precision">The precision curve.</param>
        /// <returns>
        /// ap: Average precision.<br/>
        /// mpre: Precision envelope curve.<br/>
        /// mrec: Modified recall curve with sentinel values added at the beginning and end.
        /// </returns>
        internal static (float ap, Tensor mpre, Tensor mrec) compute_ap(Tensor recall, Tensor precision)
        {
            // Append sentinel values to beginning and end

            Tensor mrec = torch.cat(new torch.Tensor[] { torch.tensor(new float[] { 0.0f }, device: recall.device), recall, torch.tensor(new float[] { 1.0f }, device: recall.device) });
            Tensor mpre = torch.cat(new torch.Tensor[] { torch.tensor(new float[] { 1.0f }, device: precision.device), precision, torch.tensor(new float[] { 0.0f }, device: recall.device) });

            mpre = mpre.flip(0).cummax(0).values.flip(0);

            // Compute the precision envelope
            float ap = 0.0f;
            // Integrate area under curve
            string method = "interp"; // methods: 'continuous', 'interp'
            if (method == "interp")
            {
                Tensor x = torch.linspace(0, 1, 101, device: mrec.device); // 101-point interp (COCO)

                // Integrate using trapezoidal rule
                ap = torch.trapezoid(interp(x, mrec, mpre), x).ToSingle();
            }
            else // 'continuous'
            {
                Tensor i = torch.where(mrec[1..] != mrec[..-1])[0];
                ap = torch.sum((mrec.index_select(0, i + 1) - mrec.index_select(0, i)) * mpre.index_select(0, i + 1)).item<float>();
            }
            return (ap, mpre, mrec);
        }


        public static Tensor interp(Tensor x, Tensor xp, Tensor fp, double left = 0)
        {
            using (NewDisposeScope())
            using (no_grad())
            {
                if (xp.dim() != 1 || fp.dim() != 1)
                {
                    throw new ArgumentException("xp and fp must be 1D tensors");
                }

                Tensor indices = torch.argsort(xp);
                Tensor xp_sorted = xp.index(indices).contiguous();
                Tensor fp_sorted = fp.index(indices).contiguous();

                Tensor result = torch.empty_like(x);

                using (Tensor left_mask = x <= xp_sorted[0])
                using (Tensor right_mask = x >= xp_sorted[-1])
                {
                    result[right_mask] = fp_sorted[-1];
                    result[left_mask] = left;
                }

                Tensor interior_mask = (x > xp_sorted[0]) & (x < xp_sorted[-1]);

                if (interior_mask.sum().ToInt64() > 0)
                {
                    Tensor x_interior = x[interior_mask];

                    Tensor indices_tensor = torch.searchsorted(xp_sorted, x_interior) - 1;
                    indices_tensor = torch.clamp(indices_tensor, 0, xp_sorted.size(0) - 2);

                    Tensor x0 = xp_sorted.gather(0, indices_tensor);
                    Tensor x1 = xp_sorted.gather(0, indices_tensor + 1);
                    Tensor y0 = fp_sorted.gather(0, indices_tensor);
                    Tensor y1 = fp_sorted.gather(0, indices_tensor + 1);

                    Tensor t = (x_interior - x0) / (x1 - x0);
                    Tensor interpolated = y0 + t * (y1 - y0);

                    result[interior_mask] = interpolated;
                }

                return result.MoveToOuterDisposeScope();
            }
        }

        /// <summary>
        /// Box filter of fraction f.
        /// </summary>
        private static Tensor smooth(Tensor y, float f = 0.05f)
        {
            int nf = (int)(y.shape[0] * f * 2) / 2 * 2 + 1;  // number of filter elements (must be odd)
            Tensor p = torch.ones(nf / 2, device: y.device) * y[0];  // ones padding
            Tensor yp = torch.cat(new[] { p, y, p });  // y padded

            // Simple convolution for smoothing
            Tensor kernel = torch.ones(nf, device: yp.device) / nf;
            Tensor result = torch.nn.functional.conv1d(yp.view(new long[] { 1, 1, -1 }), kernel.view(new long[] { 1, 1, -1 }), padding: 0);
            return result;

        }


    }
}
