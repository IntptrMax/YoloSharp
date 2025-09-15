using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Utils
{
	internal class Tal
	{
		internal class TaskAlignedAssigner : Module<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, (Tensor, Tensor, Tensor, Tensor, Tensor)>
		{
			private readonly int topk;
			private readonly int num_classes;
			private readonly int bg_idx;
			private readonly float alpha;
			private readonly float beta;
			private readonly float eps;
			private long bs;
			private long n_max_boxes;

			internal TaskAlignedAssigner(int topk = 13, int num_classes = 80, float alpha = 1.0f, float beta = 6.0f, float eps = 1e-9f) : base("TaskAlignedAssigner")
			{
				this.topk = topk;
				this.num_classes = num_classes;
				this.bg_idx = num_classes;
				this.alpha = alpha;
				this.beta = beta;
				this.eps = eps;
			}

			public override (Tensor, Tensor, Tensor, Tensor, Tensor) forward(Tensor pd_scores, Tensor pd_bboxes, Tensor anc_points, Tensor gt_labels, Tensor gt_bboxes, Tensor mask_gt)
			{
				using (NewDisposeScope())
				{
					bs = pd_scores.shape[0];
					n_max_boxes = gt_bboxes.shape[1];

					if (n_max_boxes == 0)
					{
						Device device = gt_bboxes.device;
						return (
							torch.full_like(pd_scores[TensorIndex.Ellipsis, 0], bg_idx, device: device),
							torch.zeros_like(pd_bboxes, device: device),
							torch.zeros_like(pd_scores, device: device),
							torch.zeros_like(pd_scores[TensorIndex.Ellipsis, 0], device: device),
							torch.zeros_like(pd_scores[TensorIndex.Ellipsis, 0], device: device)
						);
					}

					(Tensor mask_pos, Tensor align_metric, Tensor overlaps) = get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt);
					(Tensor target_gt_idx, Tensor fg_mask, Tensor updated_mask_pos) = select_highest_overlaps(mask_pos, overlaps, n_max_boxes);
					(Tensor target_labels, Tensor target_bboxes, Tensor target_scores) = get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask);

					align_metric *= updated_mask_pos;
					Tensor pos_align_metrics = align_metric.amax(dims: new long[] { -1 }, keepdim: true);
					Tensor pos_overlaps = (overlaps * updated_mask_pos).amax(dims: new long[] { -1 }, keepdim: true);
					Tensor norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + eps)).amax(dims: new long[] { -2 }).unsqueeze(-1);
					target_scores = target_scores * norm_align_metric;

					return (target_labels.MoveToOuterDisposeScope(), target_bboxes.MoveToOuterDisposeScope(), target_scores.MoveToOuterDisposeScope(), fg_mask.to_type(torch.ScalarType.Bool).MoveToOuterDisposeScope(), target_gt_idx.MoveToOuterDisposeScope());
				}
			}

			private (Tensor, Tensor, Tensor) get_pos_mask(Tensor pd_scores, Tensor pd_bboxes, Tensor gt_labels, Tensor gt_bboxes, Tensor anc_points, Tensor mask_gt)
			{
				using (NewDisposeScope())
				{
					Tensor mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes);
					(Tensor align_metric, Tensor overlaps) = get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt);
					Tensor mask_topk = select_topk_candidates(align_metric, topk_mask: mask_gt.expand(bs, n_max_boxes, topk).to_type(torch.ScalarType.Bool));
					Tensor mask_pos = mask_topk * mask_in_gts * mask_gt;

					return (mask_pos.MoveToOuterDisposeScope(), align_metric.MoveToOuterDisposeScope(), overlaps.MoveToOuterDisposeScope());
				}
			}
			private (Tensor, Tensor) get_box_metrics(Tensor pd_scores, Tensor pd_bboxes, Tensor gt_labels, Tensor gt_bboxes, Tensor mask_gt)
			{
				using (NewDisposeScope())
				{
					long na = pd_bboxes.shape[1];
					mask_gt = mask_gt.to_type(torch.ScalarType.Bool);

					Tensor overlaps = torch.zeros(bs, n_max_boxes, na, dtype: pd_bboxes.dtype, device: pd_bboxes.device);
					Tensor bbox_scores = torch.zeros(bs, n_max_boxes, na, dtype: pd_scores.dtype, device: pd_scores.device);

					Tensor ind = torch.zeros(2, bs, n_max_boxes, dtype: torch.int64, device: gt_labels.device);
					ind[0] = torch.arange(bs, dtype: torch.int64, device: gt_labels.device).view(-1, 1).expand(-1, n_max_boxes);
					ind[1] = gt_labels.squeeze(-1);

					Tensor pd_scores_selected = torch.zeros(this.bs, n_max_boxes, na, dtype: pd_scores.dtype, device: pd_scores.device);
					for (int i = 0; i < ind.shape[1]; i++)
					{
						for (int j = 0; j < ind.shape[2]; j++)
						{
							pd_scores_selected[i, j] = pd_scores[ind[0][i, j].ToInt64(), TensorIndex.Colon, ind[1][i, j].ToInt64()];
						}
					}

					bbox_scores[mask_gt] = pd_scores_selected[mask_gt];
					Tensor pd_boxes = pd_bboxes.unsqueeze(1).expand(bs, n_max_boxes, na, 4)[mask_gt];
					Tensor gt_boxes = gt_bboxes.unsqueeze(2).expand(bs, n_max_boxes, na, 4)[mask_gt];
					overlaps[mask_gt] = iou_calculation(gt_boxes, pd_boxes);

					Tensor align_metric = bbox_scores.pow(alpha) * overlaps.pow(beta);
					return (align_metric.MoveToOuterDisposeScope(), overlaps.MoveToOuterDisposeScope());
				}
			}

			private Tensor iou_calculation(Tensor gt_bboxes, Tensor pd_bboxes)
			{
				return bbox_iou(gt_bboxes, pd_bboxes, xywh: false, CIoU: true).squeeze(-1).clamp(0);
			}

			private Tensor select_topk_candidates(Tensor metrics, bool largest = true, Tensor topk_mask = null)
			{
				using (NewDisposeScope())
				{
					Tensor topk_metrics = torch.topk(metrics, topk, dim: -1, largest: largest).values;
					Tensor topk_idxs = torch.topk(metrics, topk, dim: -1, largest: largest).indices;

					if (topk_mask.IsInvalid)
					{
						topk_mask = (topk_metrics.amax(dims: new long[] { -1 }, keepdim: true) > eps).expand_as(topk_idxs);
					}

					topk_idxs.masked_fill_(~topk_mask, 0);

					Tensor count_tensor = torch.zeros_like(metrics, dtype: torch.int8, device: topk_idxs.device);
					Tensor ones = torch.ones_like(topk_idxs[TensorIndex.Ellipsis, TensorIndex.Slice(0, 1)], dtype: torch.int8, device: topk_idxs.device);

					for (int k = 0; k < topk; k++)
					{
						count_tensor.scatter_add_(dim: -1, index: topk_idxs[TensorIndex.Ellipsis, k].unsqueeze(-1), src: ones);
					}

					count_tensor.masked_fill_(count_tensor > 1, 0);
					return count_tensor.to_type(metrics.dtype).MoveToOuterDisposeScope();
				}
			}

			private (Tensor, Tensor, Tensor) get_targets(Tensor gt_labels, Tensor gt_bboxes, Tensor target_gt_idx, Tensor fg_mask)
			{
				using (NewDisposeScope())
				{
					Tensor batch_ind = torch.arange(bs, dtype: torch.int64, device: gt_labels.device).view(-1, 1);
					target_gt_idx = target_gt_idx + batch_ind * n_max_boxes;
					Tensor target_labels = gt_labels.@long().flatten()[target_gt_idx];

					Tensor target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[2])[target_gt_idx];

					target_labels = target_labels.clamp_(0);

					Tensor target_scores = torch.zeros(
						target_labels.shape[0], target_labels.shape[1], num_classes,
						dtype: torch.int64, device: target_labels.device
					);

					target_scores = target_scores.scatter_(dim: 2, index: target_labels.unsqueeze(-1), src: torch.ones_like(target_labels.unsqueeze(-1), dtype: torch.int64, device: target_labels.device));

					Tensor fg_scores_mask = fg_mask.unsqueeze(-1).expand(bs, -1, num_classes);
					target_scores = torch.where(fg_scores_mask > 0, target_scores, torch.zeros_like(target_scores));

					return (target_labels.MoveToOuterDisposeScope(), target_bboxes.MoveToOuterDisposeScope(), target_scores.MoveToOuterDisposeScope());
				}
			}

			private Tensor select_candidates_in_gts(Tensor xy_centers, Tensor gt_bboxes, float eps = 1e-9f)
			{
				using (NewDisposeScope())
				{
					long n_anchors = xy_centers.shape[0];
					//var (bs, n_boxes, _) = gt_bboxes.shape;
					long bs = gt_bboxes.shape[0];
					long n_boxes = gt_bboxes.shape[1];
					//var (lt, rb) = gt_bboxes.view(-1, 1, 4).chunk(2, dim: 2);
					Tensor[] lt_rb = gt_bboxes.view(-1, 1, 4).chunk(2, dim: 2);
					Tensor lt = lt_rb[0].to(xy_centers.device);
					Tensor rb = lt_rb[1].to(xy_centers.device);

					Tensor bbox_deltas = torch.cat(new Tensor[] { xy_centers[TensorIndex.None] - lt, rb - xy_centers[TensorIndex.None] }, dim: 2).view(bs, n_boxes, n_anchors, -1);
					return bbox_deltas.amin(dims: new long[] { 3 }).gt_(eps).MoveToOuterDisposeScope();

				}
			}

			private (Tensor, Tensor, Tensor) select_highest_overlaps(Tensor mask_pos, Tensor overlaps, long n_max_boxes)
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

					Tensor target_gt_idx = mask_pos.argmax(dim: -2);
					return (target_gt_idx.MoveToOuterDisposeScope(), fg_mask.MoveToOuterDisposeScope(), mask_pos.MoveToOuterDisposeScope());
				}
			}

			private Tensor bbox_iou(Tensor box1, Tensor box2, bool xywh = true, bool GIoU = false, bool DIoU = false, bool CIoU = false, float eps = 1e-7f)
			{
				using (NewDisposeScope())
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
					var inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0);

					// Union Area
					var union = w1 * h1 + w2 * h2 - inter + eps;

					// IoU
					var iou = inter / union;
					if (CIoU || DIoU || GIoU)
					{
						var cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1);  //convex (smallest enclosing box) width
						var ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1);  // convex height
						if (CIoU || DIoU)  // Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
						{
							var c2 = cw.pow(2) + ch.pow(2) + eps;   //convex diagonal squared
							var rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4;   //center dist ** 2

							if (CIoU)  // https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
							{
								var v = 4 / (MathF.PI * MathF.PI) * (atan(w2 / h2) - atan(w1 / h1)).pow(2);
								
								{
									var alpha = v / (v - iou + (1 + eps));
									return (iou - (rho2 / c2 + v * alpha)).MoveToOuterDisposeScope();  //CIoU
								}
							}
							return (iou - rho2 / c2).MoveToOuterDisposeScope();  // DIoU
						}
						var c_area = cw * ch + eps;    // convex area
						return (iou - (c_area - union) / c_area).MoveToOuterDisposeScope();  // GIoU https://arxiv.org/pdf/1902.09630.pdf
					}
					return iou.MoveToOuterDisposeScope(); //IoU
				}
			}
		}

		internal class RotatedTaskAlignedAssigner : TaskAlignedAssigner
		{
			internal RotatedTaskAlignedAssigner(int topk = 13, int num_classes = 80, float alpha = 1.0f, float beta = 6.0f, float eps = 1e-9f) : base(topk: topk, num_classes: num_classes, alpha: alpha, beta: beta, eps: eps)
			{

			}
			internal static Tensor iou_calculation(Tensor gt_bboxes, Tensor pd_bboxes)
			{
				return Utils.Metrics.probiou(gt_bboxes, pd_bboxes);
			}

			/// <summary>
			/// Select the positive anchor center in gt for rotated bounding boxes.
			/// </summary>
			/// <param name="xy_centers">Anchor center coordinates with shape (h*w, 2).</param>
			/// <param name="gt_bboxes">Ground truth bounding boxes with shape (b, n_boxes, 5).</param>
			/// <returns>Boolean mask of positive anchors with shape (b, n_boxes, h*w).</returns>
			internal static Tensor select_candidates_in_gts(Tensor xy_centers, Tensor gt_bboxes)
			{
				using (NewDisposeScope())
				{
					// (b, n_boxes, 5) --> (b, n_boxes, 4, 2)
					Tensor corners = Utils.Ops.xywhr2xyxyxyxy(gt_bboxes);
					// (b, n_boxes, 1, 2)
					Tensor[] ab_d = corners.split(1, dim: -2);
					Tensor a = ab_d[0], b = ab_d[1], d = ab_d[3];
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
					return cat(new Tensor[] { c_xy, wh }, dim);  // xywh bbox
				}
				return torch.cat(new Tensor[] { x1y1, x2y2 }, dim).MoveToOuterDisposeScope(); // xyxy bbox
			}
		}

		// Transform bbox(xyxy) to dist(ltrb).
		internal static Tensor bbox2dist(Tensor anchor_points, Tensor bbox, int reg_max)
		{
			using (NewDisposeScope())
			{
				Tensor[] x1y1x2y2 = bbox.chunk(2, -1);
				Tensor x1y1 = x1y1x2y2[0];
				Tensor x2y2 = x1y1x2y2[1];
				return cat(new Tensor[] { anchor_points - x1y1, x2y2 - anchor_points }, -1).clamp_(0, reg_max - 0.01).MoveToOuterDisposeScope();  // dist (lt, rb)
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
	}
}
