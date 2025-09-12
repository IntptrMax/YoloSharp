using TorchSharp;
using static TorchSharp.torch;

namespace Utils
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
				Tensor x1 = obb1[.., 0];
				Tensor y1 = obb1[.., 1];
				Tensor x2 = obb2[.., 0];
				Tensor y2 = obb2[.., 1];

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

				// Prepare tensors for broadcasting
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
							using (no_grad())
							{
								Tensor alpha = v / (v - iou + (1 + eps));
								return iou - (rho2 / c2 + v * alpha);  //CIoU
							}
						}
						return iou - rho2 / c2;  // DIoU
					}
					Tensor c_area = cw * ch + eps;    // convex area
					return (iou - (c_area - union) / c_area).MoveToOuterDisposeScope();  // GIoU https://arxiv.org/pdf/1902.09630.pdf
				}
				return iou.MoveToOuterDisposeScope(); //IoU
			}
		}
	}
}
