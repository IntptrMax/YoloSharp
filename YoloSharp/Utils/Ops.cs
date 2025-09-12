using TorchSharp;
using static TorchSharp.torch;

namespace Utils
{
	internal class Ops
	{
		/// <summary>
		/// Convert batched Oriented Bounding Boxes (OBB) from [x, rotation] to [xy1, xy2, xy3, xy4] format.
		/// </summary>
		/// <param name="x">Boxes in [cx, cy, w, h, rotation] format with shape (N, 5) or (B, N, 5). Rotation values should be in radians from 0 to pi/2.</param>
		/// <returns>Converted corner points with shape (N, 4, 2) or (B, N, 4, 2).</returns>
		internal static Tensor xywhr2xyxyxyxy(Tensor x)
		{
			using (NewDisposeScope())
			using (no_grad())
			{
				Tensor ctr = x[.., ..2];

				Tensor w = x[.., 2];
				Tensor h = x[.., 3];
				Tensor angle = x[.., 4];
				Tensor cos_value = cos(angle);
				Tensor sin_value = sin(angle);
				Tensor[] v1 = new Tensor[] { w / 2 * cos_value, w / 2 * sin_value };
				Tensor[] v2 = new Tensor[] { -h / 2 * sin_value, h / 2 * cos_value };

				Tensor vec1 = torch.cat(v1, -1);
				Tensor vec2 = torch.cat(v2, -1);

				Tensor pt1 = ctr + vec1 + vec2;
				Tensor pt2 = ctr + vec1 - vec2;
				Tensor pt3 = ctr - vec1 - vec2;
				Tensor pt4 = ctr - vec1 + vec2;

				return stack(new Tensor[] { pt1, pt2, pt3, pt4 }, -2).MoveToOuterDisposeScope();
			}
		}

		/// <summary>
		/// Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the	top-left corner and(x2, y2) is the bottom-right corner.Note: ops per 2 channels faster than per channel.
		/// </summary>
		/// <param name="x">Input bounding box coordinates in (x, y, width, height) format.</param>
		/// <returns>Bounding box coordinates in (x1, y1, x2, y2) format.</returns>
		internal static Tensor xywh2xyxy(Tensor x)
		{
			using (no_grad())
			{
				if (x.shape.Last() != 4)
				{
					throw new ArgumentException($"input shape last dimension expected 4 but input shape is {x.shape}");
				}

				Tensor y = zeros_like(x);
				y[TensorIndex.Ellipsis, 0] = x[TensorIndex.Ellipsis, 0] - x[TensorIndex.Ellipsis, 2] / 2; // x1
				y[TensorIndex.Ellipsis, 1] = x[TensorIndex.Ellipsis, 1] - x[TensorIndex.Ellipsis, 3] / 2; // y1
				y[TensorIndex.Ellipsis, 2] = x[TensorIndex.Ellipsis, 0] + x[TensorIndex.Ellipsis, 2] / 2; // x2
				y[TensorIndex.Ellipsis, 3] = x[TensorIndex.Ellipsis, 1] + x[TensorIndex.Ellipsis, 3] / 2; // y2
				return y;
			}
		}

		/// <summary>
		/// Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
		/// </summary>
		/// <param name="x">The input bounding box coordinates in (x1, y1, x2, y2) format.</param>
		/// <returns>The bounding box coordinates in (x, y, width, height) format.</returns>
		internal static Tensor xyxy2xywh(Tensor x)
		{
			using (no_grad())
			{
				if (x.shape.Last() != 4)
				{
					throw new ArgumentException($"input shape last dimension expected 4 but input shape is {x.shape}");
				}
				Tensor y = empty_like(x);  // faster than clone/copy
				y[TensorIndex.Ellipsis, 0] = (x[TensorIndex.Ellipsis, 0] + x[TensorIndex.Ellipsis, 2]) / 2;  // x center
				y[TensorIndex.Ellipsis, 1] = (x[TensorIndex.Ellipsis, 1] + x[TensorIndex.Ellipsis, 3]) / 2; // y center
				y[TensorIndex.Ellipsis, 2] = x[TensorIndex.Ellipsis, 2] - x[TensorIndex.Ellipsis, 0]; // width
				y[TensorIndex.Ellipsis, 3] = x[TensorIndex.Ellipsis, 3] - x[TensorIndex.Ellipsis, 1];  // height
				return y;
			}
		}

		/// <summary>
		/// Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="w"></param>
		/// <param name="h"></param>
		/// <param name="clip"></param>
		/// <param name="eps"></param>
		/// <returns></returns>
		internal static Tensor xyxy2xywhn(Tensor x, int w = 640, int h = 640, bool clip = false, float eps = 0.0f)
		{
			using (no_grad())
			{
				if (clip)
				{
					x = clip_boxes(x, new float[] { h - eps, w - eps });
				}
				Tensor y = x.clone();
				y[TensorIndex.Ellipsis, 0] = (x[TensorIndex.Ellipsis, 0] + x[TensorIndex.Ellipsis, 2]) / 2 / w;  // x center
				y[TensorIndex.Ellipsis, 1] = (x[TensorIndex.Ellipsis, 1] + x[TensorIndex.Ellipsis, 3]) / 2 / h;// y center
				y[TensorIndex.Ellipsis, 2] = (x[TensorIndex.Ellipsis, 2] - x[TensorIndex.Ellipsis, 0]) / w;  // width
				y[TensorIndex.Ellipsis, 3] = (x[TensorIndex.Ellipsis, 3] - x[TensorIndex.Ellipsis, 1]) / h;  // height
				return y;
			}
		}

		/// <summary>
		/// Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right.
		/// </summary>
		/// <param name="x"></param>
		/// <param name="w"></param>
		/// <param name="h"></param>
		/// <param name="padw"></param>
		/// <param name="padh"></param>
		/// <returns></returns>
		internal static Tensor xywhn2xyxy(Tensor x, int w = 640, int h = 640, int padw = 0, int padh = 0)
		{
			using (no_grad())
			{
				Tensor y = x.clone();
				y[TensorIndex.Ellipsis, 0] = w * (x[TensorIndex.Ellipsis, 0] - x[TensorIndex.Ellipsis, 2] / 2) + padw;  // top left x
				y[TensorIndex.Ellipsis, 1] = h * (x[TensorIndex.Ellipsis, 1] - x[TensorIndex.Ellipsis, 3] / 2) + padh;  // top left y
				y[TensorIndex.Ellipsis, 2] = w * (x[TensorIndex.Ellipsis, 0] + x[TensorIndex.Ellipsis, 2] / 2) + padw;  // bottom right x
				y[TensorIndex.Ellipsis, 3] = h * (x[TensorIndex.Ellipsis, 1] + x[TensorIndex.Ellipsis, 3] / 2) + padh;  // bottom right y
				return y;
			}
		}

		/// <summary>
		/// Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.
		/// </summary>
		/// <param name="x">The bounding boxes to clip</param>
		/// <param name="shape">The shape of the image</param>
		/// <returns>The clipped boxes</returns>
		internal static Tensor clip_boxes(Tensor x, float[] shape)
		{
			using (no_grad())
			{
				Tensor box = torch.zeros_like(x);
				box[TensorIndex.Ellipsis, 0] = x[TensorIndex.Ellipsis, 0].clamp_(0, shape[1]);  // x1
				box[TensorIndex.Ellipsis, 1] = x[TensorIndex.Ellipsis, 1].clamp_(0, shape[0]);  // y1
				box[TensorIndex.Ellipsis, 2] = x[TensorIndex.Ellipsis, 2].clamp_(0, shape[1]);  // x2
				box[TensorIndex.Ellipsis, 3] = x[TensorIndex.Ellipsis, 3].clamp_(0, shape[0]);  // y2
				return box;
			}
		}

		/// <summary>
		/// Perform non-maximum suppression (NMS) on prediction results.<br/>
		/// Applies NMS to filter overlapping bounding boxes based on confidence and IoU thresholds. Supports multiple detection formats including standard boxes, rotated boxes, and masks.
		/// </summary>
		/// <param name="prediction">Predictions with shape (batch_size, num_classes + 4 + num_masks, num_boxes) containing boxes, classes, and optional masks.</param>
		/// <param name="conf_thres">Confidence threshold for filtering detections. Valid values are between 0.0 and 1.0.</param>
		/// <param name="iou_thres">IoU threshold for NMS filtering. Valid values are between 0.0 and 1.0.</param>
		/// <param name="agnostic">Whether to perform class-agnostic NMS.</param>
		/// <param name="max_det">Maximum number of detections to keep per image.</param>
		/// <param name="nc">Number of classes. Indices after this are considered masks.</param>
		/// <param name="max_time_img">Maximum time in seconds for processing one image.</param>
		/// <param name="max_nms">Maximum number of boxes for torchvision.ops.nms().</param>
		/// <param name="max_wh">Maximum box width and height in pixels.</param>
		/// <param name="in_place">Whether to modify the input prediction tensor in place.</param>
		/// <param name="rotated">Whether to handle Oriented Bounding Boxes (OBB).</param>
		/// <param name="end2end">Whether the model is end-to-end and doesn't require NMS.</param>
		/// <returns>List of detections per image with shape (num_boxes, 6 + num_masks) containing (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).</returns>
		/// <exception cref="ArgumentException"></exception>
		internal static (List<Tensor> output, List<Tensor> keepi) non_max_suppression(
			Tensor prediction, float conf_thres = 0.25f, float iou_thres = 0.45f,
			bool agnostic = false, int max_det = 300, long nc = 0, float max_time_img = 0.05f,
			int max_nms = 30000, int max_wh = 7680, bool in_place = true,
			bool rotated = false, bool end2end = false)
		{
			using (NewDisposeScope())
			using (no_grad())
			{
				// Checks
				if (conf_thres < 0 || conf_thres > 1)
				{
					throw new ArgumentException($"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0");
				}
				if (iou_thres < 0 || iou_thres > 1)
				{
					throw new ArgumentException($"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0");
				}
				int bs = (int)prediction.shape[0]; // batch size (BCN, i.e. 1,84,6300)

				if (prediction.shape.Last() == 6 || end2end) // end-to-end model (BNC, i.e. 1,300,6)
				{
					return (prediction.split(1)
				   .Select(pred =>
				   {
					   var mask = pred[.., 4] > conf_thres;
					   return pred[mask][max_det].MoveToOuterDisposeScope();
				   })
				   .ToList(), new List<Tensor> { torch.zeros(0).MoveToOuterDisposeScope() });
				}

				nc = nc == 0 ? prediction.shape[1] - 4 : nc; // number of classes
				long extra = prediction.shape[1] - nc - 4;  // number of extra info
				long mi = 4 + nc; // mask start index
				Tensor xc = prediction[.., 4..(int)mi].amax(1) > conf_thres; // candidates

				List<Tensor> xindsList = new List<Tensor>();
				for (int idx = 0; idx < xc.shape[0]; idx++)
				{
					Tensor mask = xc[idx];
					xindsList.Add(torch.arange(mask.NumberOfElements, device: prediction.device));
				}
				Tensor xinds = torch.stack(xindsList, 0).unsqueeze(-1); // [batch, N, 1]

				// Settings
				// min_wh = 2  # (pixels) minimum box width and height
				float time_limit = 2.0f + max_time_img * bs; // seconds to quit after

				prediction = prediction.transpose(-1, -2); //shape(1,84,6300) to shape(1,6300,84)

				if (!rotated)
				{
					if (in_place)
					{
						prediction[TensorIndex.Ellipsis, ..4] = xywh2xyxy(prediction[TensorIndex.Ellipsis, ..4]);  // xywh to xyxy
					}
					else
					{
						prediction = torch.cat(new Tensor[] { xywh2xyxy(prediction[TensorIndex.Ellipsis, ..4]), prediction[TensorIndex.Ellipsis, 4..] }, dim: -1);  // xywh to xyxy
					}
				}

				//prediction[TensorIndex.Ellipsis, ..4] = torchvision.ops.box_convert(prediction[TensorIndex.Ellipsis, ..4], torchvision.ops.BoxFormats.cxcywh, torchvision.ops.BoxFormats.xyxy);
				DateTime t = DateTime.Now;

				List<Tensor> output = Enumerable.Range(0, bs).Select(_ => torch.zeros(new long[] { 0, 6 + extra }, device: prediction.device).clone()).ToList();
				List<Tensor> keepi = Enumerable.Range(0, bs).Select(_ => torch.zeros(new long[] { 0, 1 }, device: prediction.device).clone()).ToList(); // to store the kept idxs

				for (int xi = 0; xi < bs; xi++)
				{
					// Apply constraints
					// x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
					Tensor x = prediction[xi];
					Tensor xk = xinds[xi];
					Tensor filt = xc[xi];
					x = x[filt]; // confidence
					xk = xk[filt];

					long n = x.shape[0];
					// If none remain process next image
					if (n == 0)
					{
						continue;
					}

					Tensor[] box_cls_mask = x.split(new long[] { 4, nc, extra }, 1);
					Tensor box = box_cls_mask[0];
					Tensor cls = box_cls_mask[1];
					Tensor mask = box_cls_mask[2];

					(Tensor conf, Tensor j) = cls.max(1, keepdim: true);
					filt = conf.view(-1) > conf_thres;

					x = torch.cat(new Tensor[] { box, conf, j.@float(), mask }, 1)[filt];
					xk = xk[filt];

					// Check shape
					n = x.shape[0];  // number of boxes
					if (n == 0)
					{
						continue; // no boxes
					}

					if (n > max_nms)//  # excess boxes
					{
						filt = x[.., 4].argsort(descending: true)[..max_nms];  // sort by confidence and remove excess boxes
						(x, xk) = (x[filt], xk[filt]);
					}

					// Batched NMS
					Tensor c = x[.., 5..6] * max_wh;  // classes
					Tensor scores = x[.., 4];  // scores

					Tensor i = torch.zeros(0);
					if (rotated)
					{
						Tensor boxes = torch.cat(new Tensor[] { x[.., ..2] + c, x[.., 2..4], x[.., (int)(x.shape[1] - 1)..] }, dim: -1); // xywhr
						i = nms_rotated(boxes, scores, threshold: iou_thres); // NMS
					}
					else
					{
						Tensor boxes = x[.., ..4] + c;  // boxes (offset by class)
						i = torchvision.ops.nms(boxes, scores, iou_thres);  // NMS
					}

					i = i[..max_det]; // limit detections
					(output[xi], keepi[xi]) = (x[i], xk[i].reshape(-1));
					if ((DateTime.Now - t).TotalSeconds > time_limit)
					{
						// time limit exceeded
						Console.WriteLine($"NMS time limit {time_limit}s exceeded");
					}
				}
				return (output.Select(x => x.MoveToOuterDisposeScope()).ToList(), keepi.Select(x => x.MoveToOuterDisposeScope()).ToList());
			}
		}

		internal static Tensor nms_rotated(Tensor boxes, Tensor scores, float threshold = 0.45f, bool use_triu = true)
		{
			using (NewDisposeScope())
			using (no_grad())
			{
				Tensor sorted_idx = torch.argsort(scores, descending: true);
				boxes = boxes[sorted_idx];
				Tensor ious = Utils.Metrics.batch_probiou(boxes, boxes);
				Tensor pick = torch.zeros(0);
				if (use_triu)
				{
					ious = ious.triu_(diagonal: 1);
					// NOTE: handle the case when len(boxes) hence exportable by eliminating if-else condition
					pick = torch.nonzero((ious >= threshold).sum(0, type: torch.ScalarType.Bool) <= 0).squeeze_(-1);
				}
				else
				{
					long n = boxes.shape[0];
					Tensor row_idx = torch.arange(n, device: boxes.device).view(-1, 1).expand(-1, n);
					Tensor col_idx = torch.arange(n, device: boxes.device).view(1, -1).expand(n, -1);
					Tensor upper_mask = row_idx < col_idx;
					ious = ious * upper_mask;
					// Zeroing these scores ensures the additional indices would not affect the final results
					scores[~((ious >= threshold).sum(0) <= 0)] = 0;
					// NOTE: return indices with fixed length to avoid TFLite reshape error
					pick = torch.topk(scores, (int)scores.shape[0]).indices;
				}
				return sorted_idx[pick].MoveToOuterDisposeScope();
			}
		}

		/// <summary>
		/// It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box.
		/// </summary>
		/// <param name="masks">[n, h, w] tensor of masks</param>
		/// <param name="boxes">[n, 4] tensor of bbox coordinates in relative point form</param>
		/// <returns>The masks are being cropped to the bounding box.</returns>
		internal static Tensor crop_mask(Tensor masks, Tensor boxes)
		{
			using (NewDisposeScope())
			using (no_grad())
			{
				long h = masks.shape[1];
				long w = masks.shape[2];
				Tensor[] x1y1x2y2 = chunk(boxes[.., .., TensorIndex.None], 4, 1);  // x1 shape(n,1,1)
				Tensor x1 = x1y1x2y2[0];
				Tensor y1 = x1y1x2y2[1];
				Tensor x2 = x1y1x2y2[2];
				Tensor y2 = x1y1x2y2[3];
				Tensor r = arange(w, device: masks.device, dtype: x1.dtype)[TensorIndex.None, TensorIndex.None, ..]; //rows shape(1,1,w)
				Tensor c = arange(h, device: masks.device, dtype: x1.dtype)[TensorIndex.None, .., TensorIndex.None]; //cols shape(1,h,1)
				return (masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))).MoveToOuterDisposeScope();
			}
		}

		/// <summary>
		/// Apply masks to bounding boxes using mask head output.
		/// </summary>
		/// <param name="protos">Mask prototypes with shape (mask_dim, mask_h, mask_w).</param>
		/// <param name="masks_in">Mask coefficients with shape (N, mask_dim) where N is number of masks after NMS.</param>
		/// <param name="bboxes">Bounding boxes with shape (N, 4) where N is number of masks after NMS.</param>
		/// <param name="shape">Input image size as (height, width).</param>
		/// <param name="upsample">Whether to upsample masks to original image size.</param>
		/// <returns>A binary mask tensor of shape [n, h, w], where n is the number of masks after NMS, and h and w	are the height and width of the input image.The mask is applied to the bounding boxes.</returns>
		internal static Tensor process_mask(Tensor protos, Tensor masks_in, Tensor bboxes, long[] shape, bool upsample = false)
		{
			long c = protos.shape[0]; //  # CHW
			long mh = protos.shape[1];
			long mw = protos.shape[2];

			long ih = shape[0];
			long iw = shape[1];
			Tensor masks = masks_in.matmul(protos.@float().view(c, -1)).view(-1, mh, mw);  //  # CHW
			float width_ratio = (float)mw / iw;
			float height_ratio = (float)mh / ih;

			Tensor downsampled_bboxes = bboxes.clone();
			downsampled_bboxes[.., 0] *= width_ratio;
			downsampled_bboxes[.., 2] *= width_ratio;
			downsampled_bboxes[.., 3] *= height_ratio;
			downsampled_bboxes[.., 1] *= height_ratio;
			masks = crop_mask(masks, downsampled_bboxes); //  # CHW

			if (upsample)
			{
				masks = torch.nn.functional.interpolate(masks[TensorIndex.None], size: shape, mode: InterpolationMode.Bilinear, align_corners: false)[0];// # CHW
			}
			return masks.gt_(0.0);

		}
	}
}

