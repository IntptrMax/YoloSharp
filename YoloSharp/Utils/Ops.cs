using OpenCvSharp;
using TorchSharp;

namespace YoloSharp.Utils
{
    internal class Ops
    {
        /// <summary>
        /// Convert batched Oriented Bounding Boxes (OBB) from [x, rotation] to [xy1, xy2, xy3, xy4] format.
        /// </summary>
        /// <param name="x">Boxes in [cx, cy, w, h, rotation] format with shape (N, 5) or (B, N, 5). Rotation values should be in radians from 0 to pi/2.</param>
        /// <returns>Converted corner points with shape (N, 4, 2) or (B, N, 4, 2).</returns>
        internal static torch.Tensor xywhr2xyxyxyxy(torch.Tensor x)
        {
            using (torch.NewDisposeScope())
            {
                torch.Tensor ctr = x[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(0, 2)];

                torch.Tensor w = x[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(2, 3)];
                torch.Tensor h = x[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(3, 4)];
                torch.Tensor angle = x[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(4, 5)];
                torch.Tensor cos_value = torch.cos(angle);
                torch.Tensor sin_value = torch.sin(angle);
                torch.Tensor[] v1 = new torch.Tensor[] { w / 2 * cos_value, w / 2 * sin_value };
                torch.Tensor[] v2 = new torch.Tensor[] { -h / 2 * sin_value, h / 2 * cos_value };

                torch.Tensor vec1 = torch.cat(v1, -1);
                torch.Tensor vec2 = torch.cat(v2, -1);

                torch.Tensor pt1 = ctr + vec1 + vec2;
                torch.Tensor pt2 = ctr + vec1 - vec2;
                torch.Tensor pt3 = ctr - vec1 - vec2;
                torch.Tensor pt4 = ctr - vec1 + vec2;

                return torch.stack(new torch.Tensor[] { pt1, pt2, pt3, pt4 }, -2).MoveToOuterDisposeScope();
            }
        }

        /// <summary>
        /// Convert batched Oriented Bounding Boxes (OBB) from [xy1, xy2, xy3, xy4] to [xywh, rotation] format.
        /// </summary>
        /// <param name="x">Input box corners with shape (N, 8) in [xy1, xy2, xy3, xy4] format.</param>
        /// <returns>Converted data in [cx, cy, w, h, rotation] format with shape (N, 5). Rotation values are in radians from 0 to pi/2. </returns>
        internal static float[] xyxyxyxy2xywhr(float[] x)
        {
            RotatedRect rotatedRect = Cv2.MinAreaRect(new Point2f[]
            {
                new Point2f(x[0],x[1]),
                new Point2f(x[2],x[3]),
                new Point2f(x[4],x[5]),
                new Point2f(x[6],x[7]),
            });
            return new float[] { rotatedRect.Center.X, rotatedRect.Center.Y, rotatedRect.Size.Width, rotatedRect.Size.Height, rotatedRect.Angle * (float)Math.PI / 180.0f };
        }

        internal static torch.Tensor xyxyxyxy2xywhr(torch.Tensor x)
        {
            float[] xx = x.data<float>().ToArray();
            float[] re = xyxyxyxy2xywhr(xx);
            return torch.tensor(re, dtype: x.dtype, device: x.device);
        }

        /// <summary>
        /// Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the	top-left corner and(x2, y2) is the bottom-right corner.Note: ops per 2 channels faster than per channel.
        /// </summary>
        /// <param name="x">Input bounding box coordinates in (x, y, width, height) format.</param>
        /// <returns>Bounding box coordinates in (x1, y1, x2, y2) format.</returns>
        internal static torch.Tensor xywh2xyxy(torch.Tensor x)
        {
            if (x.shape.Last() != 4)
            {
                throw new ArgumentException($"input shape last dimension expected 4 but input shape is {x.shape}");
            }

            torch.Tensor y = torch.zeros_like(x);
            y[torch.TensorIndex.Ellipsis, 0] = x[torch.TensorIndex.Ellipsis, 0] - x[torch.TensorIndex.Ellipsis, 2] / 2; // x1
            y[torch.TensorIndex.Ellipsis, 1] = x[torch.TensorIndex.Ellipsis, 1] - x[torch.TensorIndex.Ellipsis, 3] / 2; // y1
            y[torch.TensorIndex.Ellipsis, 2] = x[torch.TensorIndex.Ellipsis, 0] + x[torch.TensorIndex.Ellipsis, 2] / 2; // x2
            y[torch.TensorIndex.Ellipsis, 3] = x[torch.TensorIndex.Ellipsis, 1] + x[torch.TensorIndex.Ellipsis, 3] / 2; // y2
            return y;
        }

        /// <summary>
        /// Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
        /// </summary>
        /// <param name="x">The input bounding box coordinates in (x1, y1, x2, y2) format.</param>
        /// <returns>The bounding box coordinates in (x, y, width, height) format.</returns>
        internal static torch.Tensor xyxy2xywh(torch.Tensor x)
        {
            if (x.shape.Last() != 4)
            {
                throw new ArgumentException($"input shape last dimension expected 4 but input shape is {x.shape}");
            }
            torch.Tensor y = torch.empty_like(x);  // faster than clone/copy
            y[torch.TensorIndex.Ellipsis, 0] = (x[torch.TensorIndex.Ellipsis, 0] + x[torch.TensorIndex.Ellipsis, 2]) / 2;  // x center
            y[torch.TensorIndex.Ellipsis, 1] = (x[torch.TensorIndex.Ellipsis, 1] + x[torch.TensorIndex.Ellipsis, 3]) / 2; // y center
            y[torch.TensorIndex.Ellipsis, 2] = x[torch.TensorIndex.Ellipsis, 2] - x[torch.TensorIndex.Ellipsis, 0]; // width
            y[torch.TensorIndex.Ellipsis, 3] = x[torch.TensorIndex.Ellipsis, 3] - x[torch.TensorIndex.Ellipsis, 1];  // height
            return y;
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
        internal static torch.Tensor xyxy2xywhn(torch.Tensor x, int w = 640, int h = 640, bool clip = false, float eps = 0.0f)
        {
            if (clip)
            {
                x = clip_boxes(x, new float[] { h - eps, w - eps });
            }
            torch.Tensor y = x.clone();
            y[torch.TensorIndex.Ellipsis, 0] = (x[torch.TensorIndex.Ellipsis, 0] + x[torch.TensorIndex.Ellipsis, 2]) / 2 / w;  // x center
            y[torch.TensorIndex.Ellipsis, 1] = (x[torch.TensorIndex.Ellipsis, 1] + x[torch.TensorIndex.Ellipsis, 3]) / 2 / h;// y center
            y[torch.TensorIndex.Ellipsis, 2] = (x[torch.TensorIndex.Ellipsis, 2] - x[torch.TensorIndex.Ellipsis, 0]) / w;  // width
            y[torch.TensorIndex.Ellipsis, 3] = (x[torch.TensorIndex.Ellipsis, 3] - x[torch.TensorIndex.Ellipsis, 1]) / h;  // height
            return y;
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
        internal static torch.Tensor xywhn2xyxy(torch.Tensor x, int w = 640, int h = 640, int padw = 0, int padh = 0)
        {
            torch.Tensor y = x.clone();
            y[torch.TensorIndex.Ellipsis, 0] = w * (x[torch.TensorIndex.Ellipsis, 0] - x[torch.TensorIndex.Ellipsis, 2] / 2) + padw;  // top left x
            y[torch.TensorIndex.Ellipsis, 1] = h * (x[torch.TensorIndex.Ellipsis, 1] - x[torch.TensorIndex.Ellipsis, 3] / 2) + padh;  // top left y
            y[torch.TensorIndex.Ellipsis, 2] = w * (x[torch.TensorIndex.Ellipsis, 0] + x[torch.TensorIndex.Ellipsis, 2] / 2) + padw;  // bottom right x
            y[torch.TensorIndex.Ellipsis, 3] = h * (x[torch.TensorIndex.Ellipsis, 1] + x[torch.TensorIndex.Ellipsis, 3] / 2) + padh;  // bottom right y
            return y;
        }

        /// <summary>
        /// Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.
        /// </summary>
        /// <param name="x">The bounding boxes to clip</param>
        /// <param name="shape">The shape of the image</param>
        /// <returns>The clipped boxes</returns>
        internal static torch.Tensor clip_boxes(torch.Tensor x, float[] shape)
        {
            torch.Tensor box = torch.zeros_like(x);
            box[torch.TensorIndex.Ellipsis, 0] = x[torch.TensorIndex.Ellipsis, 0].clamp_(0, shape[1]);  // x1
            box[torch.TensorIndex.Ellipsis, 1] = x[torch.TensorIndex.Ellipsis, 1].clamp_(0, shape[0]);  // y1
            box[torch.TensorIndex.Ellipsis, 2] = x[torch.TensorIndex.Ellipsis, 2].clamp_(0, shape[1]);  // x2
            box[torch.TensorIndex.Ellipsis, 3] = x[torch.TensorIndex.Ellipsis, 3].clamp_(0, shape[0]);  // y2
            return box;
        }

        /// <summary>
        /// Takes a list of keypoints and a shape (height, width) and clips the keypoints to the shape.
        /// </summary>
        /// <param name="kpts">The keypoints of the image.</param>
        /// <param name="shape">The shape of the image.</param>
        /// <returns>The clipped keypoints.</returns>
        internal static torch.Tensor clip_keypoints(torch.Tensor kpts, float[] shape)
        {
            torch.Tensor keypoints = kpts.clone();
            int w = (int)shape[1];
            int h = (int)shape[0];
            if (keypoints.shape[keypoints.shape.Length - 1] == 3)
            {
                keypoints[torch.TensorIndex.Ellipsis, 2][
                               (keypoints[torch.TensorIndex.Ellipsis, 0] < 0)
                               | (keypoints[torch.TensorIndex.Ellipsis, 0] > w)
                               | (keypoints[torch.TensorIndex.Ellipsis, 1] < 0)
                               | (keypoints[torch.TensorIndex.Ellipsis, 1] > h)
                           ] = 0.0;
            }
            keypoints[torch.TensorIndex.Ellipsis, 0] = keypoints[torch.TensorIndex.Ellipsis, 0].clip(0, w);
            keypoints[torch.TensorIndex.Ellipsis, 1] = keypoints[torch.TensorIndex.Ellipsis, 1].clip(0, h);
            return keypoints;
        }

        /// <summary>
        /// Takes a list of obb corners and a shape (height, width) and clips the corners to the shape.
        /// </summary>
        /// <param name="obb_corners">The oriented bounding box corners to clip.</param>
        /// <param name="shape">The shape of the image.</param>
        /// <returns>The clipped corners.</returns>
        internal static torch.Tensor clip_obb_corners(torch.Tensor obb_corners, float[] shape)
        {
            torch.Tensor corners = obb_corners.clone();
            int w = (int)shape[1];
            int h = (int)shape[0];
            corners[torch.TensorIndex.Ellipsis, 0] = corners[torch.TensorIndex.Ellipsis, 0].clip(0, w);
            corners[torch.TensorIndex.Ellipsis, 1] = corners[torch.TensorIndex.Ellipsis, 1].clip(0, h);
            return corners;
        }

        /// <summary>
        /// sort the OBB corners (top-left corner as the 0th point, counter-clockwise by angle)
        /// </summary>
        internal static torch.Tensor sort_obb_corners_batch(torch.Tensor obb_corners)
        {
            torch.Tensor centers = obb_corners.mean(new long[] { 1 });  // (n, 2)

            torch.Tensor dx = obb_corners[torch.TensorIndex.Ellipsis, 0] - centers[torch.TensorIndex.Ellipsis, 0].unsqueeze(1);
            torch.Tensor dy = obb_corners[torch.TensorIndex.Ellipsis, 1] - centers[torch.TensorIndex.Ellipsis, 1].unsqueeze(1);
            torch.Tensor angles = torch.atan2(dy, dx);
            torch.Tensor sorted_idx = angles.argsort(dim: 1).to(torch.int64);  // (n, 4)

            torch.Tensor sorted_x = obb_corners[torch.TensorIndex.Ellipsis, 0].gather(1, sorted_idx);
            torch.Tensor sorted_y = obb_corners[torch.TensorIndex.Ellipsis, 1].gather(1, sorted_idx);
            torch.Tensor sorted = torch.stack(new[] { sorted_x, sorted_y }, dim: 2);

            return sorted;
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
        internal static (List<torch.Tensor> output, List<torch.Tensor> keepi) non_max_suppression(
            torch.Tensor prediction, float conf_thres = 0.25f, float iou_thres = 0.45f,
            bool agnostic = false, int max_det = 300, long nc = 0, float max_time_img = 0.05f,
            int max_nms = 30000, int max_wh = 7680, bool in_place = true,
            bool rotated = false, bool end2end = false)
        {
            using (torch.NewDisposeScope())
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
                    List<torch.Tensor> otpt = new List<torch.Tensor>();
                    for (int i = 0; i < prediction.shape[0]; i++)
                    {
                        torch.Tensor pred = prediction[i];
                        otpt.Add((pred[pred[torch.TensorIndex.Colon, 4] > conf_thres][torch.TensorIndex.Slice(0, max_det)]).MoveToOuterDisposeScope());
                    }
                    return (otpt, new List<torch.Tensor> { torch.zeros(0).MoveToOuterDisposeScope() });
                }

                nc = nc == 0 ? prediction.shape[1] - 4 : nc; // number of classes
                long extra = prediction.shape[1] - nc - 4;  // number of extra info
                long mi = 4 + nc; // mask start index
                torch.Tensor xc = prediction[torch.TensorIndex.Colon, torch.TensorIndex.Slice(4, (int)mi)].amax(1) > conf_thres; // candidates

                List<torch.Tensor> xindsList = new List<torch.Tensor>();
                for (int idx = 0; idx < xc.shape[0]; idx++)
                {
                    xindsList.Add(torch.arange(xc[idx].NumberOfElements, device: prediction.device));
                }
                torch.Tensor xinds = torch.stack(xindsList, 0).unsqueeze(-1); // [batch, N, 1]
                // Settings
                // min_wh = 2  # (pixels) minimum box width and height
                float time_limit = 2.0f + max_time_img * bs; // seconds to quit after

                prediction = prediction.transpose(-1, -2); //shape(1,84,6300) to shape(1,6300,84)

                if (!rotated)
                {
                    if (in_place)
                    {
                        prediction[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(0, 4)] = xywh2xyxy(prediction[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(0, 4)]);  // xywh to xyxy
                    }
                    else
                    {
                        prediction = torch.cat(new torch.Tensor[] { xywh2xyxy(prediction[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(0, 4)]), prediction[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(4)] }, dim: -1);  // xywh to xyxy
                    }
                }

                List<torch.Tensor> output = Enumerable.Range(0, bs).Select(_ => torch.zeros(new long[] { 0, 6 + extra }, device: prediction.device).clone().MoveToOuterDisposeScope()).ToList();
                List<torch.Tensor> keepi = Enumerable.Range(0, bs).Select(_ => torch.zeros(new long[] { 0, 1 }, device: prediction.device).clone().MoveToOuterDisposeScope()).ToList(); // to store the kept idxs

                for (int xi = 0; xi < bs; xi++)
                {

                    DateTime t = DateTime.Now;
                    // Apply constraints
                    // x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
                    torch.Tensor x = prediction[xi];
                    torch.Tensor xk = xinds[xi];
                    torch.Tensor filt = xc[xi];
                    x = x[filt]; // confidence
                    xk = xk[filt];

                    long n = x.shape[0];
                    // If none remain process next image
                    if (n == 0)
                    {
                        continue;
                    }

                    torch.Tensor[] box_cls_mask = x.split(new long[] { 4, nc, extra }, 1);
                    torch.Tensor box = box_cls_mask[0];
                    torch.Tensor cls = box_cls_mask[1];
                    torch.Tensor mask = box_cls_mask[2];

                    (torch.Tensor conf, torch.Tensor j) = cls.max(1, keepdim: true);
                    filt = conf.view(-1) > conf_thres;

                    x = torch.cat(new torch.Tensor[] { box, conf, j.@float(), mask }, 1)[filt];
                    xk = xk[filt];

                    // Check shape
                    n = x.shape[0];  // number of boxes
                    if (n == 0)
                    {
                        continue; // no boxes
                    }

                    if (n > max_nms)//  # excess boxes
                    {
                        filt = x[torch.TensorIndex.Ellipsis, 4].argsort(descending: true)[torch.TensorIndex.Slice(0, max_nms)];  // sort by confidence and remove excess boxes
                        (x, xk) = (x[filt], xk[filt]);
                    }

                    // Batched NMS
                    torch.Tensor c = x[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(5, 6)] * max_wh;  // classes
                    torch.Tensor scores = x[torch.TensorIndex.Ellipsis, 4];  // scores

                    torch.Tensor i = torch.zeros(0);
                    if (rotated)
                    {
                        torch.Tensor boxes = torch.cat(new torch.Tensor[] { x[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(0, 2)] + c, x[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(2, 4)], x[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice((int)(x.shape[1] - 1))] }, dim: -1); // xywhr
                        i = nms_rotated(boxes, scores, threshold: iou_thres); // NMS
                    }
                    else
                    {
                        torch.Tensor boxes = x[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(0, 4)] + c;  // boxes (offset by class)
                        i = torchvision.ops.nms(boxes, scores, iou_thres);  // NMS
                    }

                    i = i[torch.TensorIndex.Slice(0, max_det)]; // limit detections
                    (output[xi], keepi[xi]) = (x[i].MoveToOuterDisposeScope(), xk[i].reshape(-1).MoveToOuterDisposeScope());
                    if ((DateTime.Now - t).TotalSeconds > time_limit)
                    {
                        // time limit exceeded
                        Console.WriteLine($"NMS time limit {time_limit}s exceeded");
                    }

                }
                return (output, keepi);
            }
        }

        internal static torch.Tensor nms_rotated(torch.Tensor boxes, torch.Tensor scores, float threshold = 0.45f, bool use_triu = true)
        {
            using (torch.NewDisposeScope())
            {
                torch.Tensor sorted_idx = torch.argsort(scores, descending: true);
                boxes = boxes[sorted_idx];
                torch.Tensor ious = Utils.Metrics.batch_probiou(boxes, boxes);
                torch.Tensor pick = torch.zeros(0);
                if (use_triu)
                {
                    ious = ious.triu_(diagonal: 1);
                    // NOTE: handle the case when len(boxes) hence exportable by eliminating if-else condition
                    pick = torch.nonzero((ious >= threshold).sum(0, type: torch.ScalarType.Bool) <= 0).squeeze_(-1);
                }
                else
                {
                    long n = boxes.shape[0];
                    torch.Tensor row_idx = torch.arange(n, device: boxes.device).view(-1, 1).expand(-1, n);
                    torch.Tensor col_idx = torch.arange(n, device: boxes.device).view(1, -1).expand(n, -1);
                    torch.Tensor upper_mask = row_idx < col_idx;
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
        internal static torch.Tensor crop_mask(torch.Tensor masks, torch.Tensor boxes)
        {
            using (torch.NewDisposeScope())
            {
                if (boxes.device != masks.device)
                {
                    boxes = boxes.to(masks.device);
                }
                long n = masks.shape[0];
                long h = masks.shape[1];
                long w = masks.shape[2];

                if (n < 50 && !masks.is_cuda)
                {
                    for (int i = 0; i < boxes.shape[0]; i++)
                    {
                        int x1 = boxes[i, 0].ToInt32();
                        int y1 = boxes[i, 1].ToInt32();
                        int x2 = boxes[i, 2].ToInt32();
                        int y2 = boxes[i, 3].ToInt32();

                        masks[i, torch.TensorIndex.Slice(0, y1)] = 0;
                        masks[i, torch.TensorIndex.Slice(y2)] = 0;
                        masks[i, torch.TensorIndex.Colon, torch.TensorIndex.Slice(0, x1)] = 0;
                        masks[i, torch.TensorIndex.Colon, torch.TensorIndex.Slice(x2)] = 0;
                    }
                    return masks.MoveToOuterDisposeScope();
                }
                else
                {
                    torch.Tensor[] x1_y1_x2_y2 = torch.chunk(boxes[torch.TensorIndex.Colon, torch.TensorIndex.Colon, torch.TensorIndex.None], 4, 1);  // x1 shape(n,1,1)
                    torch.Tensor x1 = x1_y1_x2_y2[0];
                    torch.Tensor y1 = x1_y1_x2_y2[1];
                    torch.Tensor x2 = x1_y1_x2_y2[2];
                    torch.Tensor y2 = x1_y1_x2_y2[3];

                    torch.Tensor r = torch.arange(w, device: masks.device, dtype: x1.dtype)[torch.TensorIndex.None, torch.TensorIndex.None, torch.TensorIndex.Colon];  // rows shape(1,1,w)
                    torch.Tensor c = torch.arange(h, device: masks.device, dtype: x1.dtype)[torch.TensorIndex.None, torch.TensorIndex.Colon, torch.TensorIndex.None];  // cols shape(1,h,1)
                    return (masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))).MoveToOuterDisposeScope();
                }

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
        internal static torch.Tensor process_mask(torch.Tensor protos, torch.Tensor masks_in, torch.Tensor bboxes, long[] shape, bool upsample = false)
        {
            using (torch.NewDisposeScope())
            {
                long c = protos.shape[0]; //  # CHW
                long mh = protos.shape[1];
                long mw = protos.shape[2];

                long ih = shape[0];
                long iw = shape[1];
                torch.Tensor masks = masks_in.matmul(protos.@float().view(c, -1)).view(-1, mh, mw);  //  # CHW
                float width_ratio = (float)mw / iw;
                float height_ratio = (float)mh / ih;

                torch.Tensor downsampled_bboxes = bboxes.clone();
                downsampled_bboxes[torch.TensorIndex.Ellipsis, 0] *= width_ratio;
                downsampled_bboxes[torch.TensorIndex.Ellipsis, 2] *= width_ratio;
                downsampled_bboxes[torch.TensorIndex.Ellipsis, 3] *= height_ratio;
                downsampled_bboxes[torch.TensorIndex.Ellipsis, 1] *= height_ratio;
                masks = crop_mask(masks, downsampled_bboxes); //  # CHW

                if (upsample)
                {
                    masks = torch.nn.functional.interpolate(masks[torch.TensorIndex.None], size: shape, mode: torch.InterpolationMode.Bilinear, align_corners: false)[0];// # CHW
                }
                return masks.gt_(0.0).MoveToOuterDisposeScope();
            }
        }

        internal static float[] cxcywhr2xyxyxyxy(float[] x)
        {
            float cx = x[0];
            float cy = x[1];
            float w = x[2];
            float h = x[3];
            float r = x[4];
            float cosR = (float)Math.Cos(r);
            float sinR = (float)Math.Sin(r);
            float wHalf = w / 2;
            float hHalf = h / 2;
            return new float[]
            {
                cx - wHalf * cosR + hHalf * sinR,
                cy - wHalf * sinR - hHalf * cosR,
                cx + wHalf * cosR + hHalf * sinR,
                cy + wHalf * sinR - hHalf * cosR,
                cx + wHalf * cosR - hHalf * sinR,
                cy + wHalf * sinR + hHalf * cosR,
                cx - wHalf * cosR - hHalf * sinR,
                cy - wHalf * sinR + hHalf * cosR,
            };
        }

    }
}

