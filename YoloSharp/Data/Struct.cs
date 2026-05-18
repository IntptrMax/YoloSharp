using TorchSharp;

namespace Data
{
    internal class Struct
    {
        internal struct LabelStruct
        {
            public string im_file;
            public string label_file;
            public (int h, int w) org_shape;
            public (int h, int w) resized_shape;
            public (int h, int w) mosic_border;
            public (int h, int w) rectangle_shape;

            public int mask_ratio;

            /// <summary>
            /// Class for the label
            /// </summary>
            public torch.Tensor cls;

            /// <summary>
            /// Image of the label, [C, H, W]
            /// </summary>
            public torch.Tensor img;

            /// <summary>
            /// Bbox of the labe, [n, 4]
            /// </summary>
            public torch.Tensor bboxes;

            /// <summary>
            /// obb data of the label, [n, 4, 2]
            /// </summary>
            public torch.Tensor obb_corners;
            /// <summary>
            /// Mask from the segment points, [1, H, W]
            /// </summary>
            public torch.Tensor mask;

            /// <summary>
            /// Key points of the label, [n, length, ndim]
            /// </summary>
            public torch.Tensor keypoints;

            /// <summary>
            /// Bbox format
            /// </summary>
            public torchvision.ops.BoxFormats bbox_format;

            /// <summary>
            /// Weather the info is normalized.
            /// </summary>
            public bool normalized;

            public LabelStruct Clone()
            {
                return new LabelStruct
                {
                    im_file = this.im_file,
                    label_file = this.label_file,
                    org_shape = this.org_shape,
                    resized_shape = this.resized_shape,
                    cls = this.cls?.clone(),
                    img = this.img?.clone(),
                    bboxes = this.bboxes?.clone(),
                    mask = this.mask?.clone(),
                    keypoints = this.keypoints?.clone(),
                    normalized = this.normalized,
                    bbox_format = this.bbox_format,
                    mask_ratio = this.mask_ratio,
                    mosic_border = this.mosic_border,
                    obb_corners = this.obb_corners?.clone(),
                    rectangle_shape = this.rectangle_shape,
                };
            }

            public void DeNormalize()
            {
                if (!this.normalized)
                {
                    return;
                }
                (int h, int w) = this.resized_shape;
                this.bboxes = this.bboxes.mul(new float[] { w, h, w, h });
                if (this.keypoints is not null)
                {
                    this.keypoints[torch.TensorIndex.Ellipsis, ..2] = this.keypoints[torch.TensorIndex.Ellipsis, ..2].mul(new float[] { w, h });
                }
                if (this.obb_corners is not null)
                {
                    this.obb_corners = this.obb_corners.mul(new float[] { w, h });
                }
                this.normalized = false;

            }

            public void Normalize()
            {
                if (this.normalized)
                {
                    return;
                }

                (int h, int w) = (this.rectangle_shape.w > 0 && this.rectangle_shape.h > 0) ? (this.rectangle_shape.h, this.rectangle_shape.w) : this.resized_shape;
                float ww = 1f / w;
                float hh = 1f / h;

                this.bboxes = this.bboxes.mul(new float[] { ww, hh, ww, hh });

                if (this.keypoints is not null)
                {
                    this.keypoints[torch.TensorIndex.Ellipsis, ..2] = this.keypoints[torch.TensorIndex.Ellipsis, ..2].mul(new float[] { ww, hh });
                }
                if (this.obb_corners is not null)
                {
                    this.obb_corners = this.obb_corners.mul(new float[] { ww, hh });
                }
                this.normalized = true;
            }

            private OpenCvSharp.Mat GetMaskFromOutlinePoints(OpenCvSharp.Point[] points, int height, int width)
            {
                OpenCvSharp.Mat mask = OpenCvSharp.Mat.Zeros(height, width, OpenCvSharp.MatType.CV_8UC1);
                OpenCvSharp.Point[][] pts = new OpenCvSharp.Point[1][];
                pts[0] = points.Select(p => new OpenCvSharp.Point((int)p.X, (int)p.Y)).ToArray();
                OpenCvSharp.Cv2.FillPoly(mask, pts, OpenCvSharp.Scalar.White);
                return mask;
            }

        }




    }
}
