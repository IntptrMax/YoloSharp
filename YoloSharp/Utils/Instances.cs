using System.Diagnostics;
using System.Runtime.CompilerServices;
using TorchSharp;

namespace Utils
{
    internal class Bboxes
    {
        public torch.Tensor bboxes;
        public torchvision.ops.BoxFormats format;

        /// <summary>
        /// Initialize the Bboxes class with bounding box data in a specified format.
        /// </summary>
        /// <param name="bboxes">Array of bounding boxes with shape (N, 4) or (4,).</param>
        /// <param name="format">Format of the bounding boxes, one of 'xyxy', 'xywh'.</param>
        internal Bboxes(torch.Tensor bboxes, torchvision.ops.BoxFormats format = torchvision.ops.BoxFormats.xyxy)
        {
            this.bboxes = bboxes;
            this.format = format;
        }

        /// <summary>
        /// Convert bounding box format from one type to another.
        /// </summary>
        /// <param name="format"></param>
        internal void convert(torchvision.ops.BoxFormats format)
        {
            this.bboxes = torchvision.ops.box_convert(this.bboxes, this.format, format);
            this.format = format;
        }

        /// <summary>
        /// Calculate the area of bounding boxes.
        /// </summary>
        /// <returns></returns>
        internal torch.Tensor areas()
        {
            if (this.format == torchvision.ops.BoxFormats.xyxy)
            {
                return (this.bboxes[torch.TensorIndex.Ellipsis, 2] - this.bboxes[torch.TensorIndex.Ellipsis, 0]) * (this.bboxes[torch.TensorIndex.Ellipsis, 3] - this.bboxes[torch.TensorIndex.Ellipsis, 1]);
            }
            else
            {
                return this.bboxes[torch.TensorIndex.Ellipsis, 3] * this.bboxes[torch.TensorIndex.Ellipsis, 2];
            }
        }

        /// <summary>
        /// Multiply bounding box coordinates by scale factor(s).
        /// </summary>
        /// <param name="scale"></param>
        internal void mul(float[] scale)
        {
            this.bboxes[torch.TensorIndex.Ellipsis, 0] = this.bboxes[torch.TensorIndex.Ellipsis, 0] * scale[0];
            this.bboxes[torch.TensorIndex.Ellipsis, 1] = this.bboxes[torch.TensorIndex.Ellipsis, 1] * scale[1];
            this.bboxes[torch.TensorIndex.Ellipsis, 2] = this.bboxes[torch.TensorIndex.Ellipsis, 2] * scale[2];
            this.bboxes[torch.TensorIndex.Ellipsis, 3] = this.bboxes[torch.TensorIndex.Ellipsis, 3] * scale[3];
        }

        /// <summary>
        /// Multiply bounding box coordinates by scale factor(s).
        /// </summary>
        /// <param name="scale"></param>
        internal void mul(float scale)
        {
            float[] scales = new float[] { scale, scale, scale, scale };
            mul(scales);
        }

        /// <summary>
        /// Add offset to bounding box coordinates.
        /// </summary>
        /// <param name="offset"></param>
        internal void add(float[] offset)
        {
            this.bboxes[torch.TensorIndex.Ellipsis, 0] = this.bboxes[torch.TensorIndex.Ellipsis, 0] + offset[0];
            this.bboxes[torch.TensorIndex.Ellipsis, 1] = this.bboxes[torch.TensorIndex.Ellipsis, 1] + offset[1];
            this.bboxes[torch.TensorIndex.Ellipsis, 2] = this.bboxes[torch.TensorIndex.Ellipsis, 2] + offset[2];
            this.bboxes[torch.TensorIndex.Ellipsis, 3] = this.bboxes[torch.TensorIndex.Ellipsis, 3] + offset[3];
        }

        /// <summary>
        /// Add offset to bounding box coordinates.
        /// </summary>
        /// <param name="offset"></param>
        internal void add(float offset)
        {
            float[] offsets = new float[] { offset, offset, offset, offset };
            add(offsets);
        }

        internal long Length => this.bboxes.shape[0];


        [IndexerName("BboxesItems")]
        internal Bboxes this[int[] index]
        {
            get
            {
                return new Bboxes(this.bboxes[index]);
            }
        }

        [IndexerName("BboxesItems")]
        internal Bboxes this[int index]
        {
            get
            {
                return this[new int[] { index }];
            }
        }

        internal Bboxes Copy()
        {
            return new Bboxes(this.bboxes.clone(), this.format);
        }



    }



    public class Instances
    {
        private Bboxes _bboxes;
        private torch.Tensor keypoints;
        private torch.Tensor segments;
        private torchvision.ops.BoxFormats bbox_format;
        private bool normalized;

        /// <summary>
        /// Initialize the Instances object with bounding boxes, segments, and keypoints.
        /// </summary>
        /// <param name="bboxes">Bounding boxes with shape (N, 4).</param>
        /// <param name="segments">Segmentation masks.</param>
        /// <param name="keypoints">Keypoints with shape (N, 17, 3) in format (x, y, visible).</param>
        /// <param name="bbox_format">Format of bboxes.</param>
        /// <param name="normalized">Whether the coordinates are normalized.</param>
        public Instances(torch.Tensor bboxes, torch.Tensor segments = null, torch.Tensor keypoints = null, torchvision.ops.BoxFormats bbox_format = torchvision.ops.BoxFormats.xywh, bool normalized = true)
        {
            this._bboxes = new Bboxes(bboxes, bbox_format);
            this.keypoints = keypoints;
            this.segments = segments;
            this.bbox_format = bbox_format;
            this.normalized = normalized;
        }

        /// <summary>
        /// Convert bounding box format.
        /// </summary>
        /// <param name="bbox_format"></param>
        internal void convert_bbox(torchvision.ops.BoxFormats bbox_format)
        {
            this._bboxes.convert(bbox_format);
        }

        /// <summary>
        /// Calculate the area of bounding boxes.
        /// </summary>
        /// <returns></returns>
        internal torch.Tensor bbox_areas()
        {
            return this._bboxes.areas();
        }

        /// <summary>
        /// Scale coordinates by given factors.
        /// </summary>
        /// <param name="scale_w">Scale factor for width.</param>
        /// <param name="scale_h">Scale factor for height.</param>
        /// <param name="bbox_only">Whether to scale only bounding boxes.</param>
        internal void scale(float scale_w, float scale_h, bool bbox_only = false)
        {
            this._bboxes.mul(scale: new float[] { scale_w, scale_h, scale_w, scale_h });
            if (bbox_only)
            {
                return;
            }

            this.segments[torch.TensorIndex.Ellipsis, 0] *= scale_w;
            this.segments[torch.TensorIndex.Ellipsis, 1] *= scale_h;

            if (this.keypoints is not null)
            {
                this.keypoints[torch.TensorIndex.Ellipsis, 0] *= scale_w;
                this.keypoints[torch.TensorIndex.Ellipsis, 1] *= scale_h;
            }
        }

        /// <summary>
        /// Convert normalized coordinates to absolute coordinates.
        /// </summary>
        /// <param name="w">Image width.</param>
        /// <param name="h">Image height.</param>
        internal void denormalize(int w, int h)
        {
            if (!this.normalized)
            {
                return;
            }
            this._bboxes.mul(scale: new float[] { w, h, w, h });
            this.segments[torch.TensorIndex.Ellipsis, 0] *= w;
            this.segments[torch.TensorIndex.Ellipsis, 1] *= h;

            if (this.keypoints is not null)
            {
                this.keypoints[torch.TensorIndex.Ellipsis, 0] *= w;
                this.keypoints[torch.TensorIndex.Ellipsis, 1] *= h;
                this.normalized = false;
            }
        }

        /// <summary>
        /// Convert absolute coordinates to normalized coordinates.
        /// </summary>
        /// <param name="w">Image width.</param>
        /// <param name="h">Image height.</param>
        internal void normalize(int w, int h)
        {
            if (this.normalized)
            {
                return;
            }
            this._bboxes.mul(scale: new float[] { 1 / w, 1 / h, 1 / w, 1 / h });

            this.segments[torch.TensorIndex.Ellipsis, 0] /= w;
            this.segments[torch.TensorIndex.Ellipsis, 1] /= h;

            if (this.keypoints is not null)
            {
                this.keypoints[torch.TensorIndex.Ellipsis, 0] /= w;
                this.keypoints[torch.TensorIndex.Ellipsis, 1] /= h;
                this.normalized = true;
            }

        }

        /// <summary>
        /// Add padding to coordinates.
        /// </summary>
        /// <param name="padw">Padding width.</param>
        /// <param name="padh">Padding height.</param>
        internal void add_padding(int padw, int padh)
        {
            Debug.Assert(!this.normalized, "you should add padding with absolute coordinates.");
            this._bboxes.add(offset: new float[] { padw, padh, padw, padh });
            this.segments[torch.TensorIndex.Ellipsis, 0] += padw;
            this.segments[torch.TensorIndex.Ellipsis, 1] += padh;
            if (this.keypoints is not null)
            {
                this.keypoints[torch.TensorIndex.Ellipsis, 0] += padw;
                this.keypoints[torch.TensorIndex.Ellipsis, 1] += padh;
            }
        }

        [IndexerName("InstancesItems")]
        internal Instances this[int[] index]
        {
            get
            {
                torch.Tensor bboxes = this.bboxes.clone();
                torch.Tensor keypoints = this.keypoints[index].clone() ?? null;
                torch.Tensor segments = this.segments[index].clone() ?? null;
                return new Instances(bboxes, segments, keypoints, this.bbox_format, this.normalized);
            }
        }

        [IndexerName("InstancesItems")]
        internal Instances this[int index]
        {
            get
            {
                return this[(new int[] { index })];
            }
        }

        /// <summary>
        /// Flip coordinates vertically.
        /// </summary>
        /// <param name="h">Image height</param>
        internal void flipud(int h)
        {
            if (this.bbox_format == torchvision.ops.BoxFormats.xyxy)
            {
                this.bboxes[torch.TensorIndex.Colon, 1] = h - this.bboxes[torch.TensorIndex.Colon, 3].clone();
                this.bboxes[torch.TensorIndex.Colon, 3] = h - this.bboxes[torch.TensorIndex.Colon, 1].clone();
            }
            else
            {
                this.bboxes[torch.TensorIndex.Colon, 1] = h - this.bboxes[torch.TensorIndex.Colon, 1];
            }

            this.segments[torch.TensorIndex.Ellipsis, 1] = h - this.segments[torch.TensorIndex.Ellipsis, 1];

            if (this.keypoints is not null)
            {
                this.keypoints[torch.TensorIndex.Ellipsis, 1] = h - this.keypoints[torch.TensorIndex.Ellipsis, 1];
            }

        }

        /// <summary>
        /// Flip coordinates horizontally.
        /// </summary>
        /// <param name="w">Image width</param>
        internal void fliplr(int w)
        {
            if (this.bbox_format == torchvision.ops.BoxFormats.xyxy)
            {
                this.bboxes[torch.TensorIndex.Colon, 0] = w - this.bboxes[torch.TensorIndex.Colon, 2].clone();
                this.bboxes[torch.TensorIndex.Colon, 2] = w - this.bboxes[torch.TensorIndex.Colon, 0].clone();
            }
            else
            {
                this.bboxes[torch.TensorIndex.Colon, 1] = w - this.bboxes[torch.TensorIndex.Colon, 0];
            }

            this.segments[torch.TensorIndex.Ellipsis, 0] = w - this.segments[torch.TensorIndex.Ellipsis, 0];

            if (this.keypoints is not null)
            {
                this.keypoints[torch.TensorIndex.Ellipsis, 0] = w - this.keypoints[torch.TensorIndex.Ellipsis, 0];
            }

        }

        /// <summary>
        /// Clip coordinates to stay within image boundaries.
        /// </summary>
        /// <param name="w">Image width.</param>
        /// <param name="h">Image height.</param>
        internal void clip(int w, int h)
        {
            var originalFormat = this.bbox_format;

            convert_bbox(torchvision.ops.BoxFormats.xyxy);

            torch.Tensor bboxTensor = _bboxes.bboxes;

            bboxTensor[torch.TensorIndex.Colon, new long[] { 0, 2 }] = bboxTensor[torch.TensorIndex.Colon, new long[] { 0, 2 }].clip(0, w);
            bboxTensor[torch.TensorIndex.Colon, new long[] { 1, 3 }] = bboxTensor[torch.TensorIndex.Colon, new long[] { 1, 3 }].clip(0, h);

            if (originalFormat != torchvision.ops.BoxFormats.xyxy)
            {
                convert_bbox(originalFormat);
            }
            this.segments[torch.TensorIndex.Ellipsis, 0] = this.segments[torch.TensorIndex.Ellipsis, 0].clip(0, w);
            this.segments[torch.TensorIndex.Ellipsis, 1] = this.segments[torch.TensorIndex.Ellipsis, 1].clip(0, h);

            if (keypoints is not null)
            {
                // Set out of bounds visibility to zero
                torch.Tensor x = keypoints[torch.TensorIndex.Ellipsis, 0];
                torch.Tensor y = keypoints[torch.TensorIndex.Ellipsis, 1];
                torch.Tensor mask = (x < 0) | (x > w) | (y < 0) | (y > h);

                keypoints[torch.TensorIndex.Ellipsis, 2][mask] = 0.0f;

                keypoints[torch.TensorIndex.Ellipsis, 0] = x.clip(0, w);
                keypoints[torch.TensorIndex.Ellipsis, 1] = y.clip(0, h);
            }
        }

        /// <summary>
        /// Remove zero-area boxes, i.e. after clipping some boxes may have zero width or height.
        /// </summary>
        internal torch.Tensor remove_zero_area_boxes()
        {
            torch.Tensor good = this.bbox_areas() > 0;
            if (!good.all().ToBoolean())
            {
                int[] index = good.data<Int32>().ToArray();
                this._bboxes = this._bboxes[index];
                if (this.segments is not null && this.segments.NumberOfElements > 0)
                {
                    this.segments = this.segments[good];
                }

                if (this.keypoints is not null)
                {
                    this.keypoints = this.keypoints[good];
                }

            }

            return good;
        }



        internal torch.Tensor bboxes => this._bboxes.bboxes;


        internal void update(torch.Tensor bboxes, torch.Tensor segments = null, torch.Tensor keypoints = null)
        {
            this._bboxes = new Bboxes(bboxes, format: this._bboxes.format);
            if (segments is not null)
            {
                this.segments = segments;
            }

            if (keypoints is not null)
            {
                this.keypoints = keypoints;
            }
        }

        internal long Length => this._bboxes.Length;



    }
}

