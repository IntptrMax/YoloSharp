using System.Diagnostics;
using TorchSharp;

namespace Data
{
    internal class Augment
    {
        public interface ITransform
        {
            public Struct.LabelStruct Apply(Struct.LabelStruct label);
        }

        internal class Compose : ITransform
        {
            private List<ITransform> transforms;
            public Compose(ITransform transforms)
            {
                this.transforms = new List<ITransform> { transforms };
            }

            public Compose(List<ITransform> transforms)
            {
                this.transforms = transforms;
            }

            public Compose()
            {
                this.transforms = new List<ITransform>();
            }

            public Struct.LabelStruct Apply(Struct.LabelStruct label)
            {
                for (int i = 0; i < transforms.Count; i++)
                {
                    label = transforms[i].Apply(label);
                }
                return label;
            }

            public void Add(ITransform transform)
            {
                this.transforms.Add(transform);
            }

            public void Insert(int index, ITransform transform)
            {
                this.transforms.Insert(index, transform);
            }

            public ITransform this[int index]
            {
                get
                {
                    return this.transforms[index];
                }
                set
                {
                    this.transforms[index] = value;
                }
            }

            public List<ITransform> ToList()
            {
                return this.transforms;
            }

        }

        internal abstract class BaseMixTransform : ITransform
        {
            protected readonly YoloDataset dataset;
            protected readonly ITransform pre_transform;
            protected readonly float p;

            /// <summary>
            /// Initialize the BaseMixTransform object for mix transformations like CutMix, MixUp and Mosaic.<br/>
            /// This class serves as a base for implementing mix transformations in image processing pipelines.
            /// </summary>
            /// <param name="dataset">The dataset object containing images and labels for mixing.</param>
            /// <param name="pre_transform">Optional transform to apply before mixing.</param>
            /// <param name="p">Probability of applying the mix transformation. Should be in the range [0.0, 1.0].</param>
            internal BaseMixTransform(YoloDataset dataset, ITransform pre_transform = null, float p = 0f)
            {
                this.dataset = dataset;
                this.pre_transform = pre_transform;
                this.p = p;
            }

            internal virtual long[] get_indexes()
            {
                return new long[] { torch.randint_long(this.dataset.Count - 1) };
            }

            internal Struct.LabelStruct[] get_mix_labels()
            {
                long[] indexs = get_indexes();
                int length = indexs.Length;
                Struct.LabelStruct[] mix_labels = new Struct.LabelStruct[length];

                for (int i = 0; i < indexs.Length; i++)
                {
                    mix_labels[i] = dataset.labels[(int)indexs[i]];
                }
                return mix_labels;
            }


            protected abstract Struct.LabelStruct _mix_transform(Struct.LabelStruct labels, long[] indexes);

            Struct.LabelStruct ITransform.Apply(Struct.LabelStruct label)
            {
                if (torch.rand_float() > this.p)
                {
                    return label;
                }

                if (this.pre_transform is not null)
                {
                    label = this.pre_transform.Apply(label);
                }
                long[] indexes = get_indexes();
                return _mix_transform(label, indexes);
            }
        }

        internal class Mosaic : BaseMixTransform
        {
            private int imgsz;
            private (int width, int height) border;
            private int n;

            /// <summary>
            /// Mosaic augmentation for image datasets.
            /// </summary>
            /// <param name="dataset">The dataset on which the mosaic augmentation is applied.</param>
            /// <param name="imgsz">Image size (height and width) after mosaic pipeline of a single image.</param>
            /// <param name="p">Probability of applying the mosaic augmentation. Must be in the range 0-1.</param>
            /// <param name="n">The grid size, either 4 (for 2x2) or 9 (for 3x3).</param>
            internal Mosaic(YoloDataset dataset, ITransform pre_transform = null, int imgsz = 640, float p = 1f, int n = 4) : base(dataset: dataset, p: p, pre_transform: pre_transform)
            {
                Debug.Assert(0 <= p && p <= 1.0, "The probability should be in range [0, 1], but got {p}.");
                Debug.Assert(n == 4 || n == 9, "grid must be equal to 4 or 9.");
                this.imgsz = imgsz;
                this.border = (-imgsz / 2, -imgsz / 2); // width, height
                this.n = n;
            }

            protected override Struct.LabelStruct _mix_transform(Struct.LabelStruct labels, long[] indexs)
            {
                return _mosaic4(labels);
            }

            internal override long[] get_indexes()
            {
                return torch.randint(0, this.dataset.Count - 1, new torch.Size(this.n - 1)).data<Int64>().ToArray();
            }

            private Struct.LabelStruct _mosaic4(Struct.LabelStruct label)
            {
                Struct.LabelStruct[] mix_labels = get_mix_labels();
                int s = this.imgsz;
                long channels = label.img.shape[0];
                int yc = torch.randint_int(-this.border.width, 2 * s + this.border.width);
                int xc = torch.randint_int(-this.border.height, 2 * s + this.border.height);
                int x1a = 0, x1b = 0, y1a = 0, y1b = 0, x2a = 0, x2b = 0, y2a = 0, y2b = 0;
                torch.Tensor img4 = torch.full(new long[] { channels, s * 2, s * 2 }, 114, dtype: torch.ScalarType.Byte); // base image with 4 tiles
                int mask_ratio = label.mask_ratio;
                torch.Tensor mask4 = label.mask is not null ? torch.full(new long[] { 1, s * 2 / mask_ratio, s * 2 / mask_ratio }, 0, dtype: torch.ScalarType.Byte) : null;  // base mask with 4 mask tiles
                List<torch.Tensor> bboxesList = new List<torch.Tensor>();
                List<torch.Tensor> keypointsList = new List<torch.Tensor>();
                List<torch.Tensor> clsList = new List<torch.Tensor>();
                List<torch.Tensor> obbCornersList = new List<torch.Tensor>();
                for (int i = 0; i < 4; i++)
                {
                    Struct.LabelStruct label_patch = i == 0 ? label : mix_labels[i - 1];
                    torch.Tensor img = label_patch.img;
                    torch.Tensor mask = label_patch.mask;
                    torch.Tensor bboxes = label_patch.bboxes;
                    torch.Tensor cls = label_patch.cls;
                    torch.Tensor keypoints = label_patch.keypoints;
                    torch.Tensor obb_corners = label_patch.obb_corners;
                    (int h, int w) = label_patch.resized_shape;

                    if (i == 0)  // top left
                    {
                        (x1a, y1a, x2a, y2a) = (Math.Max(xc - w, 0), Math.Max(yc - h, 0), xc, yc); // xmin, ymin, xmax, ymax (large image)
                        (x1b, y1b, x2b, y2b) = (w - (x2a - x1a), h - (y2a - y1a), w, h); // xmin, ymin, xmax, ymax (small image)
                    }
                    else if (i == 1)  // top right
                    {
                        (x1a, y1a, x2a, y2a) = (xc, Math.Max(yc - h, 0), Math.Min(xc + w, s * 2), yc);
                        (x1b, y1b, x2b, y2b) = (0, h - (y2a - y1a), Math.Min(w, x2a - x1a), h);
                    }
                    else if (i == 2)  // bottom left
                    {
                        (x1a, y1a, x2a, y2a) = (Math.Max(xc - w, 0), yc, xc, Math.Min(s * 2, yc + h));
                        (x1b, y1b, x2b, y2b) = (w - (x2a - x1a), 0, w, Math.Min(y2a - y1a, h));
                    }
                    else if (i == 3)  // bottom right
                    {
                        (x1a, y1a, x2a, y2a) = (xc, yc, Math.Min(xc + w, s * 2), Math.Min(s * 2, yc + h));
                        (x1b, y1b, x2b, y2b) = (0, 0, Math.Min(w, x2a - x1a), Math.Min(y2a - y1a, h));
                    }
                    img4[torch.TensorIndex.Ellipsis, y1a..y2a, x1a..x2a] = img[torch.TensorIndex.Ellipsis, y1b..y2b, x1b..x2b]; // img4[ymin:ymax, xmin:xmax]
                    if (label.mask is not null)
                    {
                        int xl = (y2a / mask_ratio) - (y1a / mask_ratio);
                        int yl = (x2a / mask_ratio) - (x1a / mask_ratio);
                        mask4[torch.TensorIndex.Ellipsis, (y1a / mask_ratio)..(y2a / mask_ratio), (x1a / mask_ratio)..(x2a / mask_ratio)] = mask[torch.TensorIndex.Ellipsis, (y1b / mask_ratio)..(y1b / mask_ratio + xl), (x1b / mask_ratio)..(x1b / mask_ratio + yl)];
                    }
                    int padw = x1a - x1b;
                    int padh = y1a - y1b;
                    if (cls is null)
                    {
                        continue;
                    }
                    bboxes = torchvision.ops.box_convert(bboxes, label_patch.bbox_format, torchvision.ops.BoxFormats.xyxy);
                    bboxes = bboxes.add(new long[] { padw, padh, padw, padh });
                    bboxesList.Add(bboxes);

                    if (keypoints is not null)
                    {
                        keypoints = label_patch.keypoints.clone();
                        keypoints[torch.TensorIndex.Ellipsis, 0..2] = keypoints[torch.TensorIndex.Ellipsis, 0..2].add(new long[] { padw, padh });
                        keypointsList.Add(keypoints);
                    }
                    if (obb_corners is not null)
                    {
                        obb_corners = label_patch.obb_corners.clone();
                        obb_corners = obb_corners.add(new long[] { padw, padh });
                        obbCornersList.Add(obb_corners);
                    }

                    clsList.Add(cls);

                }

                torch.Tensor mix_bboxes = torch.cat(bboxesList);
                torch.Tensor mix_keypoints = keypointsList.Count > 0 ? torch.cat(keypointsList) : null;
                torch.Tensor mix_cls = torch.cat(clsList);
                torch.Tensor mix_obb_corners = obbCornersList.Count > 0 ? torch.cat(obbCornersList) : null;
                torch.Tensor org_areas = torchvision.ops.box_area(mix_bboxes);
                mix_bboxes = mix_bboxes.clip(0, s * 2);
                torch.Tensor areas = torchvision.ops.box_area(mix_bboxes);
                torch.Tensor good = (areas > 0) & (areas > 0.7 * org_areas);
                //torch.Tensor good = areas > 0;
                mix_cls = mix_cls[good];
                mix_bboxes = mix_bboxes[good];
                if (mix_keypoints is not null)
                {
                    mix_keypoints = mix_keypoints[good];
                }
                if (mix_obb_corners is not null)
                {
                    mix_obb_corners = mix_obb_corners[good];
                }

                Struct.LabelStruct mix_label = new Struct.LabelStruct();
                mix_label.img = img4;
                mix_label.mask = mask4;
                mix_label.normalized = false;
                mix_label.cls = mix_cls;
                mix_label.resized_shape = (s * 2, s * 2);
                mix_label.bboxes = mix_bboxes;
                mix_label.bbox_format = torchvision.ops.BoxFormats.xyxy;
                mix_label.im_file = label.im_file;
                mix_label.keypoints = mix_keypoints;
                mix_label.org_shape = label.org_shape;
                mix_label.mosic_border = (this.border.width, this.border.height);
                mix_label.mask_ratio = label.mask_ratio;
                mix_label.obb_corners = mix_obb_corners;

                return mix_label;
            }
        }

        internal class RandomPerspective : ITransform
        {
            private readonly float degrees;
            private readonly float scale;
            private readonly float translate;
            private readonly float shear;
            private readonly float perspective;
            private int border_w;
            private int border_h;
            private readonly ITransform pre_transform;

            private (int w, int h) size;

            internal RandomPerspective(float degrees = 0f, float translate = 0.1f, float scale = 0.5f, float shear = 0.0f, float perspective = 0.0f, ITransform pre_transform = null)
            {
                this.degrees = degrees;
                this.translate = translate;
                this.scale = scale;
                this.shear = shear;
                this.perspective = perspective;
                this.pre_transform = pre_transform;
            }

            /// <summary>
            /// Apply a sequence of affine transformations centered around the image center.
            /// </summary>
            /// <remarks>
            /// This function performs a series of geometric transformations on the input image, including translation, 
            /// perspective change, rotation, scaling, and shearing.The transformations are applied in a specific order to 
            /// maintain consistency.
            /// </remarks>
            /// <param name="img">Input image to be transformed.</param>
            /// <param name="border">Border dimensions for the transformed image.</param>
            /// <returns>
            /// img: Transformed image.<br/>
            /// M: 3x3 transformation matrix.<br/>
            /// s: Scale factor applied during the transformation.<br/>
            /// </returns>
            private (torch.Tensor img, torch.Tensor mask, torch.Tensor M, float s, float a) affine_transform(Struct.LabelStruct label, (int, int) border)
            {
                using (torch.no_grad())
                using (torch.NewDisposeScope())
                {
                    torch.Tensor img = label.img;
                    torch.Tensor mask = label.mask;
                    // Center
                    torch.Tensor C = torch.eye(3, dtype: torch.ScalarType.Float32);

                    C[0, 2] = -img.shape[^1] / 2;  // x translation (pixels)
                    C[1, 2] = -img.shape[^2] / 2;  // y translation (pixels)

                    // Perspective
                    torch.Tensor P = torch.eye(3, dtype: torch.ScalarType.Float32);
                    P[2, 0] = (torch.rand_float() * 2 - 1) * this.perspective; // x perspective (about y)
                    P[2, 1] = (torch.rand_float() * 2 - 1) * this.perspective; // y perspective (about x)

                    // Rotation and Scale
                    torch.Tensor R = torch.eye(3, dtype: torch.ScalarType.Float32);
                    float a = (float)(torch.rand_float() * 2 - 1) * this.degrees;

                    // a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
                    float s = 1 + (float)(torch.rand_float() * 2 - 1) * this.scale;
                    // s = 2 ** random.uniform(-scale, scale)

                    R[..2] = GetRotationMatrix2D(0, 0, angle: a, scale: s);

                    // Shear
                    torch.Tensor S = torch.eye(3, dtype: torch.ScalarType.Float32);

                    S[0, 1] = Math.Tan((torch.rand_float() * 2 - 1) * this.shear * Math.PI / 180f); // x shear (deg)
                    S[1, 0] = Math.Tan((torch.rand_float() * 2 - 1) * this.shear * Math.PI / 180f); // y shear (deg)

                    // Translation
                    torch.Tensor T = torch.eye(3, dtype: torch.ScalarType.Float32);

                    T[0, 2] = (0.5 + (torch.rand_float() * 2 - 1) * this.translate) * this.size.w;  // x translation (pixels)
                    T[1, 2] = (0.5 + (torch.rand_float() * 2 - 1) * this.translate) * this.size.h;  // y translation (pixels)

                    // Combined rotation matrix
                    torch.Tensor M = T.mm(S.mm(R.mm(P.mm(C))));  // order of operations (right to left) is IMPORTANT

                    torch.Tensor outImg = img.clone();
                    torch.Tensor outMask = null;
                    if (this.border_h != 0 || this.border_w != 0 || !torch.all(torch.eq(M, torch.eye(3, dtype: torch.float32))).ToBoolean())
                    {
                        if (this.perspective > 0)
                        {
                            outImg = WarpPerspectiveWithGridSample(img, M, (this.size.w, this.size.h), new float[] { 114, 114, 114 });
                        }
                        else
                        {
                            outImg = WarpAffineWithGridSample(img, M[..2], (this.size.w, this.size.h), new float[] { 114, 114, 114 });
                        }

                        if (mask is not null)
                        {
                            float r = label.mask_ratio;
                            torch.Tensor S_mask = torch.diag(torch.tensor(new float[] { r, r, 1 }, device: img.device, dtype: torch.float32));
                            torch.Tensor S_inv = torch.diag(torch.tensor(new float[] { 1f / r, 1f / r, 1 }, device: img.device, dtype: torch.float32));
                            torch.Tensor M_mask = S_inv.mm(M).mm(S_mask);

                            int maskOutW = (int)(this.size.w / r);
                            int maskOutH = (int)(this.size.h / r);

                            if (this.perspective > 0)
                            {
                                outMask = WarpPerspectiveWithGridSample(mask, M_mask, (maskOutW, maskOutH), new float[] { 0 });
                            }
                            else
                            {
                                outMask = WarpAffineWithGridSample(mask, M_mask[..2], (maskOutW, maskOutH), new float[] { 0 });
                            }
                        }
                    }
                    return (outImg.MoveToOuterDisposeScope(), outMask?.MoveToOuterDisposeScope(), M.MoveToOuterDisposeScope(), s, a);
                }
            }

            private torch.Tensor WarpPerspectiveWithGridSample(torch.Tensor img, torch.Tensor M, (int w, int h) outputSize, float[] borderValue)
            {
                using (torch.no_grad())
                using (var scope = torch.NewDisposeScope())
                {
                    torch.ScalarType originalDtype = img.dtype;
                    img = img.to(torch.float32);

                    int c = (int)img.shape[0];
                    int inH = (int)img.shape[1];
                    int inW = (int)img.shape[2];
                    int outW = outputSize.w;
                    int outH = outputSize.h;

                    torch.Tensor M_inv = torch.linalg.inv(M);   // 3x3

                    float[] xs = new float[outW];
                    float[] ys = new float[outH];
                    for (int x = 0; x < outW; x++)
                    {
                        xs[x] = x;
                    }
                    for (int y = 0; y < outH; y++)
                    {
                        ys[y] = y;
                    }
                    torch.Tensor gridX = torch.tensor(xs, device: img.device).view(1, outW).repeat(outH, 1);
                    torch.Tensor gridY = torch.tensor(ys, device: img.device).view(outH, 1).repeat(1, outW);
                    torch.Tensor ones = torch.ones_like(gridX);

                    torch.Tensor gridHom = torch.stack(new[] { gridX, gridY, ones }, dim: 0);  // (3, outH, outW)
                    torch.Tensor gridFlat = gridHom.view(3, -1);
                    torch.Tensor srcFlat = M_inv.mm(gridFlat);
                    torch.Tensor srcXY = srcFlat[..2, ..];
                    torch.Tensor w_ = srcFlat[2, ..].unsqueeze(0);
                    torch.Tensor src = srcXY / w_;   // (2, N)
                    src = src.view(2, outH, outW);   // (2, outH, outW)

                    torch.Tensor grid = torch.zeros(1, outH, outW, 2, device: img.device);
                    grid[0, .., .., 0] = src[0] / (inW - 1) * 2 - 1;
                    grid[0, .., .., 1] = src[1] / (inH - 1) * 2 - 1;

                    img = img.unsqueeze(0);
                    torch.Tensor sampled = torch.nn.functional.grid_sample(
                        img, grid,
                        mode: torch.GridSampleMode.Bilinear,
                        padding_mode: torch.GridSamplePaddingMode.Border,
                        align_corners: false
                    );   // (1, C, outH, outW)

                    torch.Tensor px = src[0];
                    torch.Tensor py = src[1];
                    torch.Tensor valid = (px >= 0) & (px <= inW - 1) & (py >= 0) & (py <= inH - 1);

                    torch.Tensor borderTensor = torch.tensor(borderValue, dtype: torch.float32, device: img.device).view(c, 1, 1);
                    torch.Tensor result = torch.where(valid.unsqueeze(0), sampled[0], borderTensor.expand_as(sampled[0]));

                    if (originalDtype == torch.uint8)
                    {
                        result = torch.clamp(result, 0, 255).to(torch.uint8);
                    }
                    else
                    {
                        result = result.to(originalDtype);
                    }

                    return result.MoveToOuterDisposeScope();
                }
            }

            private torch.Tensor WarpAffineWithGridSample(torch.Tensor img, torch.Tensor M_2x3, (int w, int h) outputSize, float[] borderValue)
            {
                using (torch.no_grad())
                using (var scope = torch.NewDisposeScope())
                {
                    var originalDtype = img.dtype;
                    img = img.to(torch.float32);

                    int c = (int)img.shape[0];
                    int inH = (int)img.shape[1];
                    int inW = (int)img.shape[2];
                    int outW = outputSize.w;
                    int outH = outputSize.h;

                    torch.Tensor M_3x3 = torch.eye(3, dtype: torch.float32, device: img.device);
                    M_3x3[0, 0] = M_2x3[0, 0];
                    M_3x3[0, 1] = M_2x3[0, 1];
                    M_3x3[0, 2] = M_2x3[0, 2];
                    M_3x3[1, 0] = M_2x3[1, 0];
                    M_3x3[1, 1] = M_2x3[1, 1];
                    M_3x3[1, 2] = M_2x3[1, 2];
                    torch.Tensor M_inv = torch.linalg.inv(M_3x3);

                    float[] xs = new float[outW];
                    float[] ys = new float[outH];
                    for (int x = 0; x < outW; x++)
                    {
                        xs[x] = x;
                    }
                    for (int y = 0; y < outH; y++)
                    {
                        ys[y] = y;
                    }
                    torch.Tensor gridX = torch.tensor(xs, device: img.device).view(1, outW).repeat(outH, 1);
                    torch.Tensor gridY = torch.tensor(ys, device: img.device).view(outH, 1).repeat(1, outW);
                    torch.Tensor ones = torch.ones_like(gridX);

                    torch.Tensor gridHom = torch.stack(new[] { gridX, gridY, ones }, dim: 0);   // (3, outH, outW)
                    torch.Tensor gridFlat = gridHom.view(3, -1);                               // (3, N)
                    torch.Tensor srcFlat = M_inv.mm(gridFlat);                                 // (3, N)
                    torch.Tensor srcXY = srcFlat[..2, ..];
                    torch.Tensor w_ = srcFlat[2, ..].unsqueeze(0);                             // (1, N)
                    torch.Tensor src = srcXY / w_;
                    src = src.view(2, outH, outW);                                            // (2, outH, outW)
                    torch.Tensor srcX = src[0];
                    torch.Tensor srcY = src[1];

                    torch.Tensor grid = torch.zeros(1, outH, outW, 2, device: img.device);
                    grid[0, .., .., 0] = srcX / (inW - 1) * 2 - 1;
                    grid[0, .., .., 1] = srcY / (inH - 1) * 2 - 1;

                    torch.Tensor sampled = torch.nn.functional.grid_sample(
                        img.unsqueeze(0), grid,
                        mode: torch.GridSampleMode.Bilinear,
                        padding_mode: torch.GridSamplePaddingMode.Border,
                        align_corners: false
                    );   // (1, C, outH, outW)

                    torch.Tensor valid = (srcX >= 0) & (srcX <= inW - 1) & (srcY >= 0) & (srcY <= inH - 1);
                    torch.Tensor borderTensor = torch.tensor(borderValue, dtype: torch.float32, device: img.device).view(c, 1, 1);
                    torch.Tensor result = torch.where(valid.unsqueeze(0), sampled[0], borderTensor.expand_as(sampled[0]));

                    if (originalDtype == torch.uint8)
                    {
                        result = torch.clamp(result, 0, 255).to(torch.uint8);
                    }
                    else
                    {
                        result = result.to(originalDtype);
                    }

                    return result.MoveToOuterDisposeScope();
                }
            }

            /// <summary>
            /// This function applies an affine transformation to a set of bounding boxes using the provided transformation matrix.
            /// </summary>
            /// <param name="bboxes">Bounding boxes in xyxy format with shape (N, 4), where N is the number of bounding boxes.</param>
            /// <param name="M">Affine transformation matrix with shape (3, 3).</param>
            /// <returns>Transformed bounding boxes in xyxy format with shape (N, 4).</returns>
            private torch.Tensor apply_bboxes(torch.Tensor bboxes, torch.Tensor M)
            {
                using (torch.no_grad())
                using (torch.NewDisposeScope())
                {
                    long n = bboxes.shape[0];
                    if (n == 0)
                    {
                        return bboxes.clone();
                    }
                    torch.Tensor xy = torch.ones(new long[] { n * 4, 3 }, dtype: bboxes.dtype);
                    xy[torch.TensorIndex.Colon, ..2] = bboxes[torch.TensorIndex.Colon, torch.TensorIndex.Tensor(new long[] { 0, 1, 2, 3, 0, 3, 2, 1 })].reshape(n * 4, 2);  // x1y1, x2y2, x1y2, x2y1
                    xy = xy.mm(M.T); // transform

                    xy = (this.perspective > 0 ? (xy[torch.TensorIndex.Colon, ..2] / xy[torch.TensorIndex.Colon, 2..3]) : xy[torch.TensorIndex.Colon, ..2]).reshape(n, 8); // perspective rescale or affine   

                    // Create new boxes
                    torch.Tensor x = xy[torch.TensorIndex.Colon, torch.TensorIndex.Tensor(new long[] { 0, 2, 4, 6 })];
                    torch.Tensor y = xy[torch.TensorIndex.Colon, torch.TensorIndex.Tensor(new long[] { 1, 3, 5, 7 })];

                    return torch.concatenate(new torch.Tensor[] { x.min(1).values, y.min(1).values, x.max(1).values, y.max(1).values }).reshape(4, n).T.MoveToOuterDisposeScope();
                }
            }

            /// <summary>
            /// Apply affine transformation to keypoints.
            /// </summary>
            /// <remarks>
            /// This method transforms the input keypoints using the provided affine transformation matrix. It handles 
            /// perspective rescaling if necessary and updates the visibility of keypoints that fall outside the image
            /// boundaries after transformation.
            /// </remarks>
            /// <param name="keypoints">Array of keypoints with shape (N, K, 3), where N is the number of instances, K is the number of keypoints per instance, and 3 represents(x, y, visibility).</param>
            /// <param name="M">3x3 affine transformation matrix.</param>
            /// <returns>Transformed keypoints array with the same shape as input (N, K, 3).</returns>
            private torch.Tensor apply_keypoints(torch.Tensor keypoints, torch.Tensor M)
            {
                using (torch.no_grad())
                using (torch.NewDisposeScope())
                {
                    int n = (int)keypoints.shape[0];
                    int nkpt = (int)keypoints.shape[1];
                    if (n == 0)
                    {
                        return keypoints;
                    }
                    torch.Tensor xy = torch.ones(new long[] { n * nkpt, 3 }, dtype: keypoints.dtype);
                    torch.Tensor visible = keypoints[torch.TensorIndex.Ellipsis, 2].reshape(n * nkpt, 1);
                    xy[torch.TensorIndex.Colon, ..2] = keypoints[torch.TensorIndex.Ellipsis, ..2].reshape(n * nkpt, 2);
                    xy = xy.mm(M.T);  // transform
                    xy = xy[torch.TensorIndex.Colon, ..2] / xy[torch.TensorIndex.Colon, 2..3];  // perspective rescale or affine
                    torch.Tensor out_mask = (xy[torch.TensorIndex.Colon, 0] < 0) | (xy[torch.TensorIndex.Colon, 1] < 0) | (xy[torch.TensorIndex.Colon, 0] > this.size.w) | (xy[torch.TensorIndex.Colon, 1] > this.size.h);
                    visible[out_mask] = 0;
                    return torch.concatenate(new torch.Tensor[] { xy, visible }, axis: -1).reshape(n, nkpt, 3).MoveToOuterDisposeScope();
                }
            }


            private torch.Tensor apply_obb_corners(torch.Tensor bboxes, torch.Tensor M)
            {
                using (torch.no_grad())
                using (torch.NewDisposeScope())
                {
                    int n = (int)bboxes.shape[0];
                    if (n == 0)
                    {
                        return bboxes;
                    }

                    torch.Tensor xy = bboxes.view(-1, 2);  // (n*4, 2)

                    torch.Tensor ones = torch.ones(new long[] { xy.size(0), 1 }, dtype: bboxes.dtype, device: bboxes.device);
                    torch.Tensor xyHom = torch.cat(new[] { xy, ones }, dim: 1);

                    torch.Tensor transformed = xyHom.mm(M.T);  // (n*4, 3)

                    torch.Tensor xyTrans;
                    if (this.perspective > 0)
                    {
                        xyTrans = transformed[torch.TensorIndex.Colon, ..2] / transformed[torch.TensorIndex.Colon, 2..3];
                    }
                    else
                    {
                        xyTrans = transformed[torch.TensorIndex.Colon, ..2];
                    }

                    torch.Tensor newBboxes = xyTrans.reshape(n, 4, 2);
                    return newBboxes.MoveToOuterDisposeScope();
                }
            }

            /// <summary>
            /// Calculates an affine matrix of 2D rotation.
            /// </summary>
            /// <param name="cx">Center x of the rotation in the source image.</param>
            /// <param name="cy">Center y of the rotation in the source image.</param>
            /// <param name="angle">Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).</param>
            /// <param name="scale">Isotropic scale factor.</param>
            /// <returns></returns>
            private static torch.Tensor GetRotationMatrix2D(float cx, float cy, float angle, float scale)
            {
                float rad = angle * (float)Math.PI / 180.0f;
                float cos = (float)Math.Cos(rad);
                float sin = (float)Math.Sin(rad);
                float alpha = cos * scale;
                float beta = sin * scale;

                float tx = (1 - alpha) * cx - beta * cy;
                float ty = beta * cx + (1 - alpha) * cy;

                torch.Tensor rot2x3 = torch.tensor(new float[,]
                {
                    { alpha,  beta, tx },
                    { -beta, alpha, ty }
                }, dtype: torch.float32);
                return rot2x3;
            }

            public Struct.LabelStruct Apply(Struct.LabelStruct label)
            {
                if (label.cls.NumberOfElements < 1)
                {
                    return label;
                }
                Struct.LabelStruct result = label.Clone();
                result = this.pre_transform is null ? result : this.pre_transform.Apply(result);

                this.border_w = label.mosic_border.w;
                this.border_h = label.mosic_border.h;
                this.size = (label.resized_shape.w + this.border_w * 2, label.resized_shape.h + this.border_h * 2);
                (torch.Tensor img, torch.Tensor mask, torch.Tensor M, float s, float a) = affine_transform(label, (this.border_h, this.border_w));
                result.img = img;
                result.mask = mask;
                result.resized_shape = this.size;

                torch.Tensor bboxes = apply_bboxes(label.bboxes, M);
                bboxes = YoloSharp.Utils.Ops.clip_boxes(bboxes, new float[] { this.size.h, this.size.w });
                torch.Tensor areas = torchvision.ops.box_area(bboxes);
                torch.Tensor good = areas > 0;
                bboxes = bboxes[good];

                result.bboxes = bboxes;
                torch.Tensor keypoints = label.keypoints is not null ? apply_keypoints(label.keypoints, M)[good] : null;
                torch.Tensor obb_corners = label.obb_corners is not null ? apply_obb_corners(label.obb_corners, M)[good] : null;
                result.keypoints = label.keypoints is null ? null : YoloSharp.Utils.Ops.clip_keypoints(keypoints, new float[] { this.size.h, this.size.w });
                result.obb_corners = label.obb_corners is null ? null : YoloSharp.Utils.Ops.clip_obb_corners(obb_corners, new float[] { this.size.h, this.size.w });
                result.cls = result.cls[good];
                return result;

            }
        }

        internal class LetterBox : ITransform
        {
            private readonly int resized_width;
            private readonly int resized_height;
            private readonly int mask_ratio;
            private readonly int color;
            internal LetterBox(int resized_width = 640, int resized_height = 640, int mask_ratio = 4, int color = 114)
            {
                this.resized_width = resized_width;
                this.resized_height = resized_height;
                this.mask_ratio = mask_ratio;
                this.color = color;
            }
            Struct.LabelStruct ITransform.Apply(Struct.LabelStruct label)
            {
                if (label.normalized)
                {
                    throw new ArgumentException("Label must be denormalized for LetterBox.");
                }
                Struct.LabelStruct transformedLabel = label.Clone();

                (int pad_l, int pad_u, torch.Tensor img) = LetterboxImage(transformedLabel.img, resized_width, resized_height, color);

                transformedLabel.img = img;

                if (transformedLabel.mask is not null)
                {
                    (_, _, torch.Tensor mask) = LetterboxImage(transformedLabel.mask, resized_width / mask_ratio, resized_height / mask_ratio, 0);
                    transformedLabel.mask = mask;
                }

                if (transformedLabel.bboxes is not null)
                {
                    if (transformedLabel.bbox_format == torchvision.ops.BoxFormats.xyxy)
                    {
                        transformedLabel.bboxes = transformedLabel.bboxes.add(new float[] { pad_l, pad_u, pad_l, pad_u });
                    }
                    else
                    {
                        transformedLabel.bboxes = transformedLabel.bboxes.add(new float[] { pad_l, pad_u, 0, 0 });
                    }
                }

                if (transformedLabel.keypoints is not null)
                {
                    transformedLabel.keypoints[torch.TensorIndex.Ellipsis, 0..2] = transformedLabel.keypoints[torch.TensorIndex.Ellipsis, 0..2].add(new float[] { pad_l, pad_u });
                }

                if (transformedLabel.obb_corners is not null)
                {
                    transformedLabel.obb_corners = transformedLabel.obb_corners.add(new float[] { pad_l, pad_u });
                }

                transformedLabel.resized_shape = (this.resized_height, this.resized_width);

                return transformedLabel;
            }

            private (int pad_l, int pad_u, torch.Tensor letterboxed_img) LetterboxImage(torch.Tensor img, int w, int h, int color = 0)
            {
                int imgH = (int)img.shape[1];
                int imgW = (int)img.shape[2];

                float ratio_w = w / (float)imgW;
                float ratio_h = h / (float)imgH;
                float ratio = Math.Min(ratio_w, ratio_h);

                int new_w = (int)(imgW * ratio);
                int new_h = (int)(imgH * ratio);

                int pad_l = (w - new_w) / 2;
                int pad_r = w - new_w - pad_l;
                int pad_u = (h - new_h) / 2;
                int pad_d = h - new_h - pad_u;

                torch.Tensor img_resized = torchvision.transforms.functional.resize(img, new_h, new_w);
                torch.Tensor img_padded = torchvision.transforms.functional.pad(img_resized, new long[] { pad_l, pad_u, pad_r, pad_d }, color);

                return (pad_l, pad_u, img_padded);
            }
        }

        internal class Rectangle : ITransform
        {
            private readonly int mask_ratio;
            private readonly int color;

            internal Rectangle(int mask_ratio = 4, int color = 114)
            {
                this.mask_ratio = mask_ratio;
                this.color = color;
            }

            public Struct.LabelStruct Apply(Struct.LabelStruct label)
            {
                if (label.normalized)
                {
                    throw new ArgumentException("Label must be denormalized for LetterBox.");
                }
                Struct.LabelStruct transformedLabel = label.Clone();

                (int pad_l, int pad_u, torch.Tensor img) = RectangleImage(transformedLabel.img, label.resized_shape.w, label.resized_shape.h, label.rectangle_shape.w, label.rectangle_shape.h, color);

                transformedLabel.img = img;

                if (transformedLabel.mask is not null)
                {
                    (_, _, torch.Tensor mask) = RectangleImage(transformedLabel.mask, label.resized_shape.w / mask_ratio, label.resized_shape.h / mask_ratio, label.rectangle_shape.w / mask_ratio, label.rectangle_shape.h / mask_ratio, 0);
                    transformedLabel.mask = mask;
                }
                if (transformedLabel.bboxes is not null)
                {
                    if (transformedLabel.bbox_format == torchvision.ops.BoxFormats.xyxy)
                    {
                        transformedLabel.bboxes = transformedLabel.bboxes.add(new float[] { pad_l, pad_u, pad_l, pad_u });
                    }
                    else
                    {
                        transformedLabel.bboxes = transformedLabel.bboxes.add(new float[] { pad_l, pad_u, 0, 0 });
                    }
                }

                if (transformedLabel.keypoints is not null)
                {
                    transformedLabel.keypoints[torch.TensorIndex.Ellipsis, 0..2] = transformedLabel.keypoints[torch.TensorIndex.Ellipsis, 0..2].add(new float[] { pad_l, pad_u });
                }

                if (transformedLabel.obb_corners is not null)
                {
                    transformedLabel.obb_corners = transformedLabel.obb_corners.add(new float[] { pad_l, pad_u });
                }

                // transformedLabel.resized_shape = (this.resized_height, this.resized_width);

                return transformedLabel;
            }

            private (int pad_l, int pad_u, torch.Tensor rectangle_image) RectangleImage(torch.Tensor img, int resized_w, int resized_h, int rectangle_w, int rectangle_h, int color)
            {
                int imgH = (int)img.shape[1];
                int imgW = (int)img.shape[2];

                float ratio_w = resized_w / (float)imgW;
                float ratio_h = resized_h / (float)imgH;
                float ratio = Math.Min(ratio_w, ratio_h);

                int new_w = (int)(imgW * ratio);
                int new_h = (int)(imgH * ratio);

                int pad_l = (rectangle_w - new_w) / 2;
                int pad_r = rectangle_w - new_w - pad_l;
                int pad_u = (rectangle_h - new_h) / 2;
                int pad_d = rectangle_h - new_h - pad_u;

                torch.Tensor img_resized = torchvision.transforms.functional.resize(img, new_h, new_w);
                torch.Tensor img_padded = torchvision.transforms.functional.pad(img_resized, new long[] { pad_l, pad_u, pad_r, pad_d }, color);

                return (pad_l, pad_u, img_padded);
            }
        }


        internal class FlipLR : ITransform
        {
            private readonly float p;
            internal FlipLR(float p = 0.5f)
            {
                this.p = p;
            }
            public Struct.LabelStruct Apply(Struct.LabelStruct label)
            {
                if (torch.rand_float() > this.p)
                {
                    return label;
                }

                if (label.normalized)
                {
                    throw new ArgumentException("Label must be denormalized for FlipLR.");
                }
                Struct.LabelStruct transformedLabel = label.Clone();
                transformedLabel.img = transformedLabel.img.flip(-1L);
                if (transformedLabel.mask is not null)
                {
                    transformedLabel.mask = transformedLabel.mask.flip(-1L);
                }
                int w = transformedLabel.resized_shape.w;

                if (transformedLabel.bboxes is not null)
                {
                    if (transformedLabel.bbox_format != torchvision.ops.BoxFormats.xywh)
                    {
                        transformedLabel.bboxes[torch.TensorIndex.Ellipsis, 0] = w - transformedLabel.bboxes[torch.TensorIndex.Ellipsis, 0];
                        transformedLabel.bboxes[torch.TensorIndex.Ellipsis, 2] = w - transformedLabel.bboxes[torch.TensorIndex.Ellipsis, 2];
                    }
                    else
                    {
                        transformedLabel.bboxes[torch.TensorIndex.Ellipsis, 0] = w - transformedLabel.bboxes[torch.TensorIndex.Ellipsis, 0];
                    }
                }
                else
                {
                    transformedLabel.bboxes = torch.zeros(new long[] { 0, 4 });
                }

                if (transformedLabel.keypoints is not null)
                {
                    transformedLabel.keypoints[torch.TensorIndex.Ellipsis, 0] = w - transformedLabel.keypoints[torch.TensorIndex.Ellipsis, 0];
                }

                if (transformedLabel.obb_corners is not null)
                {
                    transformedLabel.obb_corners[torch.TensorIndex.Ellipsis, 0] = w - transformedLabel.obb_corners[torch.TensorIndex.Ellipsis, 0];
                }
                return transformedLabel;
            }

        }

        internal class FlipUD : ITransform
        {
            private readonly float p;
            internal FlipUD(float p = 0f)
            {
                this.p = p;
            }
            public Struct.LabelStruct Apply(Struct.LabelStruct label)
            {
                if (torch.rand_float() > this.p)
                {
                    return label;
                }

                if (label.normalized)
                {
                    throw new ArgumentException("Label must be denormalized for FlipUD.");
                }
                Struct.LabelStruct transformedLabel = label.Clone();
                transformedLabel.img = transformedLabel.img.flip(-2L);
                if (transformedLabel.mask is not null)
                {
                    transformedLabel.mask = transformedLabel.mask.flip(-2L);
                }
                int h = transformedLabel.resized_shape.h;

                if (transformedLabel.bbox_format != torchvision.ops.BoxFormats.xywh)
                {
                    transformedLabel.bboxes[torch.TensorIndex.Ellipsis, 1] = h - transformedLabel.bboxes[torch.TensorIndex.Ellipsis, 1];
                    transformedLabel.bboxes[torch.TensorIndex.Ellipsis, 3] = h - transformedLabel.bboxes[torch.TensorIndex.Ellipsis, 3];
                }
                else
                {
                    transformedLabel.bboxes[torch.TensorIndex.Ellipsis, 1] = h - transformedLabel.bboxes[torch.TensorIndex.Ellipsis, 1];
                }

                if (transformedLabel.keypoints is not null)
                {
                    transformedLabel.keypoints[torch.TensorIndex.Ellipsis, 1] = h - transformedLabel.keypoints[torch.TensorIndex.Ellipsis, 1];
                }

                if (transformedLabel.obb_corners is not null)
                {
                    transformedLabel.obb_corners[torch.TensorIndex.Ellipsis, 1] = h - transformedLabel.obb_corners[torch.TensorIndex.Ellipsis, 1];
                }
                return transformedLabel;

            }

        }

        internal class RandomHSV : ITransform
        {
            private readonly float hgain;
            private readonly float vgain;
            private readonly float sgain;

            internal RandomHSV(float hgain = 0.015f, float sgain = 0.7f, float vgain = 0.4f)
            {
                this.hgain = hgain;
                this.sgain = sgain;
                this.vgain = vgain;
            }

            public Struct.LabelStruct Apply(Struct.LabelStruct label)
            {
                Struct.LabelStruct transformedLabel = label.Clone();
                var transform = torchvision.transforms.ColorJitter(brightness: this.vgain, contrast: 0, hue: this.hgain, saturation: this.sgain);
                transformedLabel.img = transform.call(transformedLabel.img);
                return transformedLabel;
            }

        }
    }
}
