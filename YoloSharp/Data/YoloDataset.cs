using TorchSharp;
using Utils;
using YoloSharp.Types;

namespace Data
{
    internal class YoloDataset : BaseDataset
    {
        private string[] label_files;
        private TaskType taskType;
        private int mask_ratio;
        private readonly int nkpt;
        private readonly int ndim;
        private readonly int imgsz;
        private readonly float p_mosaic;
        private readonly int mosaic_count;
        private readonly float degress;
        private readonly float translate;
        private readonly float scale;
        private readonly float shear;
        private readonly float perspective;
        private readonly float p_flipLR;
        private readonly float p_flipUD;
        private readonly float hgain;
        private readonly float sgain;
        private readonly float vgain;
        private readonly bool isValDataset;
        private readonly string scanPath;
        private readonly bool useRectangle;

        public YoloDataset(Config config, bool isValDataset = false, bool useRectangle = false) : base(config: config, prefix: YoloSharp.Utils.Display.Color.BrightBlue)
        {
            this.taskType = this.config.TaskType;
            this.mask_ratio = config.MaskRatio;
            this.nkpt = config.KeyPoint_Num;
            this.ndim = config.KeyPoint_Dim;
            this.imgsz = config.ImageSize;
            this.p_mosaic = config.Mosaic;
            this.mosaic_count = config.MosaicCount;
            this.degress = config.Degrees;
            this.translate = config.Translate;
            this.scale = config.Scale;
            this.shear = config.Shear;
            this.perspective = config.Perspective;
            this.p_flipLR = config.FlipLR;
            this.p_flipUD = config.FlipUD;
            this.hgain = config.HSV_H;
            this.sgain = config.HSV_S;
            this.vgain = config.HSV_V;
            this.isValDataset = isValDataset;
            this.scanPath = this.isValDataset ? config.ValDataPath : config.TrainDataPath;
            base.img_path = Path.GetFullPath(Path.Combine(config.RootPath, this.scanPath));
            this.useRectangle = useRectangle;
            this.Initialize();
        }

        public override Augment.ITransform build_transforms()
        {
            Augment.Compose compose = new Augment.Compose();

            if (!isValDataset)
            {
                if (config.ImageProcessType == ImageProcessType.Mosiac)
                {
                    Augment.Mosaic mosaic = new Augment.Mosaic(this, null, this.imgsz, this.p_mosaic, this.mosaic_count);
                    compose.Add(mosaic);
                    Augment.RandomPerspective randomPerspective = new Augment.RandomPerspective(this.degress, this.translate, this.scale, this.shear, this.perspective);
                    compose.Add(randomPerspective);
                }
                else
                {
                    Augment.LetterBox letterBox = new Augment.LetterBox(this.imgsz, this.imgsz, this.mask_ratio);
                    compose.Add(letterBox);
                }

                if (this.p_flipLR > 0)
                {
                    Augment.FlipLR flipLR = new Augment.FlipLR(this.p_flipLR);
                    compose.Add(flipLR);
                }
                if (this.p_flipUD > 0)
                {
                    Augment.FlipUD flipUD = new Augment.FlipUD(this.p_flipUD);
                    compose.Add(flipUD);
                }
                Augment.RandomHSV randomHSV = new Augment.RandomHSV(this.hgain, this.sgain, this.vgain);
                compose.Add(randomHSV);
            }
            else
            {
                //Augment.LetterBox letterBoxes = new Augment.LetterBox(config.ImageSize, config.ImageSize, config.MaskRatio);
                //compose.Add(letterBoxes);

                Augment.Rectangle rectangle = new Augment.Rectangle(this.mask_ratio);
                compose.Add(rectangle);
            }

            return compose;
        }


        public override Dictionary<string, torch.Tensor> GetTensor(long index)
        {
            Data.Struct.LabelStruct label = this.labels[(int)index].Clone();
            label = transform.Apply(label);
            label.bboxes = torchvision.ops.box_convert(label.bboxes, label.bbox_format, torchvision.ops.BoxFormats.cxcywh);
            label.bbox_format = torchvision.ops.BoxFormats.cxcywh;
            torch.Tensor ourters = null;
            torch.Tensor cornerbox = null;
            if (label.obb_corners is not null)
            {
                //label.obb_corners = YoloSharp.Utils.Ops.sort_obb_corners_batch(label.obb_corners);
                long n = label.obb_corners.shape[0];
                cornerbox = torch.zeros(new long[] { n, 5 });
                ourters = torch.zeros(new long[] { n, 8 });
                for (int i = 0; i < label.obb_corners.shape[0]; i++)
                {
                    float[] xywhr = YoloSharp.Utils.Ops.xyxyxyxy2xywhr(label.obb_corners[i].data<float>().ToArray());
                    cornerbox[i] = xywhr;
                }
                cornerbox[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(0, 4)] = cornerbox[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(0, 4)] / this.imgsz;
            }

            label.Normalize();

            if (cornerbox is not null)
            {
                label.bboxes = cornerbox;
            }

            label.cls = label.cls?.view(-1, 1);
            label.img = label.img?.view(1, label.img.shape[0], label.img.shape[1], label.img.shape[2]);
            label.mask = label.mask?.view(1, label.mask.shape[0], label.mask.shape[1], label.mask.shape[2]);


            Dictionary<string, torch.Tensor> targets = new Dictionary<string, torch.Tensor>()
                {
                    { "cls", label.cls?.to(config.Device).MoveToOuterDisposeScope()},
                    { "bboxes", label.bboxes?.to(config.Device).MoveToOuterDisposeScope()},
                    { "images", label.img?.mul(1/255.0f).to(config.Device).MoveToOuterDisposeScope()},
                };
            if (taskType == TaskType.Segmentation)
            {
                targets["masks"] = label.mask?.to(config.Device).MoveToOuterDisposeScope();
            }
            if (taskType == TaskType.Pose)
            {
                targets["keypoints"] = label.keypoints?.to(config.Device).MoveToOuterDisposeScope();
            }
            return targets;
        }

        public override List<Struct.LabelStruct> get_labels()
        {
            using (torch.no_grad())
            {
                this.label_files = Utils.img2label_paths(this.im_files);
                int total = this.label_files.Length;
                List<Struct.LabelStruct> labels = new List<Struct.LabelStruct>();

                if (this.config.TaskType == TaskType.Pose && nkpt <= 0 || ndim > 3 || ndim < 2)
                {
                    throw new ArgumentOutOfRangeException("'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'");
                }
                string desc = $"{this.prefix}Scanning {(this.scanPath + "..."),-20}";
                using (Tqdm<string> pbar = new Tqdm<string>(im_files, total: total, barStyle: Tqdm.BarStyle.Classic, barColor: Tqdm.BarColor.White, barWidth: 10, showPartialChar: true))
                {
                    foreach (string im_file in pbar)
                    {
                        Struct.LabelStruct label = new Struct.LabelStruct();
                        label.im_file = im_file;
                        torch.Tensor img = torchvision.io.read_image(im_file, torchvision.io.ImageReadMode.RGB);

                        int org_h = (int)img.shape[1];
                        int org_w = (int)img.shape[2];
                        label.org_shape = (org_h, org_w);

                        float ratio_h = this.imgsz / (float)org_h;
                        float ratio_w = this.imgsz / (float)org_w;
                        float ratio = Math.Min(ratio_h, ratio_w);

                        int resized_h = (int)(ratio * org_h);
                        int resized_w = (int)(ratio * org_w);

                        label.resized_shape = (resized_h, resized_w);

                        label.img = torchvision.transforms.functional.resize(img, resized_h, resized_w);

                        int mask_w = (int)Math.Ceiling((float)resized_w / this.mask_ratio);
                        int mask_h = (int)Math.Ceiling((float)resized_h / this.mask_ratio);

                        string label_file = label_files[im_files.ToList().IndexOf(im_file)];
                        if (File.Exists(label_file))
                        {
                            string[] lines = File.ReadAllLines(label_file);
                            int rows = lines.Length;

                            float[] cls = new float[rows];
                            float[,] bboxes = new float[rows, 4];
                            float[] rotation = this.taskType == TaskType.Obb ? new float[rows] : null;
                            float[,,] keypoints = this.taskType == TaskType.Pose ? new float[rows, nkpt, ndim] : null;
                            float[,,] obb_corners = this.taskType == TaskType.Obb ? new float[rows, 4, 2] : null;
                            OpenCvSharp.Mat maskMat = this.taskType == TaskType.Segmentation ? new OpenCvSharp.Mat(mask_h, mask_w, OpenCvSharp.MatType.CV_8UC1, new OpenCvSharp.Scalar(0)) : null;

                            for (int row = 0; row < rows; row++)
                            {
                                string[] strs = lines[row].Split(' ');
                                cls[row] = float.Parse(strs[0]);
                                if (this.taskType == TaskType.Detection || taskType == TaskType.Pose)
                                {
                                    bboxes[row, 0] = float.Parse(strs[1]);
                                    bboxes[row, 1] = float.Parse(strs[2]);
                                    bboxes[row, 2] = float.Parse(strs[3]);
                                    bboxes[row, 3] = float.Parse(strs[4]);
                                }
                                if (taskType == TaskType.Obb)
                                {
                                    float x1 = float.Parse(strs[1]);
                                    float y1 = float.Parse(strs[2]);
                                    float x2 = float.Parse(strs[3]);
                                    float y2 = float.Parse(strs[4]);
                                    float x3 = float.Parse(strs[5]);
                                    float y3 = float.Parse(strs[6]);
                                    float x4 = float.Parse(strs[7]);
                                    float y4 = float.Parse(strs[8]);

                                    obb_corners[row, 0, 0] = x1;
                                    obb_corners[row, 0, 1] = y1;
                                    obb_corners[row, 1, 0] = x2;
                                    obb_corners[row, 1, 1] = y2;
                                    obb_corners[row, 2, 0] = x3;
                                    obb_corners[row, 2, 1] = y3;
                                    obb_corners[row, 3, 0] = x4;
                                    obb_corners[row, 3, 1] = y4;
                                    float x_min = new float[] { x1, x2, x3, x4 }.Min();
                                    float x_max = new float[] { x1, x2, x3, x4 }.Max();
                                    float y_min = new float[] { y1, y2, y3, y4 }.Min();
                                    float y_max = new float[] { y1, y2, y3, y4 }.Max();

                                    bboxes[row, 0] = (x_min + x_max) / 2;
                                    bboxes[row, 1] = (y_min + y_max) / 2;
                                    bboxes[row, 2] = x_max - x_min;
                                    bboxes[row, 3] = y_max - y_min;
                                }
                                if (taskType == TaskType.Segmentation)
                                {
                                    int point_count = strs.Length / 2;
                                    float max_x = float.MinValue, max_y = float.MinValue, min_x = float.MaxValue, min_y = float.MaxValue;
                                    int length = strs.Length / 2;
                                    OpenCvSharp.Point[] points = new OpenCvSharp.Point[length];
                                    for (int col = 0; col < length; col++)
                                    {
                                        float x = float.Parse(strs[2 * col + 1]);
                                        float y = float.Parse(strs[2 * col + 2]);

                                        max_x = Math.Max(max_x, x);
                                        max_y = Math.Max(max_y, y);
                                        min_x = Math.Min(min_x, x);
                                        min_y = Math.Min(min_y, y);

                                        points[col].X = (int)(x * resized_w / mask_ratio);
                                        points[col].Y = (int)(y * resized_h / mask_ratio);
                                    }
                                    using (OpenCvSharp.Mat eachMaskMat = GetMaskFromOutlinePoints(points, mask_h, mask_w))
                                    using (OpenCvSharp.Mat foreMat = new OpenCvSharp.Mat(mask_h, mask_w, OpenCvSharp.MatType.CV_8UC1, new OpenCvSharp.Scalar(row + 1f)))
                                    {
                                        foreMat.CopyTo(maskMat, eachMaskMat);
                                    }

                                    bboxes[row, 0] = (max_x + min_x) / 2;
                                    bboxes[row, 1] = (max_y + min_y) / 2;
                                    bboxes[row, 2] = max_x - min_x;
                                    bboxes[row, 3] = max_y - min_y;
                                }
                                if (taskType == TaskType.Pose)
                                {
                                    for (int col = 0; col < nkpt; col++)
                                    {
                                        keypoints[row, col, 0] = float.Parse(strs[ndim * col + 5]);
                                        keypoints[row, col, 1] = float.Parse(strs[ndim * col + 6]);
                                        if (ndim == 3)
                                        {
                                            keypoints[row, col, 2] = float.Parse(strs[ndim * col + 7]);
                                        }
                                    }
                                }

                            }

                            label.cls = torch.tensor(cls);
                            label.bboxes = torch.tensor(bboxes);

                            if (this.taskType == TaskType.Pose)
                            {
                                label.keypoints = torch.tensor(keypoints);
                            }
                            if (this.taskType == TaskType.Obb)
                            {
                                label.obb_corners = torch.tensor(obb_corners);
                            }

                            label.normalized = true;
                            label.bbox_format = torchvision.ops.BoxFormats.cxcywh;

                            if (this.taskType == TaskType.Segmentation)
                            {
                                label.mask = YoloSharp.Utils.Lib.GetTensorFromImage(maskMat, torchvision.io.ImageReadMode.GRAY);
                                maskMat?.Dispose();
                            }

                        }

                        else
                        {
                            label.cls = torch.empty(0);
                            label.bboxes = torch.empty(0, 4);
                            if (this.taskType == TaskType.Pose)
                            {
                                label.keypoints = torch.empty(0, nkpt, ndim);
                            }
                            if (this.taskType == TaskType.Obb)
                            {
                                label.obb_corners = torch.empty(0, 4, 2);
                            }
                            if (config.TaskType == TaskType.Segmentation && label.mask is null)
                            {
                                label.mask = torch.zeros(new long[] { 1, mask_h, mask_w });
                            }
                        }

                        label.mask_ratio = this.mask_ratio;
                        label.im_file = im_file;
                        label.DeNormalize();
                        labels.Add(label);
                        GC.Collect();
                        pbar.SetDescription(desc);
                    }
                }
                if (this.useRectangle || this.isValDataset)
                {
                    labels.Sort((a, b) =>
                    {
                        return ((float)a.resized_shape.h / a.resized_shape.w).CompareTo((float)b.resized_shape.h / b.resized_shape.w);
                    });

                    int bs = config.BatchSize;
                    for (int i = 0; i < Count; i++)
                    {
                        int batchStart = i / bs * bs;
                        var batchItems = labels.Skip(batchStart).Take(bs);

                        int max_w = batchItems.Max(a => a.resized_shape.w);
                        int max_h = batchItems.Max(a => a.resized_shape.h);

                        int w = ((int)Math.Ceiling(max_w / this.stride + this.pad)) * this.stride;
                        int h = ((int)Math.Ceiling(max_h / this.stride + this.pad)) * this.stride;

                        var tempLabel = labels[i].Clone();
                        tempLabel.rectangle_shape = (h, w);
                        labels[i] = tempLabel;
                    }

                }

                return labels;
            }
        }

        private OpenCvSharp.Mat GetMaskFromOutlinePoints(OpenCvSharp.Point[] points, int height, int width)
        {
            OpenCvSharp.Mat mask = OpenCvSharp.Mat.Zeros(height, width, OpenCvSharp.MatType.CV_8UC1);
            OpenCvSharp.Point[][] pts = new OpenCvSharp.Point[1][];
            pts[0] = points.Select(p => new OpenCvSharp.Point((int)p.X, (int)p.Y)).ToArray();
            OpenCvSharp.Cv2.FillPoly(mask, pts, OpenCvSharp.Scalar.White);
            return mask;
        }

        public override void CloseMosaic(bool closeMosaic)
        {
            Augment.Compose compose = new Augment.Compose();

            if (!isValDataset)
            {
                if (config.ImageProcessType == ImageProcessType.Mosiac)
                {
                    if (!closeMosaic)
                    {
                        Augment.Mosaic mosaic = new Augment.Mosaic(this, null, this.imgsz, this.p_mosaic, this.mosaic_count);
                        compose.Add(mosaic);
                        Augment.RandomPerspective randomPerspective = new Augment.RandomPerspective(this.degress, this.translate, this.scale, this.shear, this.perspective);
                        compose.Add(randomPerspective);
                    }
                    else
                    {
                        Augment.LetterBox letterBox = new Augment.LetterBox(this.imgsz, this.imgsz, this.mask_ratio);
                        compose.Add(letterBox);
                    }
                }
                else
                {
                    Augment.LetterBox letterBox = new Augment.LetterBox(this.imgsz, this.imgsz, this.mask_ratio);
                    compose.Add(letterBox);
                }


                if (this.p_flipLR > 0)
                {
                    Augment.FlipLR flipLR = new Augment.FlipLR(this.p_flipLR);
                    compose.Add(flipLR);
                }
                if (this.p_flipUD > 0)
                {
                    Augment.FlipUD flipUD = new Augment.FlipUD(this.p_flipUD);
                    compose.Add(flipUD);
                }
                Augment.RandomHSV randomHSV = new Augment.RandomHSV(this.hgain, this.sgain, this.vgain);
                compose.Add(randomHSV);
            }
            else
            {
                //Augment.LetterBox letterBoxes = new Augment.LetterBox(config.ImageSize, config.ImageSize, config.MaskRatio);
                //compose.Add(letterBoxes);

                Augment.Rectangle rectangle = new Augment.Rectangle(this.mask_ratio);
                compose.Add(rectangle);
            }

            this.transform = compose;
        }

    }
}
