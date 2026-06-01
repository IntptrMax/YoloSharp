using TorchSharp;
using YoloSharp.Types;

namespace Data
{
    internal class ClassificationDataset : BaseDataset
    {
        private static readonly double[] DEFAULT_MEAN = new double[] { 0.0, 0.0, 0.0 };
        private static readonly double[] DEFAULT_STD = new double[] { 1.0, 1.0, 1.0 };
        private readonly int imgsz;
        private readonly float p_flipLR;
        private readonly float p_flipUD;
        private readonly float hgain;
        private readonly float sgain;
        private readonly float vgain;
        private readonly bool isValDataset;
        private readonly string scanPath;
        private readonly float ratioMin;
        private readonly float ratioMax;
        private readonly float scaleMin;
        private readonly float scaleMax;
        private readonly float erasing;
        private readonly AutoAugmentType autoAugmentType;

        internal ClassificationDataset(Config config, bool isValDataset = false) : base(config: config, prefix: YoloSharp.Utils.Display.Color.BrightBlue)
        {
            this.imgsz = config.ImageSize;
            this.p_flipLR = config.FlipLR;
            this.p_flipUD = config.FlipUD;
            this.hgain = config.HSV_H;
            this.sgain = config.HSV_S;
            this.vgain = config.HSV_V;
            this.isValDataset = isValDataset;
            this.scanPath = this.isValDataset ? config.ValDataPath : config.TrainDataPath;
            base.img_path = Path.GetFullPath(Path.Combine(config.RootPath, this.scanPath));
            this.scaleMin = config.ClassifyScaleMin;
            this.scaleMax = config.ClassifyScaleMax;
            this.ratioMin = config.ClassifyRatioMin;
            this.ratioMax = config.ClassifyRatioMax;
            this.erasing = config.Erasing;
            this.autoAugmentType = config.Auto_Augment;
            this.Initialize();
        }

        public override List<Struct.LabelStruct> get_labels()
        {
            using (torch.no_grad())
            using (torch.NewDisposeScope())
            {
                List<string> classsNames = this.im_files.Select(a => { return Directory.GetParent(a).Name; }).Distinct().ToList();
                classsNames.Sort();
                List<Struct.LabelStruct> labels = new List<Struct.LabelStruct>();
                foreach (string imgFileName in this.im_files)
                {
                    string labelName = Directory.GetParent(imgFileName).Name;
                    int cls = classsNames.IndexOf(labelName);
                    Struct.LabelStruct label = new Struct.LabelStruct()
                    {
                        im_file = imgFileName,
                        cls = torch.tensor(cls).MoveToOuterDisposeScope(),
                        img = torchvision.io.read_image(imgFileName, torchvision.io.ImageReadMode.RGB).MoveToOuterDisposeScope(),
                    };
                    labels.Add(label);
                }

                return labels;
            }
        }

        public override Dictionary<string, torch.Tensor> GetTensor(long index)
        {
            using (torch.no_grad())
            using (torch.NewDisposeScope())
            {
                Data.Struct.LabelStruct label = this.labels[(int)index].Clone();
                label = transform.Apply(label);
                torch.Tensor img = label.img;
                torch.Tensor cls = label.cls;
                img = img.view(1, label.img.shape[0], label.img.shape[1], label.img.shape[2]).mul(1 / 255.0f);
                cls = cls.view(-1, 1);
                Dictionary<string, torch.Tensor> targets = new Dictionary<string, torch.Tensor>()
                {
                    { "cls", cls.to(torch.ScalarType.Int64,config.Device).MoveToOuterDisposeScope()},
                    { "images", img.to(config.Device).MoveToOuterDisposeScope()},
                };
                return targets;
            }
        }

        public override Augment.ITransform build_transforms()
        {
            List<torchvision.ITransform> transforms = new List<torchvision.ITransform>();
            if (!this.isValDataset)
            {
                transforms.Add(torchvision.transforms.RandomResizedCrop(this.imgsz, scaleMin: this.scaleMin, scaleMax: this.scaleMax, ratioMin: this.ratioMin, ratioMax: this.ratioMax));
                if (this.p_flipLR > 0)
                {
                    transforms.Add(torchvision.transforms.RandomHorizontalFlip(this.p_flipLR));
                }
                if (this.p_flipUD > 0)
                {
                    transforms.Add(torchvision.transforms.RandomVerticalFlip(this.p_flipUD));
                }

                if (this.autoAugmentType == AutoAugmentType.AutoAugment)
                {
                    transforms.Add(torchvision.transforms.AutoAugment());
                }
                else if (this.autoAugmentType == AutoAugmentType.RandAugment)
                {
                    transforms.Add(torchvision.transforms.RandAugment());
                }
                else if (this.autoAugmentType == AutoAugmentType.AugMix)
                {
                    transforms.Add(torchvision.transforms.AugMix());
                }

                if (this.erasing > 0)
                {
                    transforms.Add(new RandomErasing(this.erasing));
                }
                transforms.Add(torchvision.transforms.ColorJitter(this.vgain, this.vgain, this.sgain, this.hgain));
            }
            else
            {
                transforms.Add(torchvision.transforms.Resize(this.imgsz));
                transforms.Add(torchvision.transforms.CenterCrop(this.imgsz));
            }
            ClassificationTransforms classificationTransforms = new ClassificationTransforms(transforms);
            return classificationTransforms;
        }

        private class ClassificationTransforms : Augment.ITransform
        {
            private readonly List<torchvision.ITransform> transforms;
            internal ClassificationTransforms(List<torchvision.ITransform> transforms)
            {
                this.transforms = transforms;
            }

            public Struct.LabelStruct Apply(Struct.LabelStruct label)
            {
                Struct.LabelStruct newLabel = label.Clone();
                foreach (torchvision.ITransform transform in this.transforms)
                {
                    torch.Tensor t = transform.call(newLabel.img);
                    for (int i = 0; i < 10; i++)
                    {
                        if (t.IsInvalid)
                        {
                            t = transform.call(newLabel.img);
                        }
                        else
                        {
                            break;
                        }
                    }

                    newLabel.img = t;
                }

                return newLabel;
            }
        }

        private class RandomErasing : torchvision.ITransform
        {
            private readonly float p;
            private readonly float scaleMin;
            private readonly float scaleMax;
            private readonly float ratioMin;
            private readonly float ratioMax;
            private readonly float[]? value;
            private readonly bool inplace;
            public RandomErasing(float p = 0.5f, float scaleMin = 0.02f, float scaleMax = 0.33f, float ratioMin = 0.3f, float ratioMax = 3.3f, float[]? value = null, bool inplace = false)
            {
                this.p = p;
                this.scaleMin = scaleMin;
                this.scaleMax = scaleMax;
                this.ratioMin = ratioMin;
                this.ratioMax = ratioMax;
                this.value = value;
                this.inplace = inplace;
            }

            private (int i, int j, int h, int w, torch.Tensor v) get_params(torch.Tensor img, float scaleMin, float scaleMax, float ratioMin, float ratioMax, float[]? value = null)
            {
                int img_c = (int)img.shape[img.shape.Length - 3];
                int img_h = (int)img.shape[img.shape.Length - 2];
                int img_w = (int)img.shape[img.shape.Length - 1];
                int area = img_h * img_w;
                torch.Tensor log_ratio = torch.log(torch.tensor(new float[] { this.ratioMin, this.ratioMax }));
                float log_ratio_min = (float)Math.Log(this.ratioMin);
                float log_ratio_max = (float)Math.Log(this.ratioMax);
                for (int loop = 0; loop < 10; loop++)
                {
                    float erase_area = area * torch.empty(1).uniform_(scaleMin, scaleMax).ToSingle();
                    float aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio_min, log_ratio_max)).ToSingle();
                    int h = (int)(Math.Round(Math.Sqrt(erase_area * aspect_ratio)));
                    int w = (int)(Math.Round(Math.Sqrt(erase_area / aspect_ratio)));
                    if (!((h < img_h) && (w < img_w)))
                    {
                        continue;
                    }

                    torch.Tensor v = value is null ? torch.empty(new long[] { img_c, h, w }, dtype: torch.float32).normal_() : torch.tensor(value)[torch.TensorIndex.Colon, torch.TensorIndex.None, torch.TensorIndex.None];

                    int i = torch.randint_int(img_h - h + 1);
                    int j = torch.randint_int(img_w - w + 1);
                    return (i, j, h, w, v);
                }
                // Return original image
                return (0, 0, img_h, img_w, img);
            }

            public torch.Tensor call(torch.Tensor img)
            {
                if (torch.randn_float() > this.p)
                {
                    return img;
                }

                var (x, y, h, w, v) = get_params(img, scaleMin: this.scaleMin, scaleMax: this.scaleMax, ratioMin: this.ratioMin, ratioMax: this.ratioMax, value: value);
                return torchvision.transforms.functional.erase(img, x, y, h, w, v, this.inplace);
            }
        }

        public override void CloseMosaic(bool closeMosaic)
        {

        }

    }




}
