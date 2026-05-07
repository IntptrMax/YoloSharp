using TorchSharp;

namespace Data
{
    internal abstract class BaseDataset : torch.utils.data.Dataset
    {
        protected string img_path;
        protected int imgsz;
        protected bool augment;
        protected bool single_cls;
        protected string prefix;
        protected float fraction;
        protected string[] im_files;
        public List<Data.Struct.LabelStruct> labels;
        protected int stride;
        protected float pad;
        protected Augment.ITransform transform;
        protected Config config;

        /// <summary>
        /// Initialize BaseDataset with given configuration and options.
        /// </summary>
        /// <param name="img_path">Path to the folder containing images or list of image paths.</param>
        /// <param name="imgsz">Image size for resizing.</param>
        /// <param name="cache">Cache images to RAM or disk during training.</param>
        /// <param name="augment">If True, data augmentation is applied.</param>
        /// <param name="hyp">Hyperparameters to apply data augmentation.</param>
        /// <param name="prefix">Prefix to print in log messages.</param>
        /// <param name="rect">If True, rectangular training is used.</param>
        /// <param name="batch_size">Size of batches.</param>
        /// <param name="stride">Stride used in the model.</param>
        /// <param name="pad">Padding value.</param>
        /// <param name="single_cls">If True, single class training is used.</param>
        /// <param name="classes">List of included classes.</param>
        /// <param name="fraction">Fraction of dataset to utilize.</param>
        /// <param name="channels">Number of channels in the images (1 for grayscale, 3 for color). Color images loaded with OpenCV are in BGR channel order.</param>
        public BaseDataset(Config config, bool cache = false, bool augment = true, string prefix = "",
            int stride = 32, float pad = 0.5f, float fraction = 1f, bool single_cls = false, List<int> classes = null, int channels = 3)
        {
            this.augment = augment;
            this.single_cls = single_cls;
            this.prefix = prefix;
            this.fraction = fraction;
            this.stride = stride;
            this.pad = pad;
            this.config = config;
            this.imgsz = config.ImageSize;

        }

        protected void Initialize()
        {
            this.im_files = this.get_img_files(this.img_path);
            this.labels = this.get_labels();
            this.transform = this.build_transforms();
        }

        public override long Count => im_files.Length;

        public override Dictionary<string, torch.Tensor> GetTensor(long index)
        {
            throw new Exception();
        }

        public string[] get_img_files(string img_path)
        {
            return get_img_files(new string[] { img_path });
        }

        public string[] get_img_files(string[] img_path)
        {
            List<string> f = new List<string>(); // image files
            try
            {
                foreach (string p in img_path)
                {
                    string path = p; // os-agnostic 

                    if (Directory.Exists(path)) // dir
                    {
                        f.AddRange(Directory.EnumerateFiles(path, "*.*", SearchOption.AllDirectories));
                    }
                    else if (File.Exists(path)) // file
                    {
                        string[] lines = File.ReadAllLines(path);
                        string parentDir = Path.GetDirectoryName(path)!;
                        foreach (string line in lines)
                        {
                            string trimmedLine = line.Trim();
                            if (string.IsNullOrEmpty(trimmedLine))
                            {
                                continue;
                            }
                            string fullPath;
                            if (trimmedLine.StartsWith("./"))
                            {
                                fullPath = Path.Combine(parentDir, trimmedLine.Substring(2));
                            }
                            else
                            {
                                fullPath = trimmedLine;
                            }
                            f.Add(fullPath);
                        }
                    }
                    else
                    {
                        throw new FileNotFoundException($"{this.prefix}{path} does not exist");
                    }
                }

                List<string> imFiles = f.Where(x => { string ext = Path.GetExtension(x).ToLower(); return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tif" || ext == ".tiff"; })
                    .Select(x => Path.GetFullPath(x).ToLower()).ToList();
                imFiles.Sort();

                if (imFiles.Count == 0)
                {
                    throw new FileNotFoundException($"{this.prefix}No images found in {string.Join(", ", imFiles)}");
                }

                // retain a fraction of the dataset
                if (this.fraction < 1.0)
                {
                    int countToKeep = (int)Math.Round(imFiles.Count * this.fraction);
                    imFiles = imFiles.Take(countToKeep).ToList();
                }

                return imFiles.ToArray();
            }

            catch
            {
                throw new FileNotFoundException($"{this.prefix}Error loading data from {img_path}\n");
            }

        }

        public abstract List<Data.Struct.LabelStruct> get_labels();

        public abstract Augment.ITransform build_transforms();

    }
}
