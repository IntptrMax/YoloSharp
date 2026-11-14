using TorchSharp;
using TorchSharp.Modules;
using YoloSharp.Modules;
using YoloSharp.Types;

namespace YoloSharp.Models
{
    internal partial class Yolo
    {
        public class Yolov8 : torch.nn.Module<torch.Tensor, (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds)?>
        {
            protected ModuleList<torch.nn.Module> model;
            protected virtual int[] outputIndexs => new int[] { 4, 6, 9, 12, 15, 18, 21 };
            protected virtual int[] concatIndex => new int[] { 1, 0, 3, 2 };

            protected int reg_max;
            protected int[] ch;
            protected int[] widths;
            protected bool end2end;
            protected int? kpt_num;
            protected int? kpt_dim;
            protected YoloSize yoloSize;
            protected torch.Device? device;
            protected torch.ScalarType? dtype;
            protected int nc;

            public Yolov8(int nc = 80, int reg_max = 16, int? kpt_num = null, int? kpt_dim = null, YoloSize yoloSize = YoloSize.n, bool end2end = false, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(Yolov8))
            {
                this.kpt_num = kpt_num;
                this.kpt_dim = kpt_dim;
                this.end2end = end2end;
                this.reg_max = reg_max;
                this.device = device;
                this.dtype = dtype;
                this.yoloSize = yoloSize;
                this.nc = nc;
                model = BuildModel();
                RegisterComponents();
            }

            internal virtual ModuleList<torch.nn.Module> BuildModel()
            {
                var (depth_multiple, width_multiple, max_channels) = this.yoloSize switch
                {
                    YoloSize.n => (0.34f, 0.25f, 1024),
                    YoloSize.s => (0.34f, 0.5f, 1024),
                    YoloSize.m => (0.67f, 0.75f, 576),
                    YoloSize.l => (1.0f, 1.0f, 512),
                    YoloSize.x => (1.0f, 1.25f, 640),
                    _ => throw new ArgumentOutOfRangeException(nameof(yoloSize), yoloSize, null)
                };

                widths = new List<int> { 64, 128, 256, 512, 1024 }.Select(w => Math.Min((int)(w * width_multiple), max_channels)).ToArray();
                int[] depths = new List<int> { 3, 6, 9 }.Select(d => (int)(d * depth_multiple)).ToArray();
                ch = new int[] { widths[2], widths[3], widths[4] };
                ModuleList<torch.nn.Module> mod = new ModuleList<torch.nn.Module>(
                    // backbone:
                    new Convs.Conv(3, widths[0], 3, 2, device: device, dtype: dtype),
                    new Convs.Conv(widths[0], widths[1], 3, 2, device: device, dtype: dtype),
                    new Block.C2f(widths[1], widths[1], depths[0], true, device: device, dtype: dtype),
                    new Convs.Conv(widths[1], widths[2], 3, 2, device: device, dtype: dtype),
                    new Block.C2f(widths[2], widths[2], depths[1], true, device: device, dtype: dtype),
                    new Convs.Conv(widths[2], widths[3], 3, 2, device: device, dtype: dtype),
                    new Block.C2f(widths[3], widths[3], depths[1], true, device: device, dtype: dtype),
                    new Convs.Conv(widths[3], widths[4], 3, 2, device: device, dtype: dtype),
                    new Block.C2f(widths[4], widths[4], depths[0], true, device: device, dtype: dtype),
                    new Block.SPPF(widths[4], widths[4], 5, device: device, dtype: dtype),

                    // head:
                    torch.nn.Upsample(scale_factor: new double[] { 2, 2 }, mode: torch.UpsampleMode.Nearest),
                    new Convs.Concat(),
                    new Block.C2f(widths[3] + widths[4], widths[3], depths[0], device: device, dtype: dtype),

                    torch.nn.Upsample(scale_factor: new double[] { 2, 2 }, mode: torch.UpsampleMode.Nearest),
                    new Convs.Concat(),
                    new Block.C2f(widths[2] + widths[3], widths[2], depths[0], device: device, dtype: dtype),

                    new Convs.Conv(widths[2], widths[2], 3, 2, device: device, dtype: dtype),
                    new Convs.Concat(),
                    new Block.C2f(widths[2] + widths[3], widths[3], depths[0], device: device, dtype: dtype),

                    new Convs.Conv(widths[3], widths[3], 3, 2, device: device, dtype: dtype),
                    new Convs.Concat(),
                    new Block.C2f(widths[4] + widths[3], widths[4], depths[0], device: device, dtype: dtype),

                    new Head.Detect(nc, reg_max: this.reg_max, end2end: this.end2end, ch: ch, device: device, dtype: dtype)
                );
                return mod;
            }


            public override (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds)? forward(torch.Tensor x)
            {
                //using (NewDisposeScope())
                {
                    (Dictionary<string, torch.Tensor> inference, Dictionary<string, object>)? result = null;
                    List<torch.Tensor> outputs = new List<torch.Tensor>();
                    int catCount = 0;
                    for (int i = 0; i < model.Count; i++)
                    {
                        switch (model[i])
                        {
                            case torch.nn.Module<torch.Tensor, torch.Tensor> mod:
                                x = mod.forward(x);
                                break;
                            case Convs.Concat cat:
                                x = cat.forward(new torch.Tensor[] { x, outputs[concatIndex[catCount]] });
                                catCount++;
                                break;
                            case Head.Detect detect:
                                {
                                    result = detect.forward(new torch.Tensor[] { outputs[outputs.Count - 3], outputs[outputs.Count - 2], outputs[outputs.Count - 1] });
                                    break;
                                }
                            case Head.Classify classify:
                                {
                                    result = classify.forward(new torch.Tensor[] { x });
                                    break;
                                }

                            default:
                                {
                                    throw new Exception();
                                }
                        }

                        if (outputIndexs.Contains(i))
                        {
                            outputs.Add(x);
                        }
                    }
                    return result;
                }
            }
        }
        internal class Yolov8Segment : Yolov8
        {
            internal Yolov8Segment(int nc = 80, int reg_max = 16, YoloSize yoloSize = YoloSize.n, bool end2end = false, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nc, reg_max: reg_max, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
            {

            }

            internal override ModuleList<torch.nn.Module> BuildModel()
            {
                var mod = base.BuildModel();
                mod.RemoveAt(mod.Count - 1); // remove Detect
                mod.Add(new Head.Segment(nc: nc, ch: ch, reg_max: reg_max, npr: ch[0], end2end: end2end, device: device, dtype: dtype));
                return mod;
            }

        }
        internal class Yolov8Obb : Yolov8
        {
            internal Yolov8Obb(int nc = 80, int reg_max = 16, YoloSize yoloSize = YoloSize.n, bool end2end = false, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, reg_max: reg_max, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
            {

            }

            internal override ModuleList<torch.nn.Module> BuildModel()
            {
                var mod = base.BuildModel();
                mod.RemoveAt(mod.Count - 1); // remove Detect
                mod.Add(new Head.Obb(nc, reg_max: this.reg_max, end2end: this.end2end, ch: ch, device: device, dtype: dtype));
                return mod;
            }
        }
        internal class Yolov8Pose : Yolov8
        {
            internal Yolov8Pose(int nc = 80, int reg_max = 16, int kpt_num = 17, int kpt_dim = 3, YoloSize yoloSize = YoloSize.n, bool end2end = false, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, reg_max: reg_max, kpt_num: kpt_num, kpt_dim: kpt_dim, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
            {

            }

            internal override ModuleList<torch.nn.Module> BuildModel()
            {
                var mod = base.BuildModel();
                mod.RemoveAt(mod.Count - 1); // remove Detect
                mod.Add(new Head.Pose(nc, reg_max: reg_max, keypoint_num: this.kpt_num.Value, keypoint_dim: this.kpt_dim.Value, end2end: this.end2end, ch: ch, device: device, dtype: dtype));
                return mod;
            }
        }
        internal class Yolov8Classify : Yolov8
        {
            internal Yolov8Classify(int nc = 80, YoloSize yoloSize = YoloSize.n, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, yoloSize: yoloSize, device: device, dtype: dtype)
            {

            }

            internal override ModuleList<torch.nn.Module> BuildModel()
            {
                var mod = base.BuildModel();
                for (int i = 0; i < 14; i++)
                {
                    mod.RemoveAt(mod.Count - 1); // remove Detect
                }
                mod.Add(new Head.Classify(base.widths[4], nc, device: device, dtype: dtype));
                return mod;
            }
        }
    }
}
