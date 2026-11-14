using TorchSharp;
using TorchSharp.Modules;
using YoloSharp.Modules;
using YoloSharp.Types;

namespace YoloSharp.Models
{
    internal partial class Yolo
    {
        internal class Yolov11 : Yolov8
        {
            protected override int[] outputIndexs => new int[] { 4, 6, 10, 13, 16, 19, 22 };

            internal Yolov11(int nc = 80, int reg_max = 16, int? kpt_num = null, int? kpt_dim = null, bool end2end = false, YoloSize yoloSize = YoloSize.n, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nc, reg_max: reg_max, kpt_dim: kpt_dim, kpt_num: kpt_num, end2end: end2end, yoloSize: yoloSize, device: device, dtype: dtype)
            {

            }

            internal override ModuleList<torch.nn.Module> BuildModel()
            {
                (float depth_multiple, float width_multiple, int max_channels, bool useC3k) = yoloSize switch
                {
                    YoloSize.n => (0.5f, 0.25f, 1024, false),
                    YoloSize.s => (0.5f, 0.5f, 1024, false),
                    YoloSize.m => (0.5f, 1.0f, 512, true),
                    YoloSize.l => (1.0f, 1.0f, 512, true),
                    YoloSize.x => (1.0f, 1.5f, 768, true),
                    _ => throw new ArgumentOutOfRangeException(nameof(yoloSize), yoloSize, null)
                };

                base.widths = new List<int> { 64, 128, 256, 512, 1024 }.Select(w => Math.Min((int)(w * width_multiple), max_channels)).ToArray();
                int depthSize = (int)(2 * depth_multiple);
                ch = new int[] { widths[2], widths[3], widths[4] };

                ModuleList<torch.nn.Module> mod = new ModuleList<torch.nn.Module>(
                    new Convs.Conv(3, widths[0], 3, 2, device: device, dtype: dtype),
                    new Convs.Conv(widths[0], widths[1], 3, 2, device: device, dtype: dtype),
                    new Block.C3k2(widths[1], widths[2], depthSize, useC3k, e: 0.25f, device: device, dtype: dtype),
                    new Convs.Conv(widths[2], widths[2], 3, 2, device: device, dtype: dtype),
                    new Block.C3k2(widths[2], widths[3], depthSize, useC3k, e: 0.25f, device: device, dtype: dtype),
                    new Convs.Conv(widths[3], widths[3], 3, 2, device: device, dtype: dtype),
                    new Block.C3k2(widths[3], widths[3], depthSize, c3k: true, device: device, dtype: dtype),
                    new Convs.Conv(widths[3], widths[4], 3, 2, device: device, dtype: dtype),
                    new Block.C3k2(widths[4], widths[4], depthSize, c3k: true, device: device, dtype: dtype),
                    new Block.SPPF(widths[4], widths[4], 5, device: device, dtype: dtype),
                    new Block.C2PSA(widths[4], widths[4], depthSize, device: device, dtype: dtype),

                    torch.nn.Upsample(scale_factor: new double[] { 2, 2 }, mode: torch.UpsampleMode.Nearest),
                    new Convs.Concat(),
                    new Block.C3k2(widths[4] + widths[3], widths[3], depthSize, useC3k, device: device, dtype: dtype),

                    torch.nn.Upsample(scale_factor: new double[] { 2, 2 }, mode: torch.UpsampleMode.Nearest),
                    new Convs.Concat(),
                    new Block.C3k2(widths[3] + widths[3], widths[2], depthSize, useC3k, device: device, dtype: dtype),

                    new Convs.Conv(widths[2], widths[2], 3, 2, device: device, dtype: dtype),
                    new Convs.Concat(),
                    new Block.C3k2(widths[3] + widths[2], widths[3], depthSize, useC3k, device: device, dtype: dtype),

                    new Convs.Conv(widths[3], widths[3], 3, 2, device: device, dtype: dtype),
                    new Convs.Concat(),
                    new Block.C3k2(widths[4] + widths[3], widths[4], depthSize, c3k: true, device: device, dtype: dtype),

                    new Head.Detect(nc, reg_max: reg_max, ch: ch, legacy: false, end2end: end2end, device: device, dtype: dtype)
                );
                return mod;
            }
        }
        internal class Yolov11Segment : Yolov11
        {

            internal Yolov11Segment(int nc = 80, int reg_max = 16, YoloSize yoloSize = YoloSize.n, bool end2end = false, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nc, reg_max: reg_max, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
            {

            }

            internal override ModuleList<torch.nn.Module> BuildModel()
            {
                var mod = base.BuildModel();
                mod.RemoveAt(mod.Count - 1); // remove Detect
                mod.Add(new Head.Segment(nc: nc, ch: ch, reg_max: reg_max, npr: ch[0], end2end: end2end, legacy: false, device: device, dtype: dtype));
                return mod;
            }

        }
        internal class Yolov11Obb : Yolov11
        {
            internal Yolov11Obb(int nc = 80, int reg_max = 16, YoloSize yoloSize = YoloSize.n, bool end2end = false, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, reg_max: reg_max, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
            {

            }

            internal override ModuleList<torch.nn.Module> BuildModel()
            {
                var mod = base.BuildModel();
                mod.RemoveAt(mod.Count - 1); // remove Detect
                mod.Add(new Head.Obb(nc, reg_max: reg_max, end2end: this.end2end, legacy: false, ch: ch, device: device, dtype: dtype));
                return mod;
            }
        }
        internal class Yolov11Pose : Yolov11
        {
            internal Yolov11Pose(int nc = 80, int reg_max = 16, int kpt_num = 17, int kpt_dim = 3, YoloSize yoloSize = YoloSize.n, bool end2end = false, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, reg_max: reg_max, kpt_num: kpt_num, kpt_dim: kpt_dim, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
            {

            }

            internal override ModuleList<torch.nn.Module> BuildModel()
            {
                var mod = base.BuildModel();
                mod.RemoveAt(mod.Count - 1); // remove Detect
                mod.Add(new Head.Pose(nc, reg_max: reg_max, keypoint_num: this.kpt_num.Value, keypoint_dim: this.kpt_dim.Value, legacy: false, end2end: this.end2end, ch: ch, device: device, dtype: dtype));
                return mod;
            }
        }
        internal class Yolov11Classify : Yolov11
        {
            internal Yolov11Classify(int nc = 80, YoloSize yoloSize = YoloSize.n, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, yoloSize: yoloSize, device: device, dtype: dtype)
            {

            }

            internal override ModuleList<torch.nn.Module> BuildModel()
            {
                var mod = base.BuildModel();
                for (int i = 0; i < 13; i++)
                {
                    mod.RemoveAt(mod.Count - 1); // remove Detect
                }
                mod.Add(new Head.Classify(base.widths[4], nc, device: device, dtype: dtype));
                return mod;
            }
        }
        internal class Yolov12Classify : Yolov11
        {
            internal Yolov12Classify(int nc = 80, YoloSize yoloSize = YoloSize.n, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, yoloSize: yoloSize, device: device, dtype: dtype)
            {

            }

            internal override ModuleList<torch.nn.Module> BuildModel()
            {
                var mod = base.BuildModel();
                for (int i = 0; i < 13; i++)
                {
                    mod.RemoveAt(mod.Count - 1); // remove Detect
                }
                mod.Add(new Head.Classify(base.widths[4], nc, device: device, dtype: dtype));
                return mod;
            }
        }
    }
}
