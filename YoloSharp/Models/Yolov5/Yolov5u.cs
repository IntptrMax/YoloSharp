using TorchSharp;
using TorchSharp.Modules;
using YoloSharp.Modules;
using YoloSharp.Types;

namespace YoloSharp.Models
{
    internal partial class Yolo
    {
        internal class Yolov5u : Yolov8
        {
            protected override int[] outputIndexs => new int[] { 4, 6, 10, 14, 17, 20, 23 };

            internal Yolov5u(int nc = 80, int reg_max = 16, int? kpt_num = null, int? kpt_dim = null, YoloSize yoloSize = YoloSize.n, bool end2end = false, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nc, reg_max: reg_max, kpt_dim: kpt_dim, kpt_num: kpt_num, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
            {

            }
            internal override ModuleList<torch.nn.Module> BuildModel()
            {
                (float depth_multiple, float width_multiple) = yoloSize switch
                {
                    YoloSize.n => (0.34f, 0.25f),
                    YoloSize.s => (0.34f, 0.5f),
                    YoloSize.m => (0.67f, 0.75f),
                    YoloSize.l => (1.0f, 1.0f),
                    YoloSize.x => (1.34f, 1.25f),
                    _ => throw new ArgumentOutOfRangeException(nameof(yoloSize), yoloSize, null)
                };

                widths = new List<int> { 64, 128, 256, 512, 1024 }.Select(w => (int)(w * width_multiple)).ToArray();
                int[] depths = new List<int> { 3, 6, 9 }.Select(d => (int)(d * depth_multiple)).ToArray();
                ch = new int[] { widths[2], widths[3], widths[4] };

                ModuleList<torch.nn.Module> mod = new ModuleList<torch.nn.Module>(
                    // backbone:
                    new Convs.Conv(3, widths[0], 6, 2, 2, device: device, dtype: dtype),                                           // 0-P1/2
                    new Convs.Conv(widths[0], widths[1], 3, 2, device: device, dtype: dtype),                    // 1-P2/4
                    new Block.C3(widths[1], widths[1], depths[0], device: device, dtype: dtype),
                    new Convs.Conv(widths[1], widths[2], 3, 2, device: device, dtype: dtype),                   // 3-P3/8
                    new Block.C3(widths[2], widths[2], depths[1], device: device, dtype: dtype),
                    new Convs.Conv(widths[2], widths[3], 3, 2, device: device, dtype: dtype),                   // 5-P4/16
                    new Block.C3(widths[3], widths[3], depths[2], device: device, dtype: dtype),
                    new Convs.Conv(widths[3], widths[4], 3, 2, device: device, dtype: dtype),                  // 7-P5/32
                    new Block.C3(widths[4], widths[4], depths[0], device: device, dtype: dtype),
                    new Block.SPPF(widths[4], widths[4], 5, device: device, dtype: dtype),

                    // head:
                    new Convs.Conv(widths[4], widths[3], 1, 1, device: device, dtype: dtype),
                    torch.nn.Upsample(scale_factor: new double[] { 2, 2 }, mode: torch.UpsampleMode.Nearest),
                    new Convs.Concat(),                                                                               // cat backbone P4
                    new Block.C3(widths[4], widths[3], depths[0], false, device: device, dtype: dtype),    // 13

                    new Convs.Conv(widths[3], widths[2], 1, 1, device: device, dtype: dtype),
                    torch.nn.Upsample(scale_factor: new double[] { 2, 2 }, mode: torch.UpsampleMode.Nearest),
                    new Convs.Concat(),                                                                               // cat backbone P3
                    new Block.C3(widths[3], widths[2], depths[0], false, device: device, dtype: dtype),      // 17 (P3/8-small)

                    new Convs.Conv(widths[2], widths[2], 3, 2, device: device, dtype: dtype),
                    new Convs.Concat(),                                                                               // cat head P4
                    new Block.C3(widths[3], widths[3], depths[0], false, device: device, dtype: dtype),      // 20 (P4/16-medium)

                    new Convs.Conv(widths[3], widths[3], 3, 2, device: device, dtype: dtype),
                    new Convs.Concat(),                                                                               // cat head P5
                    new Block.C3(widths[4], widths[4], depths[0], false, device: device, dtype: dtype),     // 23 (P5/32-large)

                    new Head.Detect(nc, ch: this.ch, reg_max: this.reg_max, end2end: this.end2end, device: this.device, dtype: this.dtype)                                                               // Detect(P3, P4, P5)
                );
                return mod;

            }
        }
        internal class Yolov5uSegment : Yolov5u
        {
            // [Fix 2026-11-13] removed unused `model` field: it hid the base class field (CS0108) and was
            // never used (CS0169) — BuildModel() override already swaps the head. / 删除未使用且遮蔽基类的字段

            internal Yolov5uSegment(int nc = 80, int reg_max = 16, YoloSize yoloSize = YoloSize.n, bool end2end = false, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nc, reg_max: reg_max, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
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
        internal class Yolov5uObb : Yolov5u
        {
            internal Yolov5uObb(int nc = 80, int reg_max = 16, YoloSize yoloSize = YoloSize.n, bool end2end = false, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, reg_max: reg_max, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
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
        internal class Yolov5uPose : Yolov5u
        {
            internal Yolov5uPose(int nc = 80, int reg_max = 16, int kpt_num = 17, int kpt_dim = 3, YoloSize yoloSize = YoloSize.n, bool end2end = false, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, reg_max: reg_max, kpt_num: kpt_num, kpt_dim: kpt_dim, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
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
        internal class Yolov5uClassify : Yolov5u
        {
            internal Yolov5uClassify(int nc = 80, YoloSize yoloSize = YoloSize.n, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, yoloSize: yoloSize, device: device, dtype: dtype)
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
