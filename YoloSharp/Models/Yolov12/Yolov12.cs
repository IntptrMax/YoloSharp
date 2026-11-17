using TorchSharp;
using TorchSharp.Modules;
using YoloSharp.Modules;
using YoloSharp.Types;

namespace YoloSharp.Models
{
    internal partial class Yolo
    {
        internal class Yolov12 : Yolov8
        {
            protected override int[] outputIndexs => new int[] { 4, 6, 8, 11, 14, 17, 20 };
            internal Yolov12(int nc = 80, int reg_max = 16, int? kpt_num = null, int? kpt_dim = null, YoloSize yoloSize = YoloSize.n, bool end2end = false, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nc, reg_max: reg_max, kpt_dim: kpt_dim, kpt_num: kpt_num, end2end: end2end, yoloSize: yoloSize, device: device, dtype: dtype)
            {

            }

            internal override ModuleList<torch.nn.Module> BuildModel()
            {
                (float depth_multiple, float width_multiple, int max_channels, bool useC3k, int n_nultiple, bool useResidual, float mlp_ratio) = yoloSize switch
                {
                    YoloSize.n => (0.5f, 0.25f, 1024, false, 1, false, 2.0f),
                    YoloSize.s => (0.5f, 0.5f, 1024, false, 1, false, 2.0f),
                    YoloSize.m => (0.5f, 1.0f, 512, true, 1, false, 2.0f),
                    YoloSize.l => (1.0f, 1.0f, 512, true, 2, true, 1.2f),
                    YoloSize.x => (1.0f, 1.5f, 768, true, 2, true, 1.2f),
                    _ => throw new ArgumentOutOfRangeException(nameof(yoloSize), yoloSize, null)
                };

                base.widths = new List<int> { 64, 128, 256, 512, 1024 }.Select(w => Math.Min((int)(w * width_multiple), max_channels)).ToArray();
                int depthSize = (int)(2 * depth_multiple);
                ch = new int[] { widths[2], widths[3], widths[4] };

                ModuleList<torch.nn.Module> mod = new ModuleList<torch.nn.Module>(
                    new Convs.Conv(3, widths[0], 3, 2, device: device, dtype: dtype),                                                                     // 0-P1/2
                    new Convs.Conv(widths[0], widths[1], 3, 2, device: device, dtype: dtype),                                                          // 1-P2/4
                    new Block.C3k2(widths[1], widths[2], depthSize, useC3k, e: 0.25f, device: device, dtype: dtype),
                    new Convs.Conv(widths[2], widths[2], 3, 2, device: device, dtype: dtype),                                                         // 3-P3/8
                    new Block.C3k2(widths[2], widths[3], depthSize, useC3k, e: 0.25f, device: device, dtype: dtype),
                    new Convs.Conv(widths[3], widths[3], 3, 2, device: device, dtype: dtype),                                                         // 5-P4/16
                    new Block.A2C2f(widths[3], widths[3], n: 2 * n_nultiple, a2: true, area: 4, useResidual, mlp_ratio, device: device, dtype: dtype),
                    new Convs.Conv(widths[3], widths[4], 3, 2, device: device, dtype: dtype),                                                        // 7-P5/32
                    new Block.A2C2f(widths[4], widths[4], n: 2 * n_nultiple, a2: true, area: 1, useResidual, mlp_ratio, device: device, dtype: dtype),

                    torch.nn.Upsample(scale_factor: new double[] { 2, 2 }, mode: torch.UpsampleMode.Nearest),
                    new Convs.Concat(),                                                                                       // cat backbone P4
                    new Block.A2C2f(widths[4] + widths[3], widths[3], n: n_nultiple, a2: false, area: -1, useResidual, mlp_ratio, device: device, dtype: dtype),                                   // 11

                    torch.nn.Upsample(scale_factor: new double[] { 2, 2 }, mode: torch.UpsampleMode.Nearest),
                    new Convs.Concat(),                                                                                       // cat backbone P3
                    new Block.A2C2f(widths[3] + widths[3], widths[2], n: n_nultiple, a2: false, area: -1, useResidual, mlp_ratio, device: device, dtype: dtype),                                    // 14 (P3/8-small)

                    new Convs.Conv(widths[2], widths[2], 3, 2, device: device, dtype: dtype),
                    new Convs.Concat(),                                                                                       // cat head P4
                    new Block.A2C2f(widths[3] + widths[2], widths[3], n: n_nultiple, a2: false, area: -1, useResidual, mlp_ratio, device: device, dtype: dtype),                                        // 17 (P4/16-medium)

                    new Convs.Conv(widths[3], widths[3], 3, 2, device: device, dtype: dtype),
                    new Convs.Concat(),                                                                                       // cat head P5
                    new Block.C3k2(widths[4] + widths[3], widths[4], depthSize, c3k: true, device: device, dtype: dtype),                       // 20 (P5/32-large)

                    new Head.Detect(nc, reg_max: this.reg_max, ch: this.ch, legacy: false, end2end: this.end2end, device: this.device, dtype: this.dtype)                                                                       // Detect(P3, P4, P5)
                    );
                return mod;
            }
        }
        internal class Yolov12Segment : Yolov12
        {

            internal Yolov12Segment(int nc = 80, int reg_max = 16, YoloSize yoloSize = YoloSize.n, bool end2end = false, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nc, reg_max: reg_max, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
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
        internal class Yolov12Obb : Yolov12
        {
            internal Yolov12Obb(int nc = 80, int reg_max = 16, YoloSize yoloSize = YoloSize.n, bool end2end = false, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, reg_max: reg_max, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
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
        internal class Yolov12Pose : Yolov12
        {
            internal Yolov12Pose(int nc = 80, int reg_max = 16, int kpt_num = 17, int kpt_dim = 3, YoloSize yoloSize = YoloSize.n, bool end2end = false, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, reg_max: reg_max, kpt_num: kpt_num, kpt_dim: kpt_dim, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
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
    }
}
