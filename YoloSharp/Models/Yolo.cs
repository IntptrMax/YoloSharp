using TorchSharp;
using TorchSharp.Modules;
using YoloSharp.Modules;
using YoloSharp.Types;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static YoloSharp.Modules.Modules;

namespace YoloSharp.Models
{
	internal class Yolo
	{
		public class Yolov8 : Module<Tensor, (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds)?>
		{
			private ModuleList<Module> model;
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

			public Yolov8(int nc = 80, int reg_max = 16, int? kpt_num = null, int? kpt_dim = null, YoloSize yoloSize = YoloSize.n, bool end2end = false, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(Yolov8))
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

			internal virtual ModuleList<Module> BuildModel()
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
				ModuleList<Module> mod = ModuleList<Module>(
					// backbone:
					new Convs.Conv(3, widths[0], 3, 2, device: device, dtype: dtype),
					new Convs.Conv(widths[0], widths[1], 3, 2, device: device, dtype: dtype),
					new C2f(widths[1], widths[1], depths[0], true, device: device, dtype: dtype),
					new Convs.Conv(widths[1], widths[2], 3, 2, device: device, dtype: dtype),
					new C2f(widths[2], widths[2], depths[1], true, device: device, dtype: dtype),
					new Convs.Conv(widths[2], widths[3], 3, 2, device: device, dtype: dtype),
					new C2f(widths[3], widths[3], depths[1], true, device: device, dtype: dtype),
					new Convs.Conv(widths[3], widths[4], 3, 2, device: device, dtype: dtype),
					new C2f(widths[4], widths[4], depths[0], true, device: device, dtype: dtype),
					new Block.SPPF(widths[4], widths[4], 5, device: device, dtype: dtype),

					// head:
					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),
					new C2f(widths[3] + widths[4], widths[3], depths[0], device: device, dtype: dtype),

					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),
					new C2f(widths[2] + widths[3], widths[2], depths[0], device: device, dtype: dtype),

					new Convs.Conv(widths[2], widths[2], 3, 2, device: device, dtype: dtype),
					new Concat(),
					new C2f(widths[2] + widths[3], widths[3], depths[0], device: device, dtype: dtype),

					new Convs.Conv(widths[3], widths[3], 3, 2, device: device, dtype: dtype),
					new Concat(),
					new C2f(widths[4] + widths[3], widths[4], depths[0], device: device, dtype: dtype),

					new Head.Detect(nc, reg_max: this.reg_max, end2end: this.end2end, ch: ch, device: device, dtype: dtype)
				);
				return mod;
			}


			public override (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds)? forward(Tensor x)
			{
				//using (NewDisposeScope())
				{
					(Dictionary<string, torch.Tensor> inference, Dictionary<string, object>)? result = null;
					List<Tensor> outputs = new List<Tensor>();
					int catCount = 0;
					for (int i = 0; i < model.Count; i++)
					{
						switch (model[i])
						{
							case Module<Tensor, Tensor> mod:
								x = mod.forward(x);
								break;
							case Concat cat:
								x = cat.forward(new Tensor[] { x, outputs[concatIndex[catCount]] });
								catCount++;
								break;
							case Head.Detect detect:
								{
									result = detect.forward(new Tensor[] { outputs[^3], outputs[^2], outputs[^1] });
									break;
								}
							case Head.Classify classify:
								{
									result = classify.forward(new Tensor[] { x });
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

		internal class Yolov5u : Yolov8
		{
			protected override int[] outputIndexs => new int[] { 4, 6, 10, 14, 17, 20, 23 };

			internal Yolov5u(int nc = 80, int reg_max = 16, int? kpt_num = null, int? kpt_dim = null, YoloSize yoloSize = YoloSize.n, bool end2end = false, Device? device = null, torch.ScalarType? dtype = null) : base(nc, reg_max: reg_max, kpt_dim: kpt_dim, kpt_num: kpt_num, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
			{

			}
			internal override ModuleList<Module> BuildModel()
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

				ModuleList<Module> mod = ModuleList<Module>(
					// backbone:
					new Convs.Conv(3, widths[0], 6, 2, 2, device: device, dtype: dtype),                                           // 0-P1/2
					new Convs.Conv(widths[0], widths[1], 3, 2, device: device, dtype: dtype),                    // 1-P2/4
					new C3(widths[1], widths[1], depths[0], device: device, dtype: dtype),
					new Convs.Conv(widths[1], widths[2], 3, 2, device: device, dtype: dtype),                   // 3-P3/8
					new C3(widths[2], widths[2], depths[1], device: device, dtype: dtype),
					new Convs.Conv(widths[2], widths[3], 3, 2, device: device, dtype: dtype),                   // 5-P4/16
					new C3(widths[3], widths[3], depths[2], device: device, dtype: dtype),
					new Convs.Conv(widths[3], widths[4], 3, 2, device: device, dtype: dtype),                  // 7-P5/32
					new C3(widths[4], widths[4], depths[0], device: device, dtype: dtype),
					new Block.SPPF(widths[4], widths[4], 5, device: device, dtype: dtype),

					// head:
					new Convs.Conv(widths[4], widths[3], 1, 1, device: device, dtype: dtype),
					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),                                                                               // cat backbone P4
					new C3(widths[4], widths[3], depths[0], false, device: device, dtype: dtype),    // 13

					new Convs.Conv(widths[3], widths[2], 1, 1, device: device, dtype: dtype),
					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),                                                                               // cat backbone P3
					new C3(widths[3], widths[2], depths[0], false, device: device, dtype: dtype),      // 17 (P3/8-small)

					new Convs.Conv(widths[2], widths[2], 3, 2, device: device, dtype: dtype),
					new Concat(),                                                                               // cat head P4
					new C3(widths[3], widths[3], depths[0], false, device: device, dtype: dtype),      // 20 (P4/16-medium)

					new Convs.Conv(widths[3], widths[3], 3, 2, device: device, dtype: dtype),
					new Concat(),                                                                               // cat head P5
					new C3(widths[4], widths[4], depths[0], false, device: device, dtype: dtype),     // 23 (P5/32-large)

					new Head.Detect(nc, ch: this.ch, reg_max: this.reg_max, end2end: this.end2end, device: this.device, dtype: this.dtype)                                                               // Detect(P3, P4, P5)
				);
				return mod;

			}
		}

		internal class Yolov11 : Yolov8
		{
			protected override int[] outputIndexs => new int[] { 4, 6, 10, 13, 16, 19, 22 };

			internal Yolov11(int nc = 80, int reg_max = 16, int? kpt_num = null, int? kpt_dim = null, bool end2end = false, YoloSize yoloSize = YoloSize.n, Device? device = null, torch.ScalarType? dtype = null) : base(nc, reg_max: reg_max, kpt_dim: kpt_dim, kpt_num: kpt_num, end2end: end2end, yoloSize: yoloSize, device: device, dtype: dtype)
			{

			}

			internal override ModuleList<Module> BuildModel()
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

				ModuleList<Module> mod = new ModuleList<Module>(
					new Convs.Conv(3, widths[0], 3, 2, device: device, dtype: dtype),
					new Convs.Conv(widths[0], widths[1], 3, 2, device: device, dtype: dtype),
					new C3k2(widths[1], widths[2], depthSize, useC3k, e: 0.25f, device: device, dtype: dtype),
					new Convs.Conv(widths[2], widths[2], 3, 2, device: device, dtype: dtype),
					new C3k2(widths[2], widths[3], depthSize, useC3k, e: 0.25f, device: device, dtype: dtype),
					new Convs.Conv(widths[3], widths[3], 3, 2, device: device, dtype: dtype),
					new C3k2(widths[3], widths[3], depthSize, c3k: true, device: device, dtype: dtype),
					new Convs.Conv(widths[3], widths[4], 3, 2, device: device, dtype: dtype),
					new C3k2(widths[4], widths[4], depthSize, c3k: true, device: device, dtype: dtype),
					new Block.SPPF(widths[4], widths[4], 5, device: device, dtype: dtype),
					new C2PSA(widths[4], widths[4], depthSize, device: device, dtype: dtype),

					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),
					new C3k2(widths[4] + widths[3], widths[3], depthSize, useC3k, device: device, dtype: dtype),

					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),
					new C3k2(widths[3] + widths[3], widths[2], depthSize, useC3k, device: device, dtype: dtype),

					new Convs.Conv(widths[2], widths[2], 3, 2, device: device, dtype: dtype),
					new Concat(),
					new C3k2(widths[3] + widths[2], widths[3], depthSize, useC3k, device: device, dtype: dtype),

					new Convs.Conv(widths[3], widths[3], 3, 2, device: device, dtype: dtype),
					new Concat(),
					new C3k2(widths[4] + widths[3], widths[4], depthSize, c3k: true, device: device, dtype: dtype),

					new Head.Detect(nc, reg_max: reg_max, ch: ch, legacy: false, end2end: end2end, device: device, dtype: dtype)
				);
				return mod;
			}
		}

		internal class Yolov12 : Yolov8
		{
			protected override int[] outputIndexs => new int[] { 4, 6, 8, 11, 14, 17, 20 };
			internal Yolov12(int nc = 80, int reg_max = 16, int? kpt_num = null, int? kpt_dim = null, YoloSize yoloSize = YoloSize.n, bool end2end = false, Device? device = null, torch.ScalarType? dtype = null) : base(nc, reg_max: reg_max, kpt_dim: kpt_dim, kpt_num: kpt_num, end2end: end2end, yoloSize: yoloSize, device: device, dtype: dtype)
			{

			}

			internal override ModuleList<Module> BuildModel()
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

				ModuleList<Module> mod = new ModuleList<Module>(
					new Convs. Conv(3, widths[0], 3, 2, device: device, dtype: dtype),                                                                     // 0-P1/2
					new Convs.Conv(widths[0], widths[1], 3, 2, device: device, dtype: dtype),                                                          // 1-P2/4
					new C3k2(widths[1], widths[2], depthSize, useC3k, e: 0.25f, device: device, dtype: dtype),
					new Convs.Conv(widths[2], widths[2], 3, 2, device: device, dtype: dtype),                                                         // 3-P3/8
					new C3k2(widths[2], widths[3], depthSize, useC3k, e: 0.25f, device: device, dtype: dtype),
					new Convs.Conv(widths[3], widths[3], 3, 2, device: device, dtype: dtype),                                                         // 5-P4/16
					new A2C2f(widths[3], widths[3], n: 2 * n_nultiple, a2: true, area: 4, useResidual, mlp_ratio, device: device, dtype: dtype),
					new Convs.Conv(widths[3], widths[4], 3, 2, device: device, dtype: dtype),                                                        // 7-P5/32
					new A2C2f(widths[4], widths[4], n: 2 * n_nultiple, a2: true, area: 1, useResidual, mlp_ratio, device: device, dtype: dtype),

					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),                                                                                       // cat backbone P4
					new A2C2f(widths[4] + widths[3], widths[3], n: n_nultiple, a2: false, area: -1, useResidual, mlp_ratio, device: device, dtype: dtype),                                   // 11

					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),                                                                                       // cat backbone P3
					new A2C2f(widths[3] + widths[3], widths[2], n: n_nultiple, a2: false, area: -1, useResidual, mlp_ratio, device: device, dtype: dtype),                                    // 14 (P3/8-small)

					new Convs.Conv(widths[2], widths[2], 3, 2, device: device, dtype: dtype),
					new Concat(),                                                                                       // cat head P4
					new A2C2f(widths[3] + widths[2], widths[3], n: n_nultiple, a2: false, area: -1, useResidual, mlp_ratio, device: device, dtype: dtype),                                        // 17 (P4/16-medium)

					new Convs.Conv(widths[3], widths[3], 3, 2, device: device, dtype: dtype),
					new Concat(),                                                                                       // cat head P5
					new C3k2(widths[4] + widths[3], widths[4], depthSize, c3k: true, device: device, dtype: dtype),                       // 20 (P5/32-large)

					new Head.Detect(nc, reg_max: this.reg_max, ch: this.ch, legacy: false, end2end: this.end2end, device: this.device, dtype: this.dtype)                                                                       // Detect(P3, P4, P5)
					);
				return mod;
			}
		}

		internal class Yolov5uSegment : Yolov5u
		{
			private readonly ModuleList<Module> model;

			internal Yolov5uSegment(int nc = 80, int reg_max = 16, YoloSize yoloSize = YoloSize.n, bool end2end = false, Device? device = null, torch.ScalarType? dtype = null) : base(nc, reg_max: reg_max, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
			{

			}

			internal override ModuleList<Module> BuildModel()
			{
				var mod = base.BuildModel();
				mod.RemoveAt(mod.Count - 1); // remove Detect
				mod.Add(new Head.Segment(nc: nc, ch: ch, reg_max: reg_max, npr: ch[0], end2end: end2end, device: device, dtype: dtype));
				return mod;
			}


		}

		internal class Yolov8Segment : Yolov8
		{
			internal Yolov8Segment(int nc = 80, int reg_max = 16, YoloSize yoloSize = YoloSize.n, bool end2end = false, Device? device = null, torch.ScalarType? dtype = null) : base(nc, reg_max: reg_max, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
			{

			}

			internal override ModuleList<Module> BuildModel()
			{
				var mod = base.BuildModel();
				mod.RemoveAt(mod.Count - 1); // remove Detect
				mod.Add(new Head.Segment(nc: nc, ch: ch, reg_max: reg_max, npr: ch[0], end2end: end2end, device: device, dtype: dtype));
				return mod;
			}

		}

		internal class Yolov11Segment : Yolov11
		{

			internal Yolov11Segment(int nc = 80, int reg_max = 16, YoloSize yoloSize = YoloSize.n, bool end2end = false, Device? device = null, torch.ScalarType? dtype = null) : base(nc, reg_max: reg_max, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
			{

			}

			internal override ModuleList<Module> BuildModel()
			{
				var mod = base.BuildModel();
				mod.RemoveAt(mod.Count - 1); // remove Detect
				mod.Add(new Head.Segment(nc: nc, ch: ch, reg_max: reg_max, npr: ch[0], end2end: end2end, legacy: false, device: device, dtype: dtype));
				return mod;
			}

		}

		internal class Yolov12Segment : Yolov12
		{

			internal Yolov12Segment(int nc = 80, int reg_max = 16, YoloSize yoloSize = YoloSize.n, bool end2end = false, Device? device = null, torch.ScalarType? dtype = null) : base(nc, reg_max: reg_max, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
			{

			}

			internal override ModuleList<Module> BuildModel()
			{
				var mod = base.BuildModel();
				mod.RemoveAt(mod.Count - 1); // remove Detect
				mod.Add(new Head.Segment(nc: nc, ch: ch, reg_max: reg_max, npr: ch[0], end2end: end2end, legacy: false, device: device, dtype: dtype));
				return mod;
			}

		}

		internal class Yolov5uObb : Yolov5u
		{
			internal Yolov5uObb(int nc = 80, int reg_max = 16, YoloSize yoloSize = YoloSize.n, bool end2end = false, Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, reg_max: reg_max, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
			{

			}

			internal override ModuleList<Module> BuildModel()
			{
				var mod = base.BuildModel();
				mod.RemoveAt(mod.Count - 1); // remove Detect
				mod.Add(new Head.Obb(nc, reg_max: this.reg_max, end2end: this.end2end, ch: ch, device: device, dtype: dtype));
				return mod;
			}
		}

		internal class Yolov8Obb : Yolov8
		{
			internal Yolov8Obb(int nc = 80, int reg_max = 16, YoloSize yoloSize = YoloSize.n, bool end2end = false, Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, reg_max: reg_max, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
			{

			}

			internal override ModuleList<Module> BuildModel()
			{
				var mod = base.BuildModel();
				mod.RemoveAt(mod.Count - 1); // remove Detect
				mod.Add(new Head.Obb(nc, reg_max: this.reg_max, end2end: this.end2end, ch: ch, device: device, dtype: dtype));
				return mod;
			}
		}

		internal class Yolov11Obb : Yolov11
		{
			internal Yolov11Obb(int nc = 80, int reg_max = 16, YoloSize yoloSize = YoloSize.n, bool end2end = false, Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, reg_max: reg_max, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
			{

			}

			internal override ModuleList<Module> BuildModel()
			{
				var mod = base.BuildModel();
				mod.RemoveAt(mod.Count - 1); // remove Detect
				mod.Add(new Head.Obb(nc, reg_max: reg_max, end2end: this.end2end, legacy: false, ch: ch, device: device, dtype: dtype));
				return mod;
			}
		}

		internal class Yolov12Obb : Yolov12
		{
			internal Yolov12Obb(int nc = 80, int reg_max = 16, YoloSize yoloSize = YoloSize.n, bool end2end = false, Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, reg_max: reg_max, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
			{

			}

			internal override ModuleList<Module> BuildModel()
			{
				var mod = base.BuildModel();
				mod.RemoveAt(mod.Count - 1); // remove Detect
				mod.Add(new Head.Obb(nc, reg_max: reg_max, end2end: this.end2end, legacy: false, ch: ch, device: device, dtype: dtype));
				return mod;
			}
		}

		internal class Yolov5uPose : Yolov5u
		{
			internal Yolov5uPose(int nc = 80, int reg_max = 16, int kpt_num = 17, int kpt_dim = 3, YoloSize yoloSize = YoloSize.n, bool end2end = false, Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, reg_max: reg_max, kpt_num: kpt_num, kpt_dim: kpt_dim, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
			{

			}

			internal override ModuleList<Module> BuildModel()
			{
				var mod = base.BuildModel();
				mod.RemoveAt(mod.Count - 1); // remove Detect
				mod.Add(new Head.Pose(nc, reg_max: reg_max, keypoint_num: this.kpt_num.Value, keypoint_dim: this.kpt_dim.Value, end2end: this.end2end, ch: ch, device: device, dtype: dtype));
				return mod;
			}
		}

		internal class Yolov8Pose : Yolov8
		{
			internal Yolov8Pose(int nc = 80, int reg_max = 16, int kpt_num = 17, int kpt_dim = 3, YoloSize yoloSize = YoloSize.n, bool end2end = false, Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, reg_max: reg_max, kpt_num: kpt_num, kpt_dim: kpt_dim, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
			{

			}

			internal override ModuleList<Module> BuildModel()
			{
				var mod = base.BuildModel();
				mod.RemoveAt(mod.Count - 1); // remove Detect
				mod.Add(new Head.Pose(nc, reg_max: reg_max, keypoint_num: this.kpt_num.Value, keypoint_dim: this.kpt_dim.Value, end2end: this.end2end, ch: ch, device: device, dtype: dtype));
				return mod;
			}
		}

		internal class Yolov11Pose : Yolov11
		{
			internal Yolov11Pose(int nc = 80, int reg_max = 16, int kpt_num = 17, int kpt_dim = 3, YoloSize yoloSize = YoloSize.n, bool end2end = false, Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, reg_max: reg_max, kpt_num: kpt_num, kpt_dim: kpt_dim, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
			{

			}

			internal override ModuleList<Module> BuildModel()
			{
				var mod = base.BuildModel();
				mod.RemoveAt(mod.Count - 1); // remove Detect
				mod.Add(new Head.Pose(nc, reg_max: reg_max, keypoint_num: this.kpt_num.Value, keypoint_dim: this.kpt_dim.Value, legacy: false, end2end: this.end2end, ch: ch, device: device, dtype: dtype));
				return mod;
			}
		}

		internal class Yolov12Pose : Yolov12
		{
			internal Yolov12Pose(int nc = 80, int reg_max = 16, int kpt_num = 17, int kpt_dim = 3, YoloSize yoloSize = YoloSize.n, bool end2end = false, Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, reg_max: reg_max, kpt_num: kpt_num, kpt_dim: kpt_dim, yoloSize: yoloSize, end2end: end2end, device: device, dtype: dtype)
			{

			}

			internal override ModuleList<Module> BuildModel()
			{
				var mod = base.BuildModel();
				mod.RemoveAt(mod.Count - 1); // remove Detect
				mod.Add(new Head.Pose(nc, reg_max: reg_max, keypoint_num: this.kpt_num.Value, keypoint_dim: this.kpt_dim.Value, legacy: false, end2end: this.end2end, ch: ch, device: device, dtype: dtype));
				return mod;
			}
		}

		internal class Yolov5uClassify : Yolov5u
		{
			internal Yolov5uClassify(int nc = 80, YoloSize yoloSize = YoloSize.n, Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, yoloSize: yoloSize, device: device, dtype: dtype)
			{

			}

			internal override ModuleList<Module> BuildModel()
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

		internal class Yolov8Classify : Yolov8
		{
			internal Yolov8Classify(int nc = 80, YoloSize yoloSize = YoloSize.n, Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, yoloSize: yoloSize, device: device, dtype: dtype)
			{

			}

			internal override ModuleList<Module> BuildModel()
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

		internal class Yolov11Classify : Yolov11
		{
			internal Yolov11Classify(int nc = 80, YoloSize yoloSize = YoloSize.n, Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, yoloSize: yoloSize, device: device, dtype: dtype)
			{

			}

			internal override ModuleList<Module> BuildModel()
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
			internal Yolov12Classify(int nc = 80, YoloSize yoloSize = YoloSize.n, Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, yoloSize: yoloSize, device: device, dtype: dtype)
			{

			}

			internal override ModuleList<Module> BuildModel()
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
