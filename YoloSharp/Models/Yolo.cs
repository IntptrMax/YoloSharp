using TorchSharp;
using TorchSharp.Modules;
using YoloSharp.Types;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static YoloSharp.Modules.Modules;

namespace YoloSharp.Models
{
	internal class Yolo
	{
		public class Yolov5 : Yolov8
		{
			protected override int[] outputIndexs => new int[] { 4, 6, 10, 14, 17, 20, 23 };

			public Yolov5(int nc = 80, YoloSize yoloSize = YoloSize.n, Device? device = null, torch.ScalarType? dtype = null) : base(nc, yoloSize, device, dtype)
			{

			}

			internal override ModuleList<Module> BuildModel(int nc, YoloSize yoloSize, Device? device, torch.ScalarType? dtype)
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

				float p3_d = 8.0f;
				float p4_d = 16.0f;
				float p5_d = 32.0f;

				float[][] ach = new float[][] { new float[] { 10 / p3_d, 13 / p3_d, 16 / p3_d, 30 / p3_d, 33 / p3_d, 23 / p3_d }, // P3/8
						new float[] { 30 / p4_d, 61 / p4_d, 62 / p4_d, 45 / p4_d, 59 / p4_d, 119 / p4_d },// P4/16
						new float[] { 116 / p5_d, 90 / p5_d, 156 / p5_d, 198 / p5_d, 373 / p5_d, 326 / p5_d }};   // P5/32

				widths = new List<int> { 64, 128, 256, 512, 1024 }.Select(w => (int)(w * width_multiple)).ToArray();
				int[] depths = new List<int> { 3, 6, 9 }.Select(d => (int)(d * depth_multiple)).ToArray();
				ch = new int[] { widths[2], widths[3], widths[4] };

				ModuleList<Module> mod = ModuleList<Module>(
					// backbone:
					new Conv(3, widths[0], 6, 2, 2, device: device, dtype: dtype),                                           // 0-P1/2
					new Conv(widths[0], widths[1], 3, 2, device: device, dtype: dtype),                    // 1-P2/4
					new C3(widths[1], widths[1], depths[0], device: device, dtype: dtype),
					new Conv(widths[1], widths[2], 3, 2, device: device, dtype: dtype),                   // 3-P3/8
					new C3(widths[2], widths[2], depths[1], device: device, dtype: dtype),
					new Conv(widths[2], widths[3], 3, 2, device: device, dtype: dtype),                   // 5-P4/16
					new C3(widths[3], widths[3], depths[2], device: device, dtype: dtype),
					new Conv(widths[3], widths[4], 3, 2, device: device, dtype: dtype),                  // 7-P5/32
					new C3(widths[4], widths[4], depths[0], device: device, dtype: dtype),
					new SPPF(widths[4], widths[4], 5, device: device, dtype: dtype),

					// head:
					new Conv(widths[4], widths[3], 1, 1, device: device, dtype: dtype),
					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),                                                                               // cat backbone P4
					new C3(widths[4], widths[3], depths[0], false, device: device, dtype: dtype),    // 13

					new Conv(widths[3], widths[2], 1, 1, device: device, dtype: dtype),
					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),                                                                               // cat backbone P3
					new C3(widths[3], widths[2], depths[0], false, device: device, dtype: dtype),      // 17 (P3/8-small)

					new Conv(widths[2], widths[2], 3, 2, device: device, dtype: dtype),
					new Concat(),                                                                               // cat head P4
					new C3(widths[3], widths[3], depths[0], false, device: device, dtype: dtype),      // 20 (P4/16-medium)

					new Conv(widths[3], widths[3], 3, 2, device: device, dtype: dtype),
					new Concat(),                                                                               // cat head P5
					new C3(widths[4], widths[4], depths[0], false, device: device, dtype: dtype),     // 23 (P5/32-large)

					new Yolov5Detect(nc, ch, ach, device: device, dtype: dtype)                                                               // Detect(P3, P4, P5)
				);
				return mod;
			}

		}

		public class Yolov5u : Yolov8
		{
			protected override int[] outputIndexs => new int[] { 4, 6, 10, 14, 17, 20, 23 };

			public Yolov5u(int nc = 80, YoloSize yoloSize = YoloSize.n, Device? device = null, torch.ScalarType? dtype = null) : base(nc, yoloSize, device, dtype)
			{

			}
			internal override ModuleList<Module> BuildModel(int nc, YoloSize yoloSize, Device? device, torch.ScalarType? dtype)
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
					new Conv(3, widths[0], 6, 2, 2, device: device, dtype: dtype),                                           // 0-P1/2
					new Conv(widths[0], widths[1], 3, 2, device: device, dtype: dtype),                    // 1-P2/4
					new C3(widths[1], widths[1], depths[0], device: device, dtype: dtype),
					new Conv(widths[1], widths[2], 3, 2, device: device, dtype: dtype),                   // 3-P3/8
					new C3(widths[2], widths[2], depths[1], device: device, dtype: dtype),
					new Conv(widths[2], widths[3], 3, 2, device: device, dtype: dtype),                   // 5-P4/16
					new C3(widths[3], widths[3], depths[2], device: device, dtype: dtype),
					new Conv(widths[3], widths[4], 3, 2, device: device, dtype: dtype),                  // 7-P5/32
					new C3(widths[4], widths[4], depths[0], device: device, dtype: dtype),
					new SPPF(widths[4], widths[4], 5, device: device, dtype: dtype),

					// head:
					new Conv(widths[4], widths[3], 1, 1, device: device, dtype: dtype),
					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),                                                                               // cat backbone P4
					new C3(widths[4], widths[3], depths[0], false, device: device, dtype: dtype),    // 13

					new Conv(widths[3], widths[2], 1, 1, device: device, dtype: dtype),
					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),                                                                               // cat backbone P3
					new C3(widths[3], widths[2], depths[0], false, device: device, dtype: dtype),      // 17 (P3/8-small)

					new Conv(widths[2], widths[2], 3, 2, device: device, dtype: dtype),
					new Concat(),                                                                               // cat head P4
					new C3(widths[3], widths[3], depths[0], false, device: device, dtype: dtype),      // 20 (P4/16-medium)

					new Conv(widths[3], widths[3], 3, 2, device: device, dtype: dtype),
					new Concat(),                                                                               // cat head P5
					new C3(widths[4], widths[4], depths[0], false, device: device, dtype: dtype),     // 23 (P5/32-large)

					new Yolov8Detect(nc, ch, device: device, dtype: dtype)                                                               // Detect(P3, P4, P5)
				);
				return mod;

			}
		}

		public class Yolov8 : Module<Tensor, Tensor[]>
		{
			private ModuleList<Module> model;
			protected virtual int[] outputIndexs => new int[] { 4, 6, 9, 12, 15, 18, 21 };
			protected virtual int[] concatIndex => new int[] { 1, 0, 3, 2 };

			protected int[] ch;
			protected int[] kpt_shape;
			protected int[] widths;


			public Yolov8(int nc = 80, YoloSize yoloSize = YoloSize.n, Device? device = null, torch.ScalarType? dtype = null, int[] kpt_shape = null) : base(nameof(Yolov8))
			{
				this.kpt_shape = kpt_shape;
				model = BuildModel(nc, yoloSize, device, dtype);
				RegisterComponents();
			}

			internal virtual ModuleList<Module> BuildModel(int nc, YoloSize yoloSize, Device? device, torch.ScalarType? dtype)
			{
				var (depth_multiple, width_multiple, max_channels) = yoloSize switch
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
					new Conv(3, widths[0], 3, 2, device: device, dtype: dtype),
					new Conv(widths[0], widths[1], 3, 2, device: device, dtype: dtype),
					new C2f(widths[1], widths[1], depths[0], true, device: device, dtype: dtype),
					new Conv(widths[1], widths[2], 3, 2, device: device, dtype: dtype),
					new C2f(widths[2], widths[2], depths[1], true, device: device, dtype: dtype),
					new Conv(widths[2], widths[3], 3, 2, device: device, dtype: dtype),
					new C2f(widths[3], widths[3], depths[1], true, device: device, dtype: dtype),
					new Conv(widths[3], widths[4], 3, 2, device: device, dtype: dtype),
					new C2f(widths[4], widths[4], depths[0], true, device: device, dtype: dtype),
					new SPPF(widths[4], widths[4], 5, device: device, dtype: dtype),

					// head:
					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),
					new C2f(widths[3] + widths[4], widths[3], depths[0], device: device, dtype: dtype),

					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),
					new C2f(widths[2] + widths[3], widths[2], depths[0], device: device, dtype: dtype),

					new Conv(widths[2], widths[2], 3, 2, device: device, dtype: dtype),
					new Concat(),
					new C2f(widths[2] + widths[3], widths[3], depths[0], device: device, dtype: dtype),

					new Conv(widths[3], widths[3], 3, 2, device: device, dtype: dtype),
					new Concat(),
					new C2f(widths[4] + widths[3], widths[4], depths[0], device: device, dtype: dtype),

					new Yolov8Detect(nc, ch, device: device, dtype: dtype)
				);
				return mod;
			}


			public override Tensor[] forward(Tensor x)
			{
				using (NewDisposeScope())
				{
					List<Tensor> outputs = new List<Tensor>();
					int catCount = 0;
					Tensor[] list = null;
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
							case Modules.Modules.Classify classify:
								{
									list = classify.forward(new Tensor[] { x });
									break;
								}
							case Modules.Modules.Yolov5Detect v5Detect:
								{
									list = v5Detect.forward(new Tensor[] { outputs[outputs.Count - 3], outputs[outputs.Count - 2], outputs[outputs.Count - 1] });
									break;
								}
							case Modules.Modules.Yolov8Detect v8Detect:
								{
									list = v8Detect.forward(new Tensor[] { outputs[outputs.Count - 3], outputs[outputs.Count - 2], outputs[outputs.Count - 1] });
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
					list = list.Select(x => x.MoveToOuterDisposeScope()).ToArray();
					return list;

					//Tensor[] list = ((Module<Tensor[], Tensor[]>)model.Last()).forward(new Tensor[] { outputs[outputs.Count - 3], outputs[outputs.Count - 2], outputs[outputs.Count - 1] });
					//list = list.Select(x => x.MoveToOuterDisposeScope()).ToArray();
					//return list;
				}
			}

		}

		public class Yolov11 : Yolov8
		{
			protected override int[] outputIndexs => new int[] { 4, 6, 10, 13, 16, 19, 22 };

			public Yolov11(int nc = 80, YoloSize yoloSize = YoloSize.n, Device? device = null, torch.ScalarType? dtype = null, int[] kpt_shape = null) : base(nc, yoloSize, device, dtype, kpt_shape: kpt_shape)
			{

			}

			internal override ModuleList<Module> BuildModel(int nc, YoloSize yoloSize, Device? device, torch.ScalarType? dtype)
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
					new Conv(3, widths[0], 3, 2, device: device, dtype: dtype),
					new Conv(widths[0], widths[1], 3, 2, device: device, dtype: dtype),
					new C3k2(widths[1], widths[2], depthSize, useC3k, e: 0.25f, device: device, dtype: dtype),
					new Conv(widths[2], widths[2], 3, 2, device: device, dtype: dtype),
					new C3k2(widths[2], widths[3], depthSize, useC3k, e: 0.25f, device: device, dtype: dtype),
					new Conv(widths[3], widths[3], 3, 2, device: device, dtype: dtype),
					new C3k2(widths[3], widths[3], depthSize, c3k: true, device: device, dtype: dtype),
					new Conv(widths[3], widths[4], 3, 2, device: device, dtype: dtype),
					new C3k2(widths[4], widths[4], depthSize, c3k: true, device: device, dtype: dtype),
					new SPPF(widths[4], widths[4], 5, device: device, dtype: dtype),
					new C2PSA(widths[4], widths[4], depthSize, device: device, dtype: dtype),

					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),
					new C3k2(widths[4] + widths[3], widths[3], depthSize, useC3k, device: device, dtype: dtype),

					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),
					new C3k2(widths[3] + widths[3], widths[2], depthSize, useC3k, device: device, dtype: dtype),

					new Conv(widths[2], widths[2], 3, 2, device: device, dtype: dtype),
					new Concat(),
					new C3k2(widths[3] + widths[2], widths[3], depthSize, useC3k, device: device, dtype: dtype),

					new Conv(widths[3], widths[3], 3, 2, device: device, dtype: dtype),
					new Concat(),
					new C3k2(widths[4] + widths[3], widths[4], depthSize, c3k: true, device: device, dtype: dtype),

					new Yolov8Detect(nc, ch, false, device: device, dtype: dtype)
				);
				return mod;
			}
		}

		public class Yolov12 : Yolov8
		{
			protected override int[] outputIndexs => new int[] { 4, 6, 8, 11, 14, 17, 20 };
			public Yolov12(int nc = 80, YoloSize yoloSize = YoloSize.n, Device? device = null, torch.ScalarType? dtype = null) : base(nc, yoloSize, device, dtype)
			{

			}

			internal override ModuleList<Module> BuildModel(int nc, YoloSize yoloSize, Device? device, torch.ScalarType? dtype)
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
					new Conv(3, widths[0], 3, 2, device: device, dtype: dtype),                                                                     // 0-P1/2
					new Conv(widths[0], widths[1], 3, 2, device: device, dtype: dtype),                                                          // 1-P2/4
					new C3k2(widths[1], widths[2], depthSize, useC3k, e: 0.25f, device: device, dtype: dtype),
					new Conv(widths[2], widths[2], 3, 2, device: device, dtype: dtype),                                                         // 3-P3/8
					new C3k2(widths[2], widths[3], depthSize, useC3k, e: 0.25f, device: device, dtype: dtype),
					new Conv(widths[3], widths[3], 3, 2, device: device, dtype: dtype),                                                         // 5-P4/16
					new A2C2f(widths[3], widths[3], n: 2 * n_nultiple, a2: true, area: 4, useResidual, mlp_ratio, device: device, dtype: dtype),
					new Conv(widths[3], widths[4], 3, 2, device: device, dtype: dtype),                                                        // 7-P5/32
					new A2C2f(widths[4], widths[4], n: 2 * n_nultiple, a2: true, area: 1, useResidual, mlp_ratio, device: device, dtype: dtype),

					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),                                                                                       // cat backbone P4
					new A2C2f(widths[4] + widths[3], widths[3], n: n_nultiple, a2: false, area: -1, useResidual, mlp_ratio, device: device, dtype: dtype),                                   // 11

					Upsample(scale_factor: new double[] { 2, 2 }, mode: UpsampleMode.Nearest),
					new Concat(),                                                                                       // cat backbone P3
					new A2C2f(widths[3] + widths[3], widths[2], n: n_nultiple, a2: false, area: -1, useResidual, mlp_ratio, device: device, dtype: dtype),                                    // 14 (P3/8-small)

					new Conv(widths[2], widths[2], 3, 2, device: device, dtype: dtype),
					new Concat(),                                                                                       // cat head P4
					new A2C2f(widths[3] + widths[2], widths[3], n: n_nultiple, a2: false, area: -1, useResidual, mlp_ratio, device: device, dtype: dtype),                                        // 17 (P4/16-medium)

					new Conv(widths[3], widths[3], 3, 2, device: device, dtype: dtype),
					new Concat(),                                                                                       // cat head P5
					new C3k2(widths[4] + widths[3], widths[4], depthSize, c3k: true, device: device, dtype: dtype),                       // 20 (P5/32-large)

					new Yolov8Detect(nc, ch, false, device: device, dtype: dtype)                                                                       // Detect(P3, P4, P5)
					);
				return mod;
			}
		}

		public class Yolov5uSegment : Yolov5u
		{
			private readonly ModuleList<Module> model;

			public Yolov5uSegment(int nc = 80, YoloSize yoloSize = YoloSize.n, Device? device = null, torch.ScalarType? dtype = null) : base(nc, yoloSize, device, dtype)
			{

			}

			internal override ModuleList<Module> BuildModel(int nc, YoloSize yoloSize, Device? device, torch.ScalarType? dtype)
			{
				var mod = base.BuildModel(nc, yoloSize, device, dtype);
				mod.RemoveAt(mod.Count - 1); // remove Detect
				mod.Add(new Segment(ch, nc, npr: ch[0], device: device, dtype: dtype));
				return mod;
			}


		}

		public class Yolov8Segment : Yolov8
		{
			public Yolov8Segment(int nc = 80, YoloSize yoloSize = YoloSize.n, Device? device = null, torch.ScalarType? dtype = null) : base(nc, yoloSize, device, dtype)
			{

			}

			internal override ModuleList<Module> BuildModel(int nc, YoloSize yoloSize, Device? device, torch.ScalarType? dtype)
			{
				var mod = base.BuildModel(nc, yoloSize, device, dtype);
				mod.RemoveAt(mod.Count - 1); // remove Detect
				mod.Add(new Segment(ch, nc, npr: ch[0], device: device, dtype: dtype));
				return mod;
			}

		}

		public class Yolov11Segment : Yolov11
		{

			public Yolov11Segment(int nc = 80, YoloSize yoloSize = YoloSize.n, Device? device = null, torch.ScalarType? dtype = null) : base(nc, yoloSize, device, dtype)
			{

			}

			internal override ModuleList<Module> BuildModel(int nc, YoloSize yoloSize, Device? device, torch.ScalarType? dtype)
			{
				var mod = base.BuildModel(nc, yoloSize, device, dtype);
				mod.RemoveAt(mod.Count - 1); // remove Detect
				mod.Add(new Segment(ch, nc, npr: ch[0], legacy: false, device: device, dtype: dtype));
				return mod;
			}
		}

		public class Yolov8Obb : Yolov8
		{
			public Yolov8Obb(int nc = 80, YoloSize yoloSize = YoloSize.n, Device? device = null, torch.ScalarType? dtype = null) : base(nc, yoloSize, device, dtype)
			{

			}

			internal override ModuleList<Module> BuildModel(int nc, YoloSize yoloSize, Device? device, torch.ScalarType? dtype)
			{
				var mod = base.BuildModel(nc, yoloSize, device, dtype);
				mod.RemoveAt(mod.Count - 1); // remove Detect
				mod.Add(new OBB(ch, nc, device: device, dtype: dtype));
				return mod;
			}
		}

		public class Yolov11Obb : Yolov11
		{
			public Yolov11Obb(int nc = 80, YoloSize yoloSize = YoloSize.n, Device? device = null, torch.ScalarType? dtype = null) : base(nc, yoloSize, device, dtype)
			{

			}

			internal override ModuleList<Module> BuildModel(int nc, YoloSize yoloSize, Device? device, torch.ScalarType? dtype)
			{
				var mod = base.BuildModel(nc, yoloSize, device, dtype);
				mod.RemoveAt(mod.Count - 1); // remove Detect
				mod.Add(new OBB(ch, nc, device: device, dtype: dtype));
				return mod;
			}
		}

		public class Yolov8Pose : Yolov8
		{
			public Yolov8Pose(int nc = 80, int[] kpt_shape = null, YoloSize yoloSize = YoloSize.n, Device? device = null, torch.ScalarType? dtype = null) : base(nc, yoloSize, device, dtype, kpt_shape: kpt_shape)
			{

			}

			internal override ModuleList<Module> BuildModel(int nc, YoloSize yoloSize, Device? device, torch.ScalarType? dtype)
			{
				var mod = base.BuildModel(nc, yoloSize, device, dtype);
				mod.RemoveAt(mod.Count - 1); // remove Detect
				mod.Add(new Pose(nc: nc, kpt_shape: this.kpt_shape, ch: ch, device: device, dtype: dtype));
				return mod;
			}
		}

		public class Yolov11Pose : Yolov11
		{
			public Yolov11Pose(int nc = 80, int[] kpt_shape = null, YoloSize yoloSize = YoloSize.n, Device? device = null, torch.ScalarType? dtype = null) : base(nc, yoloSize, device, dtype, kpt_shape)
			{

			}

			internal override ModuleList<Module> BuildModel(int nc, YoloSize yoloSize, Device? device, torch.ScalarType? dtype)
			{
				var mod = base.BuildModel(nc, yoloSize, device, dtype);
				mod.RemoveAt(mod.Count - 1); // remove Detect
				mod.Add(new Pose(nc: nc, kpt_shape: kpt_shape, ch: ch, device: device, dtype: dtype));
				return mod;
			}
		}

		public class Yolov8Classify : Yolov8
		{
			public Yolov8Classify(int nc = 80, YoloSize yoloSize = YoloSize.n, Device? device = null, torch.ScalarType? dtype = null) : base(nc, yoloSize, device, dtype)
			{

			}

			internal override ModuleList<Module> BuildModel(int nc, YoloSize yoloSize, Device? device, torch.ScalarType? dtype)
			{
				var mod = base.BuildModel(nc, yoloSize, device, dtype);
				for (int i = 0; i < 14; i++)
				{
					mod.RemoveAt(mod.Count - 1); // remove Detect
				}
				mod.Add(new Classify(base.widths[4], nc, device: device, dtype: dtype));
				return mod;
			}
		}

		public class Yolov11Classify : Yolov11
		{
			public Yolov11Classify(int nc = 80, YoloSize yoloSize = YoloSize.n, Device? device = null, torch.ScalarType? dtype = null) : base(nc, yoloSize, device, dtype)
			{
			}
			internal override ModuleList<Module> BuildModel(int nc, YoloSize yoloSize, Device? device, torch.ScalarType? dtype)
			{
				var mod = base.BuildModel(nc, yoloSize, device, dtype);
				for (int i = 0; i < 13; i++)
				{
					mod.RemoveAt(mod.Count - 1); // remove Detect
				}
				mod.Add(new Classify(base.widths[4], nc, device: device, dtype: dtype));
				return mod;
			}

		}
	}
}
