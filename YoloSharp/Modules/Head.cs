using TorchSharp;
using TorchSharp.Modules;

namespace YoloSharp.Modules
{
	internal class Head
	{
		/// <summary>
		/// YOLOv9 Detect head — contributed back to the open-source YoloSharp project.
		/// v9 yaml Detect over 6 inputs [aux(P3,P4,P5), main(P3,P4,P5)]; inference uses the
		/// LAST 3 (main). Channel basis: cv2/cv3 built per-branch from its own ch (aux uses ch[0] basis,
		/// matching official v9 Detect: c2/c3 from ch[0] for first half, c4/c5 from ch[nl] for second).
		/// Weight layout verified 1:1 with real yolov9-s: cv2.{0..5}, cv3.{0..5}, dfl.
		/// </summary>
		/// <summary>
		/// [Rewrite 2026-11-04] Aligned with the ACTUAL head used by yolov9-s.yaml training:
		/// `Detect` (v8-style, models/yolo.py:78 is DDetect — but the yaml's Detect entry maps to
		/// the v8-style Detect with nl=len(ch)=6). Verified against python: Model('yolov9-s.yaml')
		/// builds `Detect [1, [128,192,256,128,192,256]]` — SIX groups, all in forward+loss,
		/// c2/c3 all based on ch[0], plain convs (no groups=4). Aux branch layers 22-28 DO feed
		/// the head. Previous attempts aligned to DDetect/DualDDetect — wrong target (that's the
		/// converted.pt inference-only checkpoint, not the training yaml).
		/// Matches the actual yolov9-s.yaml training head: v8-style Detect with all six groups, ch[0] base, plain convs. / 对齐yolov9-s.yaml实际训练头
		/// </summary>
		internal class DetectV9 : Detect
		{
			internal DetectV9(int nc = 80, int[] ch = null, int reg_max = 16, torch.Device? device = null, torch.ScalarType? dtype = null)
				: base(nc, reg_max, ch: ch, legacy: true, end2end: false, device: device, dtype: dtype)
			{
				// Rebuild to 6 groups, all based on ch[0] (official Detect: c2=max(ch0/4,4R,16),
				// c3=max(ch0,min(nc*2,128))). Plain convs. / 六组全按ch[0]基准，普通卷积。
				int c2 = Math.Max(Math.Max(ch[0] / 4, reg_max * 4), 16);
				int c3 = Math.Max(ch[0], Math.Min(Math.Max(nc * 2, 1), 128));
				var all2 = new List<Sequential>();
				var all3 = new List<Sequential>();
				for (int i = 0; i < ch.Length; i++)
				{
					all2.Add(torch.nn.Sequential(
						new Convs.Conv(ch[i], c2, 3, device: device, dtype: dtype),
						new Convs.Conv(c2, c2, 3, device: device, dtype: dtype),
						torch.nn.Conv2d(c2, 4 * reg_max, 1, device: device, dtype: dtype)));
					all3.Add(torch.nn.Sequential(
						new Convs.Conv(ch[i], c3, 3, device: device, dtype: dtype),
						new Convs.Conv(c3, c3, 3, device: device, dtype: dtype),
						torch.nn.Conv2d(c3, nc, 1, device: device, dtype: dtype)));
				}
				cv2 = torch.nn.ModuleList(all2.ToArray());
				cv3 = torch.nn.ModuleList(all3.ToArray());
				SetPrivate("nl", ch.Length);
				SetPrivate("no", nc + reg_max * 4);
				SetPrivate("stride", new int[] { 8, 16, 32, 8, 16, 32 });   // 6 layers: aux(P3,P4,P5)+main(P3,P4,P5)
				RegisterComponents();
			}

			void SetPrivate(string field, object value)
			{
				var f = typeof(Detect).GetField(field, System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
				f?.SetValue(this, value);
			}

			public override (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds) forward(torch.Tensor[] x)
			{
				// All 6 groups participate (training AND eval) — identical to v8 Detect with nl=6.
				// / All 6 groups participate (training and inference) — isomorphic to the nl=6 v8 Detect. / 全6组参与
				return base.forward(x);
			}
		}

		internal class Detect : torch.nn.Module<torch.Tensor[], (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds)>
		{
			private long[] shape = null;
			protected torch.Tensor anchors = torch.empty(0); // init
			protected torch.Tensor strides = torch.empty(0); // init

			protected readonly int nc;
			protected readonly int nl;
			private readonly int reg_max;
			private readonly int no;
			private readonly int[] stride;
			protected ModuleList<Sequential> cv2;
			protected ModuleList<Sequential> cv3;
			private readonly torch.nn.Module<torch.Tensor, torch.Tensor> dfl;

			protected ModuleList<Sequential> one2one_cv2;
			protected ModuleList<Sequential> one2one_cv3;

			bool legacy = true;  // backward compatibility for v3/v5/v8/v9 models
			bool dynamic = false;  // force grid reconstruction
			protected int max_det = 300;    // max_det
			bool agnostic_nms = false;

			bool xyxy = false;    // xyxy or xywh output

			protected readonly bool end2end;

			internal Detect(int nc = 80, int reg_max = 16, int[]? ch = null, bool legacy = true, bool end2end = false, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(Detect))
			{
				this.legacy = legacy;
				this.nc = nc; // number of classes
				this.nl = ch.Length;  // number of detection layers
				this.reg_max = reg_max; // DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
				this.no = nc + this.reg_max * 4; // number of outputs per anchor
												 //this.stride = torch.zeros(this.nl).@int().data<int>().ToArray(); // strides computed during build
				this.stride = new int[] { 8, 16, 32 };
				int c2 = Math.Max(16, Math.Max(ch[0] / 4, this.reg_max * 4));
				int c3 = Math.Max(ch[0], Math.Min(this.nc, 100)); // channels
				this.end2end = end2end;
				this.cv2 = torch.nn.ModuleList(ch.Select(x => torch.nn.Sequential(new Convs.Conv(x, c2, 3, device: device, dtype: dtype), new Convs.Conv(c2, c2, 3, device: device, dtype: dtype), torch.nn.Conv2d(c2, 4 * this.reg_max, 1, device: device, dtype: dtype))).ToArray());
				this.cv3 = this.legacy ?
					torch.nn.ModuleList(ch.Select(x => torch.nn.Sequential(new Convs.Conv(x, c3, 3, device: device, dtype: dtype), new Convs.Conv(c3, c3, 3, device: device, dtype: dtype), torch.nn.Conv2d(c3, this.nc, 1, device: device, dtype: dtype))).ToArray()) :
				   torch.nn.ModuleList(ch.Select(x => torch.nn.Sequential(torch.nn.Sequential(new Convs.DWConv(x, x, 3, device: device, dtype: dtype), new Convs.Conv(x, c3, 1, device: device, dtype: dtype)), torch.nn.Sequential(new Convs.DWConv(c3, c3, 3, device: device, dtype: dtype), new Convs.Conv(c3, c3, 1, device: device, dtype: dtype)), torch.nn.Conv2d(c3, this.nc, 1, device: device, dtype: dtype))).ToArray());

				this.dfl = this.reg_max > 1 ? new Block.DFL(this.reg_max, device: device, dtype: dtype) : torch.nn.Identity();
			}

			protected virtual Dictionary<string, ModuleList<Sequential>> one2many()
			{
				Dictionary<string, ModuleList<Sequential>> result = new Dictionary<string, ModuleList<Sequential>>();
				result["box_head"] = this.cv2;
				result["cls_head"] = this.cv3;
				return result;
			}

			protected virtual Dictionary<string, ModuleList<Sequential>> one2one()
			{
				Dictionary<string, ModuleList<Sequential>> result = new Dictionary<string, ModuleList<Sequential>>();
				result["box_head"] = this.one2one_cv2;
				result["cls_head"] = this.one2one_cv3;
				return result;
			}

			protected virtual Dictionary<string, object> forward_head(torch.Tensor[] x, Dictionary<string, ModuleList<Sequential>> heads)
			{
				Dictionary<string, object> result = new Dictionary<string, object>();
				bool get_box_head = heads.TryGetValue("box_head", out ModuleList<Sequential>? box_head);
				bool get_cls_head = heads.TryGetValue("cls_head", out ModuleList<Sequential>? cls_head);
				if (!get_box_head || !get_cls_head)
				{
					return result;
				}
				long bs = x[0].shape[0];  // batch size
				torch.Tensor boxes = torch.cat(Enumerable.Range(0, this.nl).Select(i => box_head![i].forward(x[i]).view(bs, 4 * this.reg_max, -1)).ToArray(), dim: -1);
				torch.Tensor scores = torch.cat(Enumerable.Range(0, this.nl).Select(i => cls_head![i].forward(x[i]).view(bs, this.nc, -1)).ToArray(), dim: -1);
				result["feats"] = x;
				result["boxes"] = boxes;
				result["scores"] = scores;
				return result;
			}

			public override (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds) forward(torch.Tensor[] x)
			{
				Dictionary<string, object> preds = this.forward_head(x, this.one2many());
				if (this.end2end)
				{
					torch.Tensor[] x_detach = x.Select(xi => xi.detach()).ToArray();
					Dictionary<string, ModuleList<Sequential>> heads = this.one2one();
					Dictionary<string, object> one2one = this.forward_head(x_detach, heads);
					preds = new Dictionary<string, object>
					{
						["one2many"] = preds,
						["one2one"] = one2one
					};
				}
				if (this.training)
				{
					return (null, preds);
				}
				torch.Tensor y = this._inference(this.end2end ? (Dictionary<string, object>)preds["one2one"] : preds);
				if (this.end2end)
				{
					y = this.postprocess(y.permute(0, 2, 1));
				}
				Dictionary<string, torch.Tensor> inference = new Dictionary<string, torch.Tensor>();
				inference["boxes"] = y;
				return (inference, preds);
			}

			protected virtual torch.Tensor postprocess(torch.Tensor preds)
			{
				//boxes, scores
				torch.Tensor[] boxes_scores = preds.split(new long[] { 4, this.nc }, dim: -1);
				torch.Tensor boxes = boxes_scores[0];
				torch.Tensor scores = boxes_scores[1];

				(scores, torch.Tensor conf, torch.Tensor idx) = this.get_topk_index(scores, this.max_det);
				boxes = boxes.gather(dim: 1, index: idx.repeat(1, 1, 4));
				return torch.cat(new torch.Tensor[] { boxes, scores, conf }, dim: -1);
			}

			public virtual void bias_init()
			{
				using (torch.no_grad())
				{
					for (int i = 0; i < this.nl; i++)
					{
						SetLastConvBias(this.cv2[i], 2.0f, count: null);

						float clsValue = (float)Math.Log(5.0 / this.nc / Math.Pow(640.0 / this.stride[i], 2));
						SetLastConvBias(this.cv3[i], clsValue, count: this.nc);
					}

					if (end2end && one2one_cv2 != null && one2one_cv3 != null)
					{
						for (int i = 0; i < this.nl; i++)
						{
							SetLastConvBias(this.one2one_cv2[i], 2.0f, count: null);
							float clsValue = (float)Math.Log(5.0 / this.nc / Math.Pow(640.0 / this.stride[i], 2));
							SetLastConvBias(this.one2one_cv3[i], clsValue, count: this.nc);
						}
					}
				}
			}


			protected internal static void SetLastConvBias(torch.nn.Module seq, float value, int? count)
			{
				Conv2d lastConv = null;
				foreach (var (name, module) in seq.named_children())
				{
					if (module is Conv2d conv)
					{
						lastConv = conv;
					}
				}
				if (lastConv?.bias is null)
				{
					throw new InvalidOperationException($"bias_init: can't find a Conv2d with bias in {seq.GetType().Name}");
				}

				using (torch.no_grad())
				{
					// [Fix 2026-11-04] BF16-safe bias write: fill_/index-fill on BFloat16 silently
					// no-ops in this TorchSharp build ("No support for BFloat16") — the good prior
					// log(5/nc/640²) never landed → epoch-1 cls loss exploded (1661). Compute in
					// Float32 (full vector when count==null, else the first `count` slots) and copy
					// back so the value survives any dtype.
					// BF16-safe pattern: fill_ silently no-ops on BF16 — build the target vector in Float32 then copy_ back. / BF16安全写法
					int n = (int)lastConv.bias.shape[0];
					int k = count is null ? n : Math.Min(count.Value, n);
					var biasF32 = lastConv.bias.to(torch.ScalarType.Float32);
					var filled = torch.full(new long[] { k }, value, torch.ScalarType.Float32, device: biasF32.device);
					if (k == n)
					{
						lastConv.bias.copy_(filled.to(lastConv.bias.dtype));
					}
					else
					{
						biasF32[torch.TensorIndex.Slice(0, k)] = filled;
						lastConv.bias.copy_(biasF32.to(lastConv.bias.dtype));
					}
				}
			}

			public virtual void one2one_init()
			{
				if (end2end)
				{
					Sequential[] cv2_copy = new Sequential[this.cv2.Count];
					cv2.CopyTo(cv2_copy, 0);
					this.one2one_cv2 = torch.nn.ModuleList(cv2_copy);

					Sequential[] cv3_copy = new Sequential[this.cv3.Count];
					cv3.CopyTo(cv3_copy, 0);
					this.one2one_cv3 = torch.nn.ModuleList(cv3_copy);

					register_module("one2one_cv2", one2one_cv2);
					register_module("one2one_cv3", one2one_cv3);
				}
			}

			public void remove_one2one()
			{
				this._internal_submodules.Remove("one2one_cv2");
				this._internal_submodules.Remove("one2one_cv3");
			}

			protected (torch.Tensor scores, torch.Tensor conf, torch.Tensor idx) get_topk_index(torch.Tensor scores, int max_det)
			{
				// i.e. shape(16,8400,84)
				long batch_size = scores.shape[0];
				long anchors = scores.shape[1];
				long nc = scores.shape[2];

				int k = (int)Math.Min(max_det, anchors);
				if (this.agnostic_nms)
				{
					(scores, torch.Tensor labels) = scores.max(dim: -1, keepdim: true);
					(scores, torch.Tensor indices) = scores.topk(k, dim: 1);
					labels = labels.gather(1, indices);
					return (scores, labels, indices);
				}
				torch.Tensor ori_index = scores.max(dim: -1).values.topk(k).indexes.unsqueeze(-1);
				scores = scores.gather(dim: 1, index: ori_index.repeat(1, 1, nc));
				(scores, torch.Tensor index) = scores.flatten(1).topk(k);
				torch.Tensor idx = ori_index[torch.arange(batch_size)[torch.TensorIndex.Ellipsis, torch.TensorIndex.None], (index / nc).@long()]; // original index
				return (scores[torch.TensorIndex.Ellipsis, torch.TensorIndex.None], (index % nc)[torch.TensorIndex.Ellipsis, torch.TensorIndex.None].@float(), idx);

			}

			// Decode bounding boxes.
			protected virtual torch.Tensor decode_bboxes(torch.Tensor bboxes, torch.Tensor anchors, bool xywh = true)
			{
				return Utils.Tal.dist2bbox(bboxes, anchors, xywh: (xywh && !this.end2end && !this.xyxy), dim: 1);
			}

			protected virtual torch.Tensor _inference(Dictionary<string, object> x)
			{
				torch.Tensor dbox = this._get_decode_boxes(x);
				return torch.cat(new torch.Tensor[] { dbox, ((torch.Tensor)x["scores"]).sigmoid() }, 1);
			}

			private torch.Tensor _get_decode_boxes(Dictionary<string, object> x)
			{
				long[] shape = ((torch.Tensor[])(x["feats"]))[0].shape; // BCHW
				if (this.dynamic || this.shape != shape)
				{
					(this.anchors, this.strides) = Utils.Tal.make_anchors((torch.Tensor[])x["feats"], this.stride, 0.5f);
					this.anchors = this.anchors.transpose(0, 1);
					this.strides = this.strides.transpose(0, 1);
					this.shape = shape;
				}

				torch.Tensor dbox = this.decode_bboxes(this.dfl.forward((torch.Tensor)x["boxes"]), this.anchors.unsqueeze(0)) * this.strides;
				return dbox;
			}

			public virtual void fuse()
			{
				this.cv2?.Clear();
				this.cv3?.Clear();
				this.cv2?.Clear();
				this.cv3?.Clear();
				this.cv2?.Dispose();
				this.cv3?.Dispose();
				this.cv2 = null;
				this.cv3 = null;
			}
		}

		internal class Segment : Detect
		{
			private readonly int nm;
			private readonly int npr;
			private readonly Block.Proto proto;
			private readonly int c4;
			private ModuleList<Sequential> cv4;
			private ModuleList<Sequential> one2one_cv4;

			internal Segment(int nc = 80, int nm = 32, int npr = 256, int reg_max = 16, int[] ch = null, bool legacy = true, bool end2end = false, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, reg_max: reg_max, end2end: end2end, ch: ch, legacy: legacy, device: device, dtype: dtype)
			{
				this.nm = nm;  // number of masks
				this.npr = npr;  // number of protos
				proto = new Block.Proto(ch[0], this.npr, this.nm, device: device, dtype: dtype);  // protos
				c4 = Math.Max(ch[0] / 4, this.nm);

				cv4 = new ModuleList<Sequential>(ch.Select(x =>
				   torch.nn.Sequential(
						new Convs.Conv(x, c4, 3, device: device, dtype: dtype),
						new Convs.Conv(c4, c4, 3, device: device, dtype: dtype),
						torch.nn.Conv2d(c4, this.nm, 1, device: device, dtype: dtype)
					)).ToArray());

				register_module("proto", this.proto);
				register_module("cv4", this.cv4);
			}

			protected override Dictionary<string, ModuleList<Sequential>> one2many()
			{
				Dictionary<string, ModuleList<Sequential>> result = new Dictionary<string, ModuleList<Sequential>>();
				result["box_head"] = this.cv2;
				result["cls_head"] = this.cv3;
				result["mask_head"] = this.cv4!;
				return result;
			}

			protected override Dictionary<string, ModuleList<Sequential>> one2one()
			{
				Dictionary<string, ModuleList<Sequential>> result = new Dictionary<string, ModuleList<Sequential>>();
				result["box_head"] = this.one2one_cv2;
				result["cls_head"] = this.one2one_cv3;
				result["mask_head"] = this.one2one_cv4!;
				return result;
			}

			public override (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds) forward(torch.Tensor[] x)
			{
				var outputs = base.forward(x);
				Dictionary<string, object> preds = outputs.preds;
				torch.Tensor proto = this.proto.forward(x[0]);
				if (preds is Dictionary<string, object>)
				{
					if (this.end2end)
					{
						((Dictionary<string, object>)preds["one2many"])["proto"] = proto;
						((Dictionary<string, object>)preds["one2one"])["proto"] = proto.detach();
					}
					else
					{
						preds["proto"] = proto;
					}
				}
				if (this.training)
				{
					return (null, preds);
				}
				outputs.inference.Add("proto", proto);

				return (outputs.inference, preds);
			}

			protected override torch.Tensor _inference(Dictionary<string, object> x)
			{
				torch.Tensor preds = base._inference(x);
				return torch.cat(new torch.Tensor[] { preds, (torch.Tensor)x["mask_coefficient"] }, dim: 1);
			}

			protected override Dictionary<string, object> forward_head(torch.Tensor[] x, Dictionary<string, ModuleList<Sequential>> heads)
			{
				Dictionary<string, object> preds = base.forward_head(x, heads);
				if (heads.TryGetValue("mask_head", out ModuleList<Sequential> mask_head))
				{
					long bs = x[0].shape[0];  // batch size
					preds["mask_coefficient"] = torch.cat(Enumerable.Range(0, this.nl).Select(i => mask_head[i].forward(x[i]).view(bs, this.nm, -1)).ToArray(), 2);
				}
				return preds;
			}

			protected override torch.Tensor postprocess(torch.Tensor preds)
			{
				using (torch.NewDisposeScope())
				{
					torch.Tensor[] box_scores_mask_cofficient = preds.split(new long[] { 4, this.nc, this.nm }, dim: -1);
					torch.Tensor boxes = box_scores_mask_cofficient[0];
					torch.Tensor scores = box_scores_mask_cofficient[1];
					torch.Tensor mask_coefficient = box_scores_mask_cofficient[2];
					(scores, torch.Tensor conf, torch.Tensor idx) = this.get_topk_index(scores, this.max_det);
					boxes = boxes.gather(dim: 1, index: idx.repeat(1, 1, 4));
					mask_coefficient = mask_coefficient.gather(dim: 1, index: idx.repeat(1, 1, this.nm));
					return torch.cat(new torch.Tensor[] { boxes, scores, conf, mask_coefficient }, dim: -1).MoveToOuterDisposeScope();
				}
			}

			public override void one2one_init()
			{
				base.one2one_init();
				if (end2end)
				{
					Sequential[] cv4_copy = new Sequential[this.cv4.Count];
					cv4.CopyTo(cv4_copy, 0);
					this.one2one_cv4 = torch.nn.ModuleList(cv4_copy);

					register_module("one2one_cv4", one2one_cv4);
				}
				else
				{
					this.one2one_cv4 = null;
				}

			}

			public override void fuse()
			{
				this.cv2?.Clear();
				this.cv3?.Clear();
				this.cv4?.Clear();
				this.cv2?.Clear();
				this.cv3?.Clear();
				this.cv4?.Clear();
				this.cv2?.Dispose();
				this.cv3?.Dispose();
				this.cv4?.Dispose();
				this.cv2 = null;
				this.cv3 = null;
				this.cv4 = null;
			}
		}

		internal class Obb : Detect
		{
			private readonly int ne;
			private ModuleList<Sequential>? cv4;
			private ModuleList<Sequential>? one2one_cv4;
			private torch.Tensor angle;

			internal Obb(int nc = 80, int ne = 1, int reg_max = 16, int[] ch = null, bool legacy = true, bool end2end = false, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, reg_max: reg_max, end2end: end2end, legacy: legacy, ch: ch, device: device, dtype: dtype)
			{
				this.ne = ne;  // number of extra parameters

				int c4 = Math.Max(ch[0] / 4, this.ne);
				this.cv4 = new ModuleList<Sequential>(ch.Select(x => torch.nn.Sequential(new Convs.Conv(x, c4, 3, device: device, dtype: dtype), new Convs.Conv(c4, c4, 3, device: device, dtype: dtype), torch.nn.Conv2d(c4, this.ne, 1, device: device, dtype: dtype))).ToArray());
				register_module("cv4", this.cv4);
			}

			protected override Dictionary<string, ModuleList<Sequential>> one2many()
			{
				Dictionary<string, ModuleList<Sequential>> result = new Dictionary<string, ModuleList<Sequential>>();
				result["box_head"] = this.cv2;
				result["cls_head"] = this.cv3;
				result["angle_head"] = this.cv4!;
				return result;
			}

			protected override Dictionary<string, ModuleList<Sequential>> one2one()
			{
				Dictionary<string, ModuleList<Sequential>> result = new Dictionary<string, ModuleList<Sequential>>();
				result["box_head"] = this.one2one_cv2;
				result["cls_head"] = this.one2one_cv3;
				result["angle_head"] = this.one2one_cv4!;
				return result;
			}

			protected override torch.Tensor _inference(Dictionary<string, object> x)
			{
				using (torch.NewDisposeScope())
				{
					this.angle = (torch.Tensor)x["angle"];
					torch.Tensor preds = base._inference(x);
					return torch.cat(new torch.Tensor[] { preds, (torch.Tensor)x["angle"] }, dim: 1).MoveToOuterDisposeScope();
				}
			}

			protected override Dictionary<string, object> forward_head(torch.Tensor[] x, Dictionary<string, ModuleList<Sequential>> heads)
			{
				var preads = base.forward_head(x, heads);
				if (heads.TryGetValue("angle_head", out ModuleList<Sequential>? angle_head))
				{
					long bs = x[0].shape[0];

					torch.Tensor local_angle = torch.cat(Enumerable.Range(0, this.nl).Select(i => angle_head[i].forward(x[i]).view(bs, this.ne, -1)).ToArray(), 2);
					local_angle = (local_angle.sigmoid() - 0.25) * Math.PI;  // [-pi/4, 3pi/4]
					preads["angle"] = local_angle;
				}
				return preads;
			}

			protected override torch.Tensor decode_bboxes(torch.Tensor bboxes, torch.Tensor anchors, bool xywh = true)
			{
				return Utils.Tal.dist2rbox(bboxes, this.angle, anchors, dim: 1);
			}

			protected override torch.Tensor postprocess(torch.Tensor preds)
			{
				using (torch.NewDisposeScope())
				{
					torch.Tensor[] boxes_scores_angle = preds.split(new long[] { 4, this.nc, this.ne }, dim: -1);
					torch.Tensor boxes = boxes_scores_angle[0];
					torch.Tensor scores = boxes_scores_angle[1];
					torch.Tensor angle = boxes_scores_angle[2];
					(scores, torch.Tensor conf, torch.Tensor idx) = this.get_topk_index(scores, this.max_det);
					boxes = boxes.gather(dim: 1, index: idx.repeat(1, 1, 4));
					angle = angle.gather(dim: 1, index: idx.repeat(1, 1, this.ne));
					return torch.cat(new torch.Tensor[] { boxes, scores, conf, angle }, dim: -1).MoveToOuterDisposeScope();
				}
			}

			public override void one2one_init()
			{
				base.one2one_init();
				if (end2end)
				{
					Sequential[] cv4_copy = new Sequential[this.cv4.Count];
					cv4.CopyTo(cv4_copy, 0);
					this.one2one_cv4 = torch.nn.ModuleList(cv4_copy);

					register_module("one2one_cv4", one2one_cv4);
				}
				else
				{
					this.one2one_cv4 = null;
				}
			}

			public override void fuse()
			{
				this.cv2?.Clear();
				this.cv3?.Clear();
				this.cv4?.Clear();
				this.cv2?.Dispose();
				this.cv3?.Dispose();
				this.cv4?.Dispose();
				this.cv2 = null;
				this.cv3 = null;
				this.cv4 = null;
			}
		}

		internal class Pose : Detect
		{
			private readonly int keypoint_num;
			private readonly int keypoint_dim;
			private readonly int nk;
			private ModuleList<Sequential> cv4;
			private ModuleList<Sequential> one2one_cv4;

			internal Pose(int nc = 1, int keypoint_num = 17, int keypoint_dim = 3, int reg_max = 16, int[] ch = null, bool legacy = true, bool end2end = false, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, reg_max: reg_max, end2end: end2end, ch: ch, legacy: legacy, device: device, dtype: dtype)
			{
				this.keypoint_num = keypoint_num;
				this.keypoint_dim = keypoint_dim;
				this.nk = keypoint_num * keypoint_dim;
				int c4 = Math.Max(ch[0] / 4, this.nk);
				this.cv4 = new ModuleList<Sequential>(ch.Select(x => torch.nn.Sequential(new Convs.Conv(x, c4, 3, device: device, dtype: dtype), new Convs.Conv(c4, c4, 3, device: device, dtype: dtype), torch.nn.Conv2d(c4, this.nk, 1, device: device, dtype: dtype))).ToArray());
				register_module("cv4", this.cv4);
			}

			protected override Dictionary<string, ModuleList<Sequential>> one2many()
			{
				Dictionary<string, ModuleList<Sequential>> result = new Dictionary<string, ModuleList<Sequential>>();
				result["box_head"] = this.cv2;
				result["cls_head"] = this.cv3;
				result["pose_head"] = this.cv4;
				return result;
			}

			protected override Dictionary<string, ModuleList<Sequential>> one2one()
			{
				Dictionary<string, ModuleList<Sequential>> result = new Dictionary<string, ModuleList<Sequential>>();
				result["box_head"] = this.one2one_cv2;
				result["cls_head"] = this.one2one_cv3;
				result["pose_head"] = this.one2one_cv4;
				return result;
			}

			/// <summary>
			/// Decode predicted bounding boxes and class probabilities, concatenated with keypoints.
			/// </summary>
			/// <param name="x"></param>
			/// <returns></returns>
			protected override torch.Tensor _inference(Dictionary<string, object> x)
			{
				torch.Tensor preds = base._inference(x);
				return torch.cat(new torch.Tensor[] { preds, this.kpts_decode((torch.Tensor)x["kpts"]) }, dim: 1);
			}

			protected override Dictionary<string, object> forward_head(torch.Tensor[] x, Dictionary<string, ModuleList<Sequential>> heads)
			{
				Dictionary<string, object> preds = base.forward_head(x, heads);
				if (heads.TryGetValue("pose_head", out ModuleList<Sequential>? pose_head))
				{
					long bs = x[0].shape[0];  // batch size

					//preds["kpts"] = torch.cat([pose_head[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)
					preds["kpts"] = torch.cat(Enumerable.Range(0, this.nl).Select(i => pose_head[i].forward(x[i]).view(bs, this.nk, -1)).ToArray(), 2);
				}
				return preds;
			}

			/// <summary>
			/// Post-process YOLO model predictions.
			/// </summary>
			/// <param name="preds"> Raw predictions with shape (batch_size, num_anchors, 4 + nc + nk) with last dimension format[x1, y1, x2, y2, class_probs, keypoints].</param>
			/// <returns>Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + self.nk) and last dimension format[x1, y1, x2, y2, max_class_prob, class_index, keypoints].</returns>
			protected override torch.Tensor postprocess(torch.Tensor preds)
			{
				using (torch.NewDisposeScope())
				{
					torch.Tensor[] boxes_scores_kpts = preds.split(new long[] { 4, this.nc, this.nk }, dim: -1);
					torch.Tensor boxes = boxes_scores_kpts[0];
					torch.Tensor scores = boxes_scores_kpts[1];
					torch.Tensor kpts = boxes_scores_kpts[2];
					(scores, torch.Tensor conf, torch.Tensor idx) = this.get_topk_index(scores, this.max_det);
					boxes = boxes.gather(dim: 1, index: idx.repeat(1, 1, 4));
					kpts = kpts.gather(dim: 1, index: idx.repeat(1, 1, this.nk));
					return torch.cat(new torch.Tensor[] { boxes, scores, conf, kpts }, dim: -1).MoveToOuterDisposeScope();
				}
			}

			public override void one2one_init()
			{
				base.one2one_init();
				if (end2end)
				{
					Sequential[] cv4_copy = new Sequential[this.cv4.Count];
					cv4.CopyTo(cv4_copy, 0);
					this.one2one_cv4 = torch.nn.ModuleList(cv4_copy);

					register_module("one2one_cv4", one2one_cv4);
				}
				else
				{
					this.one2one_cv4 = null;
				}
			}

			public override void fuse()
			{
				this.cv2?.Clear();
				this.cv3?.Clear();
				this.cv4?.Clear();
				this.cv2?.Dispose();
				this.cv3?.Dispose();
				this.cv4?.Dispose();
				this.cv2 = null;
				this.cv3 = null;
				this.cv4 = null;
			}

			private torch.Tensor kpts_decode(torch.Tensor kpts)
			{
				long ndim = this.keypoint_dim;
				long bs = kpts.shape[0];

				torch.Tensor y = kpts.clone();
				if (ndim == 3)
				{
					y[torch.TensorIndex.Colon, torch.TensorIndex.Slice(2, null, ndim)] = y[torch.TensorIndex.Colon, torch.TensorIndex.Slice(2, null, ndim)].sigmoid_();

				}
				y[torch.TensorIndex.Colon, torch.TensorIndex.Slice(0, null, ndim)] = (y[torch.TensorIndex.Colon, torch.TensorIndex.Slice(0, null, ndim)] * 2.0 + (this.anchors[0] - 0.5)) * this.strides;
				y[torch.TensorIndex.Colon, torch.TensorIndex.Slice(1, null, ndim)] = (y[torch.TensorIndex.Colon, torch.TensorIndex.Slice(1, null, ndim)] * 2.0 + (this.anchors[1] - 0.5)) * this.strides;
				return y;
			}
		}

		internal class Classify : torch.nn.Module<torch.Tensor[], (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds)>
		{
			private readonly Convs.Conv conv;
			private readonly AdaptiveAvgPool2d pool;
			private readonly Dropout drop;
			private readonly Linear linear;

			internal Classify(int c1, int c2, int k = 1, int s = 1, int? p = null, int g = 1, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(Classify))
			{
				int c_ = 1280;  // efficientnet_b0 size
				this.conv = new Convs.Conv(c1, c_, k, s, p, g, device: device, dtype: dtype);
				this.pool = torch.nn.AdaptiveAvgPool2d(1);  // to x(b,c_,1,1)
				this.drop = torch.nn.Dropout(p: 0.0, inplace: true);
				this.linear = torch.nn.Linear(c_, c2, device: device, dtype: dtype);  // to x(b,c2)
				RegisterComponents();
			}

			public override (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds) forward(torch.Tensor[] input)
			{
				torch.Tensor x = torch.cat(input, dim: 1);
				x = this.linear.forward(this.drop.forward(this.pool.forward(this.conv.forward(x)).flatten(1)));
				Dictionary<string, object> cls = new Dictionary<string, object>();
				cls["cls"] = x;
				if (this.training)
				{
					return (null, cls);
				}
				torch.Tensor y = x.softmax(1);
				Dictionary<string, torch.Tensor> inference = new Dictionary<string, torch.Tensor>();
				inference["cls"] = y;
				return (inference, cls);
			}
		}

		// ═══════════════════════════════════════════════════════════════════════════
		// [Add 2026-10-30] Below heads mirror ultralytics head.py order (Segment26/OBB26/Pose26/
		// Depth/WorldDetect/LRPCHead/YOLOEDetect/YOLOESegment/YOLOESegment26/RTDETRDecoder/
		// v10Detect/SemanticSegment). These task families (YOLO26/YOLO-E/RT-DETR/Depth/World) are
		// not yet exercised by the YoloSharp training/prediction pipeline — constructors keep the
		// full ultralytics docstrings; heads that depend on unported ecosystems throw
		// NotSupportedException on construction.
		// ═══════════════════════════════════════════════════════════════════════════

		/// <summary>
		/// Ultralytics YOLO26 models segment head for segmentation models.
		/// </summary>
		internal class Segment26 : Segment
		{
			/// <summary>
			/// Initialize the YOLO26 segment head.
			///
			/// Args:
			///     nc (int): Number of classes.
			///     nm (int): Number of masks.
			/// </summary>
			internal Segment26(int nc = 80, int nm = 32, int npr = 256, int reg_max = 16, int[]? ch = null, bool legacy = true, torch.Device? device = null, torch.ScalarType? dtype = null)
				: base(nc, nm, npr, reg_max, ch, legacy, device: device, dtype: dtype)
			{
			}
		}

		/// <summary>
		/// Ultralytics YOLO26 models OBB head for oriented bounding box detection.
		/// </summary>
		internal class OBB26 : Obb
		{
			/// <summary>
			/// Initialize the YOLO26 OBB head.
			///
			/// Args:
			///     nc (int): Number of classes.
			/// </summary>
			internal OBB26(int nc = 80, int ne = 1, int reg_max = 16, int[]? ch = null, bool legacy = true, torch.Device? device = null, torch.ScalarType? dtype = null)
				: base(nc, ne, reg_max, ch, legacy, device: device, dtype: dtype)
			{
			}
		}

		/// <summary>
		/// Ultralytics YOLO26 models pose head for keypoints detection.
		/// </summary>
		internal class Pose26 : Pose
		{
			/// <summary>
			/// Initialize the YOLO26 pose head.
			///
			/// Args:
			///     nc (int): Number of classes.
			///     kpt_shape (tuple): Shape of keypoints (num_kpts, dim).
			/// </summary>
			internal Pose26(int nc = 1, int keypoint_num = 17, int keypoint_dim = 3, int reg_max = 16, int[]? ch = null, bool legacy = true, torch.Device? device = null, torch.ScalarType? dtype = null)
				: base(nc, keypoint_num, keypoint_dim, reg_max, ch, legacy, device: device, dtype: dtype)
			{
			}
		}

		/// <summary>
		/// Depth head for depth estimation models (ultralytics YOLO11-Depth family).
		/// </summary>
		internal class Depth : torch.nn.Module<torch.Tensor[], (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds)>
		{
			private readonly Convs.Conv cv2;

			/// <summary>
			/// Initialize the Depth head.
			///
			/// Args:
			///     ch (int): Input channels from the final backbone stage.
			/// </summary>
			internal Depth(int ch = 512, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(Depth))
			{
				this.cv2 = new Convs.Conv(ch, 1, 1, device: device, dtype: dtype);
				RegisterComponents();
			}

			public override (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds) forward(torch.Tensor[] input)
			{
				torch.Tensor x = input[input.Length - 1];
				x = this.cv2.forward(x);
				Dictionary<string, object> preds = new Dictionary<string, object>();
				preds["depth"] = x;
				Dictionary<string, torch.Tensor> inference = new Dictionary<string, torch.Tensor>();
				inference["depth"] = x.sigmoid();
				return (inference, preds);
			}
		}

		/// <summary>
		/// WorldDetect head for open-vocabulary detection using text embeddings (YOLO-World family).
		/// </summary>
		internal class WorldDetect : Detect
		{
			/// <summary>
			/// Initialize WorldDetect head.
			///
			/// Args:
			///     nc (int): Number of classes.
			///     embed (int): Embedding size for text features.
			/// </summary>
			internal WorldDetect(int nc = 80, int embed = 512, int reg_max = 16, int[]? ch = null, bool legacy = true, torch.Device? device = null, torch.ScalarType? dtype = null)
				: base(nc, reg_max, ch, legacy, device: device, dtype: dtype)
			{
				throw new NotSupportedException(
					"WorldDetect (YOLO-World open-vocabulary) is not yet supported in YoloSharp: it requires " +
					"offline text-embedding features produced by the CLIP tokenizer.");
			}
		}

		/// <summary>
		/// LRPCHead: low-rank projection head for efficient channel compression.
		/// </summary>
		internal class LRPCHead : torch.nn.Module<torch.Tensor, torch.Tensor>
		{
			private readonly TorchSharp.Modules.Linear linear;

			/// <summary>
			/// Initialize LRPCHead.
			///
			/// Args:
			///     c1 (int): Input channels.
			///     c2 (int): Output channels.
			/// </summary>
			internal LRPCHead(int c1, int c2, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(LRPCHead))
			{
				this.linear = torch.nn.Linear(c1, c2, device: device, dtype: dtype);
				RegisterComponents();
			}

			public override torch.Tensor forward(torch.Tensor x)
			{
				return this.linear.forward(x.flatten(2).transpose(1, 2)).transpose(1, 2);
			}
		}

		/// <summary>
		/// YOLOEDetect head for promptable visual detection (YOLO-E family).
		/// </summary>
		internal class YOLOEDetect : Detect
		{
			/// <summary>
			/// Initialize YOLOEDetect head.
			///
			/// Args:
			///     nc (int): Number of classes.
			/// </summary>
			internal YOLOEDetect(int nc = 80, int reg_max = 16, int[]? ch = null, bool legacy = true, torch.Device? device = null, torch.ScalarType? dtype = null)
				: base(nc, reg_max, ch, legacy, device: device, dtype: dtype)
			{
				throw new NotSupportedException(
					"YOLOEDetect (YOLO-E promptable detection) is not yet supported in YoloSharp: it requires " +
					"the YOLO-E visual-prompt embedding (SAVPE) and text branch.");
			}
		}

		/// <summary>
		/// YOLOESegment head for promptable visual segmentation (YOLO-E family).
		/// </summary>
		internal class YOLOESegment : YOLOEDetect
		{
			internal YOLOESegment(int nc = 80, int reg_max = 16, int[]? ch = null, bool legacy = true, torch.Device? device = null, torch.ScalarType? dtype = null)
				: base(nc, reg_max, ch, legacy, device: device, dtype: dtype)
			{
			}
		}

		/// <summary>
		/// YOLOESegment26 head (YOLO26 variant of YOLO-E segmentation).
		/// </summary>
		internal class YOLOESegment26 : YOLOESegment
		{
			internal YOLOESegment26(int nc = 80, int reg_max = 16, int[]? ch = null, bool legacy = true, torch.Device? device = null, torch.ScalarType? dtype = null)
				: base(nc, reg_max, ch, legacy, device: device, dtype: dtype)
			{
			}
		}

		/// <summary>
		/// RTDETRDecoder: decoder module for Real-Time DETR (RT-DETR) models.
		/// </summary>
		internal class RTDETRDecoder : torch.nn.Module
		{
			/// <summary>
			/// Initialize RTDETRDecoder.
			///
			/// Args:
			///     nc (int): Number of classes.
			///     nd (int): Number of decoder layers.
			/// </summary>
			internal RTDETRDecoder(int nc = 80, int nd = 6, torch.Device? device = null, torch.ScalarType? dtype = null)
				: base(nameof(RTDETRDecoder))
			{
				throw new NotSupportedException(
					"RTDETRDecoder is not yet supported in YoloSharp: requires the Deformable-DETR training " +
					"objective (Hungarian matching with denoising groups) which is not ported.");
			}
		}

		/// <summary>
		/// v10Detect head supporting end-to-end NMS-free detection (YOLOv10 family).
		/// </summary>
		internal class v10Detect : Detect
		{
			/// <summary>
			/// Initialize v10Detect head.
			///
			/// Args:
			///     nc (int): Number of classes.
			/// </summary>
			internal v10Detect(int nc = 80, int reg_max = 16, int[]? ch = null, bool legacy = true, torch.Device? device = null, torch.ScalarType? dtype = null)
				: base(nc, reg_max, ch, legacy, device: device, dtype: dtype)
			{
				// YOLOv10 one-to-one head branches are additional registered modules in ultralytics;
				// YoloSharp inference path treats v10 models through the standard Detect head.
			}
		}

		/// <summary>
		/// SemanticSegment head for semantic segmentation models.
		/// </summary>
		internal class SemanticSegment : torch.nn.Module<torch.Tensor[], (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds)>
		{
			private readonly TorchSharp.Modules.Conv2d seg_head;

			/// <summary>
			/// Initialize SemanticSegment head.
			///
			/// Args:
			///     c1 (int): Input channels.
			///     nc (int): Number of classes.
			/// </summary>
			internal SemanticSegment(int c1 = 512, int nc = 80, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(SemanticSegment))
			{
				this.seg_head = torch.nn.Conv2d(c1, nc, 1, device: device, dtype: dtype);
				RegisterComponents();
			}

			public override (Dictionary<string, torch.Tensor> inference, Dictionary<string, object> preds) forward(torch.Tensor[] input)
			{
				torch.Tensor x = input[input.Length - 1];
				x = this.seg_head.forward(x);
				Dictionary<string, object> preds = new Dictionary<string, object>();
				preds["semseg"] = x;
				Dictionary<string, torch.Tensor> inference = new Dictionary<string, torch.Tensor>();
				inference["semseg"] = x;
				return (inference, preds);
			}
		}
	}
}
