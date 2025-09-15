using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torchvision;

namespace Utils
{
	internal class Loss
	{
		/// <summary>
		/// Returns label smoothing BCE targets for reducing overfitting
		/// </summary>
		/// <param name="eps"></param>
		/// <returns>pos: `1.0 - 0.5*eps`, neg: `0.5*eps`.</returns>
		private static (float pos, float neg) Smooth_BCE(float eps = 0.1f)
		{
			// For details see https://github.com/ultralytics/yolov3/issues/238;  //issuecomment-598028441"""
			return (1.0f - 0.5f * eps, 0.5f * eps);
		}

		private class BCEBlurWithLogitsLoss : Module<Tensor, Tensor, Tensor>
		{
			private readonly BCEWithLogitsLoss loss_fcn;
			private readonly float alpha;
			public BCEBlurWithLogitsLoss(float alpha = 0.05f, Reduction reduction = Reduction.None) : base(nameof(BCEBlurWithLogitsLoss))
			{
				this.loss_fcn = BCEWithLogitsLoss(reduction: reduction);  // must be nn.BCEWithLogitsLoss()
				this.alpha = alpha;
			}

			public override Tensor forward(Tensor pred, Tensor t)
			{
				using (NewDisposeScope())
				{
					Tensor loss = loss_fcn.forward(pred, t);
					pred = sigmoid(pred);  // prob from logits
					Tensor dx = pred - t;// ;  // reduce only missing label effects
										 // dx = (pred - true).abs()  ;  // reduce missing label and false label effects
					Tensor alpha_factor = 1 - exp((dx - 1) / (alpha + 1e-4));
					loss *= alpha_factor;
					return loss.mean();
				}
			}
		}

		private class FocalLoss : Module<Tensor, Tensor, Tensor>
		{
			private readonly BCEWithLogitsLoss loss_fcn;
			private readonly float alpha;
			private readonly float gamma;
			private Reduction reduction;
			public FocalLoss(BCEWithLogitsLoss loss_fcn, float gamma = 1.5f, float alpha = 0.25f) : base(nameof(FocalLoss))
			{
				this.loss_fcn = loss_fcn;  // must be nn.BCEWithLogitsLoss()
				this.gamma = gamma;
				this.alpha = alpha;
				reduction = loss_fcn.reduction;
			}

			public override Tensor forward(Tensor pred, Tensor t)
			{
				using (NewDisposeScope())
				{
					Tensor loss = loss_fcn.forward(pred, t);
					Tensor pred_prob = sigmoid(pred);  // prob from logits
					Tensor p_t = true * pred_prob + (1 - t) * (1 - pred_prob);
					Tensor alpha_factor = t * alpha + (1 - t) * (1 - alpha);
					Tensor modulating_factor = (1.0 - p_t).pow(gamma);

					loss *= alpha_factor * modulating_factor;

					loss = reduction switch
					{
						Reduction.Mean => loss.mean(),
						Reduction.Sum => loss.sum(),
						Reduction.None => loss,
						_ => loss
					};
					return loss.MoveToOuterDisposeScope();
				}
			}
		}

		private class DFLoss : Module<Tensor, Tensor, Tensor>
		{
			private readonly int reg_max;
			public int regMax => reg_max;

			public DFLoss(int reg_max = 16) : base(nameof(DFLoss))
			{
				this.reg_max = reg_max;
			}

			public override Tensor forward(Tensor pred_dist, Tensor target)
			{
				using (NewDisposeScope())
				{
					target = target.clamp_(0, reg_max - 1 - 0.01);

					Tensor tl = target.@long(); // target left
					Tensor tr = tl + 1; //target right
					Tensor wl = tr - target; //weight left
					Tensor wr = 1 - wl; //weight right
					return (
						functional.cross_entropy(pred_dist, tl.view(-1), reduction: Reduction.None).view(tl.shape) * wl
						+ functional.cross_entropy(pred_dist, tr.view(-1), reduction: Reduction.None).view(tl.shape) * wr
					).mean(new long[] { -1 }, keepdim: true).MoveToOuterDisposeScope();
				}
			}
		}

		private class BboxLoss : Module
		{
			protected readonly DFLoss? dfl_loss;
			protected readonly int reg_max;

			public BboxLoss(int regMax = 16) : base(nameof(BboxLoss))
			{
				dfl_loss = regMax > 1 ? new DFLoss(regMax) : null;
				reg_max = regMax;
			}

			public virtual (Tensor loss_iou, Tensor loss_dfl) forward(Tensor pred_dist, Tensor pred_bboxes, Tensor anchor_points, Tensor target_bboxes, Tensor target_scores, Tensor target_scores_sum, Tensor fg_mask)
			{
				using (NewDisposeScope())
				{
					// Step 1: Compute weight
					Tensor weight = target_scores.sum(new long[] { -1 })[fg_mask].unsqueeze(-1);

					// Step 2: Compute IoU
					Tensor iou = Utils.Metrics.bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], false, true);
					Tensor lossIou = ((1.0 - iou) * weight).sum() / target_scores_sum;

					// Step 3: Compute DFL loss
					Tensor lossDfl;
					if (dfl_loss is not null)
					{
						Tensor targetLtrb = Utils.Tal.bbox2dist(anchor_points, target_bboxes, reg_max - 1);
						lossDfl = dfl_loss.forward(pred_dist[fg_mask].view(-1, reg_max), targetLtrb[fg_mask]) * weight;
						lossDfl = lossDfl.sum() / target_scores_sum;
					}
					else
					{
						lossDfl = tensor(0.0, device: pred_dist.device);
					}

					return (lossIou.MoveToOuterDisposeScope(), lossDfl.MoveToOuterDisposeScope());
				}
			}
		}

		private class RotatedBboxLoss : BboxLoss
		{
			internal RotatedBboxLoss(int regMax) : base(regMax: regMax)
			{

			}

			public override (Tensor loss_iou, Tensor loss_dfl) forward(Tensor pred_dist, Tensor pred_bboxes, Tensor anchor_points, Tensor target_bboxes, Tensor target_scores, Tensor target_scores_sum, Tensor fg_mask)
			{
				using (NewDisposeScope())
				{
					Tensor weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1);
					Tensor iou = Metrics.probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask]);
					Tensor loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum;
					Tensor loss_dfl = torch.zeros(0);
					// DFL loss
					if (dfl_loss is not null)
					{
						Tensor target_ltrb = Utils.Tal.bbox2dist(anchor_points, torchvision.ops.box_convert(target_bboxes[.., ..4], ops.BoxFormats.xywh, ops.BoxFormats.xyxy), this.dfl_loss.regMax - 1);
						loss_dfl = this.dfl_loss.forward(pred_dist[fg_mask].view(-1, this.dfl_loss.regMax), target_ltrb[fg_mask]) * weight;
						loss_dfl = loss_dfl.sum() / target_scores_sum;
					}
					else
					{
						loss_dfl = torch.tensor(0.0).to(pred_dist.device);
					}
					return (loss_iou.MoveToOuterDisposeScope(), loss_dfl.MoveToOuterDisposeScope());
				}
			}
		}

		internal class V5DetectionLoss : Module<Tensor[], Tensor, (Tensor, Tensor)>
		{
			private readonly float lambda_coord = 5.0f;
			private readonly float lambda_noobj = 0.5f;
			private readonly float cp;
			private readonly float cn;
			private float[] balance;
			private readonly int ssi;
			private readonly float gr;
			private readonly bool autobalance;
			private readonly int na;
			private readonly int nc;
			private readonly int nl;
			private readonly float[][] anchors;

			private Device device = new Device(TorchSharp.DeviceType.CPU);
			private torch.ScalarType dtype = torch.ScalarType.Float32;

			private readonly float anchor_t = 4.0f;
			private readonly bool sort_obj_iou = false;
			private readonly float h_box = 0.05f;
			private readonly float h_obj = 1.0f;
			private readonly float h_cls = 0.5f;
			private readonly float h_cls_pw = 1.0f;
			private readonly float h_obj_pw = 1.0f;
			private readonly float fl_gamma = 0.0f;
			private readonly float h_label_smoothing = 0.0f;

			public V5DetectionLoss(int nc = 80, bool autobalance = false) : base("V5DetectionLoss")
			{
				int model_nl = 3;
				int[] model_stride = new int[] { 8, 16, 32 };
				float p3_d = 8.0f;
				float p4_d = 16.0f;
				float p5_d = 32.0f;
				float[][] anchors = new float[][] {
					new float[]{ 10/p3_d, 13 / p3_d, 16 / p3_d, 30 / p3_d, 33 / p3_d, 23/p3_d },
					new float[]{   30/p4_d, 61 / p4_d, 62 / p4_d, 45 / p4_d, 59 / p4_d, 119/p4_d},
					new float[]{116/p5_d, 90 / p5_d, 156 / p5_d, 198 / p5_d, 373 / p5_d, 326/p5_d }};

				(cp, cn) = Smooth_BCE(h_label_smoothing);
				balance = model_nl == 3 ? new float[] { 4.0f, 1.0f, 0.4f } : new float[] { 4.0f, 1.0f, 0.25f, 0.06f, 0.02f };
				ssi = autobalance ? model_stride.ToList().IndexOf(16) : 0;
				gr = 1.0f;
				this.autobalance = autobalance;

				nl = anchors.Length;
				na = anchors[0].Length / 2; // =3 获得每个grid的anchor数量
				this.nc = nc; // number of classes
				this.anchors = anchors;
			}

			public override (Tensor, Tensor) forward(Tensor[] preds, Tensor targets)
			{
				device = targets.device;
				dtype = targets.dtype;

				var BCEcls = BCEWithLogitsLoss(pos_weights: tensor(new float[] { h_cls_pw }, device: device));
				var BCEobj = BCEWithLogitsLoss(pos_weights: tensor(new float[] { h_obj_pw }, device: device));

				//var BCEcls = new FocalLoss(BCEWithLogitsLoss(pos_weights: torch.tensor(new float[] { h_cls_pw }, device: this.device)), fl_gamma);
				//var BCEobj = new FocalLoss(BCEWithLogitsLoss(pos_weights: torch.tensor(new float[] { h_obj_pw }, device: this.device)), fl_gamma);

				var lcls = zeros(1, device: device, dtype: float32);  // class loss
				var lbox = zeros(1, device: device, dtype: float32);  // box loss
				var lobj = zeros(1, device: device, dtype: float32);  // object loss

				var (tcls, tbox, indices, anchors) = build_targets(preds, targets);
				Tensor tobj = zeros(0);
				for (int i = 0; i < preds.Length; i++)
				{
					var pi = preds[i].clone();
					var b = indices[i][0];
					var a = indices[i][1];
					var gj = indices[i][2];
					var gi = indices[i][3];
					tobj = zeros(preds[i].shape.Take(4).ToArray(), device: device, dtype: dtype);  // targets obj
					long n = b.shape[0];
					if (n > 0)
					{
						var temp = pi[b, a, gj, gi].split(new long[] { 2, 2, 1, nc }, 1);
						var pxy = temp[0];
						var pwh = temp[1];
						var pcls = temp[3];

						pxy = pxy.sigmoid() * 2 - 0.5f;
						pwh = (pwh.sigmoid() * 2).pow(2) * anchors[i];
						var pbox = cat(new Tensor[] { pxy, pwh }, 1);  // predicted box
						var iou = Utils.Metrics.bbox_iou(pbox, tbox[i], CIoU: true).squeeze();  // iou(prediction, targets)
						lbox += (1.0f - iou).mean();  // iou loss

						// Objectness
						iou = iou.detach().clamp(0).type(tobj.dtype);

						if (sort_obj_iou)
						{
							var j = iou.argsort();
							(b, a, gj, gi, iou) = (b[j], a[j], gj[j], gi[j], iou[j]);
						}
						if (gr < 1)
						{
							iou = 1.0f - gr + gr * iou;
						}

						tobj[b, a, gj, gi] = iou;   // iou ratio

						// Classification
						if (nc > 1)  // cls loss (only if multiple classes)
						{
							var tt = full_like(pcls, cn, device: device, dtype: torch.ScalarType.Float32);  // targets
							tt[arange(n), tcls[i]] = cp;
							lcls += BCEcls.forward(pcls, tt.to(dtype));  // BCE
						}

					}

					var obji = BCEobj.forward(pi[TensorIndex.Ellipsis, 4], tobj);
					lobj += obji * balance[i];  // obj loss
					if (autobalance)
					{
						balance[i] = balance[i] * 0.9999f + 0.0001f / obji.detach().item<float>();
					}
				}
				if (autobalance)
				{
					for (int i = 0; i < balance.Length; i++)
					{
						balance[i] = balance[i] / balance[ssi];
					}
				}

				lbox *= h_box;
				lobj *= h_obj;
				lcls *= h_cls;
				long bs = tobj.shape[0];  // batch size

				return ((lbox + lobj + lcls) * bs, cat(new Tensor[] { lbox, lobj, lcls }).detach());
			}

			private (List<Tensor>, List<Tensor>, List<List<Tensor>>, List<Tensor>) build_targets(Tensor[] p, Tensor targets)
			{
				var tcls = new List<Tensor>();
				var tbox = new List<Tensor>();
				var indices = new List<List<Tensor>>();
				var anch = new List<Tensor>();
				int na = this.na;
				int nt = (int)targets.shape[0];  // number of anchors, targets
												 //tcls, tbox, indices, anch = [], [], [], []
				var gain = ones(7, device: device, dtype: dtype);// normalized to gridspace gain
				var ai = arange(na, device: device, dtype: dtype).view(na, 1).repeat(1, nt);  // same as .repeat_interleave(nt)
				targets = cat(new Tensor[] { targets.repeat(na, 1, 1), ai.unsqueeze(-1) }, 2);// append anchor indices


				float g = 0.5f;  // bias
				var off = tensor(new int[,] { { 0, 0 }, { 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 } }, device: device) * g;
				for (int i = 0; i < nl; i++)
				{
					Tensor anchors = this.anchors[i];
					anchors = anchors.view(3, 2).to(dtype, device);
					var shape = p[i].shape;
					var temp = tensor(new float[] { shape[3], shape[2], shape[3], shape[2] }, device: device, dtype: dtype);

					gain.index_put_(temp, new long[] { 2, 3, 4, 5 });
					var t = targets * gain;
					Tensor offsets = zeros(0, device: device);
					if (nt != 0)
					{
						var r = t[TensorIndex.Ellipsis, TensorIndex.Slice(4, 6)] / anchors.unsqueeze(1);
						var j = max(r, 1 / r).max(2).values < anchor_t;  // compare
						t = t[j];  //filter
						var gxy = t[TensorIndex.Ellipsis, TensorIndex.Slice(2, 4)];   // grid xy
						var gxi = gain[TensorIndex.Ellipsis, TensorIndex.Slice(2, 4)] - gxy; // inverse
						Tensor jk = (gxy % 1 < g & gxy > 1).T;
						j = jk[0];
						var k = jk[1];
						Tensor lm = (gxi % 1 < g & gxi > 1).T;
						var l = lm[0];
						var m = lm[1];
						j = stack(new Tensor[] { ones_like(j), j, k, l, m });
						t = t.repeat(new long[] { 5, 1, 1 })[j];
						offsets = (zeros_like(gxy).unsqueeze(0) + off.unsqueeze(1))[j];
					}
					else
					{
						t = targets[0];
						offsets = zeros(1);
					}

					Tensor[] ck = t.chunk(4, 1); // (image, class), grid xy, grid wh, anchors
					var bc = ck[0];
					var gxy_ = ck[1];
					var gwh = ck[2];
					var a = ck[3];

					a = a.@long().view(-1);
					bc = bc.@long().T; // anchors, image, class
					Tensor b = bc[0];
					Tensor c = bc[1];

					var gij = (gxy_ - offsets).@long();// grid indices
					var gi = gij.T[0];
					var gj = gij.T[1];

					indices.Add(new List<Tensor> { b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1) });// image, anchor, grid
					tbox.Add(cat(new Tensor[] { gxy_ - gij, gwh }, 1));  // box
					anch.Add(anchors[a]); // anchors
					tcls.Add(c);// class
				}
				return (tcls, tbox, indices, anch);

			}

		}

		internal class V8DetectionLoss : Module<Tensor[], Tensor, (Tensor, Tensor)>
		{
			protected readonly int[] stride;
			protected readonly int nc;
			protected readonly int no;
			protected readonly int reg_max;
			private readonly int tal_topk;
			protected Device? device;
			protected torch.ScalarType? dtype;
			private readonly bool use_dfl;
			private readonly float hyp_box;
			private readonly float hyp_cls;
			private readonly float hyp_dfl;

			protected readonly BCEWithLogitsLoss bce;
			private readonly Utils.Tal.TaskAlignedAssigner assigner;
			private readonly BboxLoss bbox_loss;
			protected readonly Tensor proj;

			public V8DetectionLoss(int nc = 80, int reg_max = 16, int tal_topk = 10, int[]? stride = null, float hyp_box = 7.5f, float hyp_cls = 0.5f, float hyp_dfl = 1.5f) : base(nameof(V8DetectionLoss))
			{
				this.stride = stride is null ? new int[] { 8, 16, 32 } : stride;
				this.bce = BCEWithLogitsLoss(reduction: Reduction.None);
				this.nc = nc; // number of classes
				this.no = nc + reg_max * 4;
				this.reg_max = reg_max;
				this.use_dfl = reg_max > 1;
				this.tal_topk = tal_topk;
				this.hyp_box = hyp_box;
				this.hyp_cls = hyp_cls;
				this.hyp_dfl = hyp_dfl;

				this.assigner = new Utils.Tal.TaskAlignedAssigner(topk: tal_topk, num_classes: this.nc, alpha: 0.5f, beta: 6.0f);
				this.bbox_loss = new BboxLoss(this.reg_max);
				this.proj = torch.arange(this.reg_max);
			}

			public override (Tensor, Tensor) forward(Tensor[] preds, Tensor targets)
			{
				using (NewDisposeScope())
				{
					this.device = preds[0].device;
					this.dtype = preds[0].dtype;
					Tensor loss = zeros(3, device: device, dtype: float32); // box, cls, dfl
					Tensor[] feats = (Tensor[])preds.Clone();

					Tensor[] feats_mix = feats.Select(xi => xi.view(feats[0].shape[0], no, -1)).ToArray();
					Tensor[] pred_distri_scores = cat(feats_mix, 2).split(new long[] { reg_max * 4, nc }, 1);
					Tensor pred_distri = pred_distri_scores[0];
					Tensor pred_scores = pred_distri_scores[1];

					pred_scores = pred_scores.permute(0, 2, 1).contiguous();
					pred_distri = pred_distri.permute(0, 2, 1).contiguous();

					long batch_size = pred_scores.shape[0];

					Tensor imgsz = tensor(feats[0].shape[2..], device: device, dtype: dtype) * stride[0]; // image size (h,w)
					var (anchor_points, stride_tensor) = Utils.Tal.make_anchors(feats, stride, 0.5f);
					var indices = tensor(new long[] { 1, 0, 1, 0 }, device: device);

					// Targets
					var scale_tensor = index_select(imgsz, 0, indices).to(device);
					targets = this.postprocess(targets, batch_size, scale_tensor);
					var gt_labels_bboxes = targets.split(new long[] { 1, 4 }, 2); // cls, xyxy
					var gt_labels = gt_labels_bboxes[0];
					var gt_bboxes = gt_labels_bboxes[1];
					var mask_gt = gt_bboxes.sum(2, keepdim: true).gt_(0.0);

					// Pboxes
					Tensor pred_bboxes = this.bbox_decode(anchor_points, pred_distri);  // xyxy, (b, h*w, 4)

					(_, Tensor target_bboxes, Tensor target_scores, Tensor fg_mask, _) = assigner.forward(pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype), anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt);

					Tensor target_scores_sum = max(target_scores.sum());
					loss[1] = bce.forward(pred_scores, target_scores).sum() / target_scores_sum;  // BCE
																								  //float loss1 = (this.bce.forward(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum).ToSingle();  // BCE
					if (fg_mask.sum().ToInt64() > 0)
					{
						target_bboxes /= stride_tensor;
						(loss[0], loss[2]) = new BboxLoss().forward(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask);
					}
					loss[0] *= hyp_box;  // box gain
					loss[1] *= hyp_cls;// cls gain
					loss[2] *= hyp_dfl;// dfl gain
					return ((loss.sum() * batch_size).MoveToOuterDisposeScope(), loss.MoveToOuterDisposeScope());
				}
			}

			// Decode predicted object bounding box coordinates from anchor points and distribution.
			private Tensor bbox_decode(Tensor anchor_points, Tensor pred_dist)
			{
				if (use_dfl)
				{
					long b = pred_dist.shape[0], a = pred_dist.shape[1], c = pred_dist.shape[2]; // batch, anchors, channels
					pred_dist = pred_dist.view(b, a, 4, c / 4).softmax(3).matmul(this.proj.to(pred_dist.dtype, pred_dist.device));
				}
				return Utils.Tal.dist2bbox(pred_dist, anchor_points, xywh: false);
			}

			private Tensor postprocess(Tensor targets, long batch_size, Tensor scale_tensor)
			{
				using (NewDisposeScope())
				{
					// Preprocesses the target counts and matches with the input batch size to output a tensor.
					long nl = targets.shape[0];
					long ne = targets.shape[1];

					if (nl == 0)
					{
						return zeros(new long[] { batch_size, 0, ne - 1 }, device: device).MoveToOuterDisposeScope();
					}
					else
					{
						Tensor i = targets[TensorIndex.Colon, 0];  // image index
						(_, _, Tensor counts) = i.unique(return_counts: true);
						Tensor @out = zeros(new long[] { batch_size, counts.max().ToInt64(), ne - 1 }, device: device);

						for (int j = 0; j < batch_size; j++)
						{
							Tensor matches = i == j;
							long n = matches.sum().ToInt64();
							if (n > 0)
							{
								// Get the indices where matches is True
								var indices = nonzero(matches).squeeze().to(torch.ScalarType.Int64);

								// Select the rows from targets
								var selectedRows = targets.index_select(0, indices.contiguous());

								// Slice the rows to exclude the first column
								var selectedRowsSliced = selectedRows.narrow(1, 1, ne - 1);

								// Assign to the output tensor
								@out[j, TensorIndex.Slice(0, n)] = selectedRowsSliced;
							}

						}
						// Convert xywh to xyxy format and scale
						@out[TensorIndex.Ellipsis, TensorIndex.Slice(1, 5)] = Utils.Ops.xywh2xyxy(@out[TensorIndex.Ellipsis, TensorIndex.Slice(1, 5)].mul(scale_tensor));
						return @out.MoveToOuterDisposeScope();
					}
				}
			}

		}

		internal class V8SegmentationLoss : Module<Tensor[], Tensor, Tensor, (Tensor, Tensor)>
		{
			private readonly int[] stride;
			private readonly int nc;
			private readonly int no;
			private readonly int reg_max;
			private readonly int tal_topk;
			private Device device;
			private torch.ScalarType dtype;
			private readonly bool use_dfl;

			private readonly BCEWithLogitsLoss bce;

			private readonly float hyp_box = 7.5f;
			private readonly float hyp_cls = 0.5f;
			private readonly float hyp_dfl = 1.5f;
			private readonly bool over_laps = true;

			public V8SegmentationLoss(int nc = 80, int reg_max = 16, int tal_topk = 10) : base(nameof(V8SegmentationLoss))
			{
				stride = new int[] { 8, 16, 32 };
				bce = BCEWithLogitsLoss(reduction: Reduction.None);
				this.nc = nc; // number of classes
				no = nc + reg_max * 4;
				this.reg_max = reg_max;
				use_dfl = reg_max > 1;
				this.tal_topk = tal_topk;
			}

			public override (Tensor, Tensor) forward(Tensor[] preds, Tensor targets, Tensor masks)
			{
				using (NewDisposeScope())
				{
					device = preds[0].device;
					dtype = preds[0].dtype;

					Tensor loss = zeros(4).to(device);  // box, cls, dfl
					Tensor[] feats = new Tensor[] { preds[0], preds[1], preds[2] };
					Tensor pred_masks = preds[3];
					Tensor proto = preds[4];

					long batch_size = proto.shape[0];
					long mask_h = proto.shape[2];
					long mask_w = proto.shape[3];
					Tensor[] pred_distri_scores = cat(feats.Select(xi => xi.view(feats[0].shape[0], no, -1)).ToArray(), 2).split(new long[] { reg_max * 4, nc }, 1);
					Tensor pred_distri = pred_distri_scores[0];
					Tensor pred_scores = pred_distri_scores[1];

					// B, grids, ..
					pred_scores = pred_scores.permute(0, 2, 1).contiguous();
					pred_distri = pred_distri.permute(0, 2, 1).contiguous();
					pred_masks = pred_masks.permute(0, 2, 1).contiguous();

					Tensor imgsz = tensor(feats[0].shape[2..], device: device, dtype: dtype) * stride[0]; // image size (h,w)
					(Tensor anchor_points, Tensor stride_tensor) = Utils.Tal.make_anchors(feats, stride, 0.5f);
					var indices = tensor(new long[] { 1, 0, 1, 0 }, device: device);

					// Select elements from imgsz
					var scale_tensor = index_select(imgsz, 0, indices).to(device);
					var tgs = postprocess(targets, batch_size, scale_tensor);
					Tensor[] gt_labels_bboxes = tgs.split(new long[] { 1, 4 }, 2);  // cls, xyxy
					Tensor gt_labels = gt_labels_bboxes[0];
					Tensor gt_bboxes = gt_labels_bboxes[1];
					Tensor mask_gt = gt_bboxes.sum(2, keepdim: true).gt_(0.0);

					// Pboxes
					Tensor pred_bboxes = bbox_decode(anchor_points, pred_distri);  // xyxy, (b, h*w, 4)

					Utils.Tal.TaskAlignedAssigner assigner = new Utils.Tal.TaskAlignedAssigner(topk: tal_topk, num_classes: nc, alpha: 0.5f, beta: 6.0f);
					var (_, target_bboxes, target_scores, fg_mask, target_gt_idx) = assigner.forward(pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype), anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt);
					var target_scores_sum = max(target_scores.sum());
					loss[2] = bce.forward(pred_scores, target_scores).sum() / target_scores_sum; //BCE
					if (fg_mask.sum().ToSingle() > 0)
					{
						(loss[0], loss[3]) = new BboxLoss().forward(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor, target_scores, target_scores_sum, fg_mask);
						loss[1] = calculate_segmentation_loss(fg_mask, masks, target_gt_idx, target_bboxes, /*batch_idx,*/ proto, pred_masks, imgsz, over_laps);
					}

					loss[0] *= hyp_box;    // box gain
					loss[1] *= hyp_box;    // seg gain
					loss[2] *= hyp_cls;    // cls gain
					loss[3] *= hyp_dfl;    // dfl gain

					return ((loss.sum() * batch_size).MoveToOuterDisposeScope(), loss.detach().MoveToOuterDisposeScope());    // loss(box, cls, dfl)
				}
			}

			private Tensor postprocess(Tensor targets, long batch_size, Tensor scale_tensor)
			{
				using (NewDisposeScope())
				{
					// Preprocesses the target counts and matches with the input batch size to output a tensor.
					long nl = targets.shape[0];
					long ne = targets.shape[1];

					if (nl == 0)
					{
						return zeros(new long[] { batch_size, 0, ne - 1 }, device: device);
					}
					else
					{
						Tensor i = targets[TensorIndex.Colon, 0];  // image index
						var (_, _, counts) = i.unique(return_counts: true);
						Tensor _out = zeros(new long[] { batch_size, counts.max().ToInt64(), ne - 1 }, device: device);

						for (int j = 0; j < batch_size; j++)
						{
							Tensor matches = i == j;
							long n = matches.sum().ToInt64();
							if (n > 0)
							{
								// Get the indices where matches is True
								var indices = nonzero(matches).squeeze().to(torch.ScalarType.Int64);

								// Select the rows from targets
								var selectedRows = targets.index_select(0, indices.contiguous());

								// Slice the rows to exclude the first column
								var selectedRowsSliced = selectedRows.narrow(1, 1, ne - 1);

								// Assign to the output tensor
								_out[j, TensorIndex.Slice(0, n)] = selectedRowsSliced;
							}

						}
						// Convert xywh to xyxy format and scale
						_out[TensorIndex.Ellipsis, TensorIndex.Slice(1, 5)] = Utils.Ops.xywh2xyxy(_out[TensorIndex.Ellipsis, TensorIndex.Slice(1, 5)].mul(scale_tensor));
						return _out.MoveToOuterDisposeScope();
					}
				}
			}

			private Tensor bbox_decode(Tensor anchor_points, Tensor pred_dist)
			{
				using (NewDisposeScope())
				{
					// Decode predicted object bounding box coordinates from anchor points and distribution.
					Tensor proj = arange(reg_max, dtype: pred_dist.dtype, device: pred_dist.device);
					if (use_dfl)
					{
						pred_dist = pred_dist.view(pred_dist.shape[0], pred_dist.shape[1], 4, pred_dist.shape[2] / 4).softmax(3).matmul(proj);
					}

					return dist2bbox(pred_dist, anchor_points, xywh: false).MoveToOuterDisposeScope();
				}
			}

			private Tensor dist2bbox(Tensor distance, Tensor anchor_points, bool xywh = true, int dim = -1)
			{
				using (NewDisposeScope())
				{
					Tensor[] ltrb = distance.chunk(2, dim);
					Tensor lt = ltrb[0];
					Tensor rb = ltrb[1];

					Tensor x1y1 = anchor_points - lt;
					Tensor x2y2 = anchor_points + rb;

					if (xywh)
					{
						Tensor c_xy = (x1y1 + x2y2) / 2;
						Tensor wh = x2y2 - x1y1;
						return cat(new Tensor[] { c_xy, wh }, dim);  // xywh bbox
					}
					return cat(new Tensor[] { x1y1, x2y2 }, dim).MoveToOuterDisposeScope(); // xyxy bbox
				}
			}

			/// <summary>
			/// Calculate the loss for instance segmentation.
			/// <para>Notes:<br/>
			///    The batch loss can be computed for improved speed at higher memory usage.<br/>
			///    For example, pred_mask can be computed as follows:<br/>
			///    pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
			/// </para>
			/// </summary>
			/// <param name="fg_mask">A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.</param>
			/// <param name="masks">Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).</param>
			/// <param name="target_gt_idx">Indexes of ground truth objects for each anchor of shape (BS, N_anchors).</param>
			/// <param name="target_bboxes">Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).</param>
			/// <param name="proto">Prototype masks of shape (BS, 32, H, W).</param>
			/// <param name="pred_masks">Predicted masks for each anchor of shape (BS, N_anchors, 32).</param>
			/// <param name="imgsz">Size of the input image as a tensor of shape (2), i.e., (H, W).</param>
			/// <param name="overlap">Whether the masks in `masks` tensor overlap.</param>
			/// <returns>The calculated loss for instance segmentation.</returns>
			private Tensor calculate_segmentation_loss(Tensor fg_mask, Tensor masks, Tensor target_gt_idx, Tensor target_bboxes,/* torch.Tensor batch_idx,*/ Tensor proto, Tensor pred_masks, Tensor imgsz, bool overlap)
			{
				using (NewDisposeScope())
				{

					long mask_h = proto.shape[2];
					long mask_w = proto.shape[3];
					Tensor loss = 0;

					var indices = tensor(new long[] { 1, 0, 1, 0 }, device: device);
					// Select elements from imgsz
					var scale_tensor = index_select(imgsz, 0, indices).to(device);

					// Normalize to 0-1
					Tensor target_bboxes_normalized = target_bboxes / scale_tensor;

					// Areas of target bboxes
					//Tensor marea = xyxy2xywh(target_bboxes_normalized)[TensorIndex.Ellipsis, 2..].prod(2);
					Tensor marea = torchvision.ops.box_convert(target_bboxes_normalized, torchvision.ops.BoxFormats.xyxy, torchvision.ops.BoxFormats.cxcywh)[TensorIndex.Ellipsis, 2..].prod(2);

					// Normalize to mask size
					Tensor mxyxy = target_bboxes_normalized * tensor(new long[] { mask_w, mask_h, mask_w, mask_h }, device: proto.device);

					for (int i = 0; i < fg_mask.shape[0]; i++)
					{
						if (fg_mask[i].any().ToBoolean())
						{
							Tensor mask_idx = target_gt_idx[i][fg_mask[i]];
							Tensor gt_mask = zeros(0);
							if (over_laps)
							{
								gt_mask = masks[i] == (mask_idx + 1).view(-1, 1, 1);
								gt_mask = gt_mask.@float();
							}
							else
							{
								//gt_mask = masks[batch_idx.view(-1) == i][mask_idx];
							}

							loss += single_mask_loss(gt_mask, pred_masks[i][fg_mask[i]], proto[i], mxyxy[i][fg_mask[i]], marea[i][fg_mask[i]]);
						}
					}
					return (loss / fg_mask.sum()).MoveToOuterDisposeScope();
				}
			}

			/// <summary>
			/// Compute the instance segmentation loss for a single image.
			/// <para>Notes:<br/>
			///    The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the<br/>
			///    predicted masks from the prototype masks and predicted mask coefficients.<br/>
			/// </para>
			/// </summary>
			/// <param name="gt_mask">Ground truth mask of shape (n, H, W), where n is the number of objects.</param>
			/// <param name="pred">Predicted mask coefficients of shape (n, 32).</param>
			/// <param name="proto">Prototype masks of shape (32, H, W).</param>
			/// <param name="xyxy"> Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).</param>
			/// <param name="area"> Area of each ground truth bounding box of shape (n,).</param>
			/// <returns>The calculated mask loss for a single image.</returns>
			private Tensor single_mask_loss(Tensor gt_mask, Tensor pred, Tensor proto, Tensor xyxy, Tensor area)
			{
				using (NewDisposeScope())
				{
					Tensor pred_mask = einsum("in,nhw->ihw", pred, proto); //(n, 32) @ (32, 80, 80) -> (n, 80, 80)
					Tensor loss = functional.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction: Reduction.None);
					return (Utils.Ops.crop_mask(loss, xyxy).mean(dimensions: new long[] { 1, 2 }) / area).sum().MoveToOuterDisposeScope();
				}
			}
		}

		internal class V8OBBLoss : V8DetectionLoss
		{
			private readonly Utils.Tal.RotatedTaskAlignedAssigner assigner;
			private readonly RotatedBboxLoss bbox_loss;

			public V8OBBLoss(int nc = 80, int reg_max = 16, int tal_topk = 10) : base(nc: nc, reg_max: reg_max, tal_topk: tal_topk)
			{
				this.assigner = new Utils.Tal.RotatedTaskAlignedAssigner(topk: 10, num_classes: this.nc, alpha: 0.5f, beta: 6.0f);
				this.bbox_loss = new RotatedBboxLoss(this.reg_max);
			}

			private Tensor preprocess(Tensor targets, int batch_size, Tensor scale_tensor)
			{
				// Preprocess targets for oriented bounding box detection.
				this.device = targets.device;
				this.dtype = targets.dtype;
				Tensor @out = torch.zeros(batch_size, 0, 6, device: this.device);
				if (targets.shape[0] != 0)
				{
					Tensor i = targets[.., 0];  // image index
					(_, _, Tensor counts) = i.unique(return_counts: true);

					counts = counts.to(torch.int32);
					@out = torch.zeros(new long[] { batch_size, counts.max().ToInt64(), 6 }, device: this.device);

					for (int j = 0; j < batch_size; j++)
					{
						Tensor matches = (i == j);
						int n = matches.sum().ToInt32();
						if (n > 0)
						{
							Tensor bboxes = targets[TensorIndex.Tensor(matches), 2..];
							bboxes[.., ..4].mul_(scale_tensor);
							@out[j, ..n] = torch.cat(new Tensor[] { targets[TensorIndex.Tensor(matches), 1..2], bboxes }, dim: -1);
						}
					}

				}
				return @out; //Preprocess targets for oriented bounding box detection.
			}

			public override (Tensor, Tensor) forward(Tensor[] preds, Tensor targets)
			{
				// Calculate and return the loss for oriented bounding box detection.
				Tensor loss = torch.zeros(3, device: this.device);  // box, cls, dfl
				Tensor[] feats = new Tensor[] { preds[1], preds[2], preds[3] };
				Tensor pred_angle = preds[4];

				long batch_size = pred_angle.shape[0];  // batch size, number of masks, mask height, mask width

				Tensor[] pred_mix = feats.Select(xi => xi.view(new long[] { feats[0].shape[0], this.no, -1 })).ToArray();
				Tensor[] pred_dist_scores = torch.cat(pred_mix, 2).split(new long[] { this.reg_max * 4, this.nc }, 1);
				Tensor pred_distri = pred_dist_scores[0];
				Tensor pred_scores = pred_dist_scores[1];

				// b, grids, ..
				pred_scores = pred_scores.permute(0, 2, 1).contiguous();
				pred_distri = pred_distri.permute(0, 2, 1).contiguous();
				pred_angle = pred_angle.permute(0, 2, 1).contiguous();

				dtype = pred_scores.dtype;
				device = pred_scores.device;
				Tensor imgsz = torch.tensor(feats[0].shape[2..], device: device, dtype: dtype) * this.stride[0];  // image size (h,w)
				(Tensor anchor_points, Tensor stride_tensor) = Utils.Tal.make_anchors(feats, this.stride, 0.5f);

				// targets
				//	try
				//	{
				//		Tensor batch_idx = batch["batch_idx"].view(-1, 1);
				//		targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1);

				//		Tensor rw = targets[.., 4] * imgsz[0];
				//		Tensor rh = targets[.., 5] * imgsz[1];

				//		targets = targets[(rw >= 2) & (rh >= 2)]; // filter rboxes of tiny size to stabilize training
				//		targets = this.preprocess(targets.to(this.device), batch_size, scale_tensor: imgsz[[1, 0, 1, 0]]);


				//		gt_labels, gt_bboxes = targets.split((1, 5), 2)  // cls, xywhr

				//mask_gt = gt_bboxes.sum(2, keepdim: true).gt_(0.0);
				//	}
				//	catch (Exception e)
				//	{

				//	}


				return (torch.zeros(0), torch.zeros(0));

			}
		}

	}
}



