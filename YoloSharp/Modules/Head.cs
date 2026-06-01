using TorchSharp;
using TorchSharp.Modules;

namespace YoloSharp.Modules
{
    internal class Head
    {
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

            public void bias_init()
            {
                this.cv2[cv2.Count - 1].named_parameters().Where(a => a.name.Contains("bias")).Select(a => a.parameter.fill_(2.0f));

                for (int i = 0; i < this.stride.Length; i++)
                {
                    float cv3_value = (float)Math.Log(5 / this.nc / Math.Pow(640 / this.stride[i], 2));
                    this.cv3[cv3.Count - 1].named_parameters().Where(a => a.name.Contains("bias")).Select(a => a.parameter.fill_(cv3_value));
                }

                if (end2end)
                {
                    this.one2one_cv2[one2one_cv2.Count - 1].named_parameters().Where(a => a.name.Contains("bias")).Select(a => a.parameter.fill_(2.0f));

                    for (int i = 0; i < this.stride.Length; i++)
                    {
                        float cv3_value = (float)Math.Log(5 / this.nc / Math.Pow(640 / this.stride[i], 2));
                        this.one2one_cv3[one2one_cv3.Count - 1].named_parameters().Where(a => a.name.Contains("bias")).Select(a => a.parameter.fill_(cv3_value));
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
    }
}
