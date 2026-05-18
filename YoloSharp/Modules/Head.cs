using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Modules
{
    internal class Head
    {
        internal class Detect : torch.nn.Module<torch.Tensor[], (torch.Tensor y, Dictionary<string, object> preds)>
        {
            private long[] shape = null;
            protected torch.Tensor anchors = torch.empty(0); // init
            protected torch.Tensor strides = torch.empty(0); // init

            private readonly int nc;
            protected readonly int nl;
            private readonly int reg_max;
            private readonly int no;
            private readonly int[] stride;
            private ModuleList<Sequential> cv2;
            private ModuleList<Sequential> cv3;
            private readonly torch.nn.Module<torch.Tensor, torch.Tensor> dfl;

            private readonly ModuleList<Sequential> one2one_cv2;
            private readonly ModuleList<Sequential> one2one_cv3;

            bool legacy = false;  // backward compatibility for v3/v5/v8/v9 models
            bool dynamic = false;  // force grid reconstruction
            bool export = false;    // export mode
            int max_det = 300;    // max_det
            bool agnostic_nms = false;

            bool xyxy = false;    // xyxy or xywh output

            public bool end2end { get; set; }

            internal Detect(int nc = 80, int reg_max = 16, bool end2end = false, int[]? ch = null) : base(nameof(Detect))
            {
                this.nc = nc; // number of classes
                this.nl = ch.Length;  // number of detection layers
                this.reg_max = reg_max; // DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
                this.no = nc + this.reg_max * 4; // number of outputs per anchor
                this.stride = torch.zeros(this.nl).data<int>().ToArray(); // strides computed during build
                int c2 = Math.Max(16, Math.Max(ch[0] / 4, this.reg_max * 4));
                int c3 = Math.Max(ch[0], Math.Min(this.nc, 100)); // channels

                this.cv2 = torch.nn.ModuleList(ch.Select(x => torch.nn.Sequential(new YoloSharp.Modules.Modules.Conv(x, c2, 3), new YoloSharp.Modules.Modules.Conv(c2, c2, 3), torch.nn.Conv2d(c2, 4 * this.reg_max, 1))).ToArray());
                this.cv3 = this.legacy ?
                    torch.nn.ModuleList(ch.Select(x => torch.nn.Sequential(new YoloSharp.Modules.Modules.Conv(x, c3, 3), new YoloSharp.Modules.Modules.Conv(c3, c3, 3), torch.nn.Conv2d(c3, this.nc, 1))).ToArray()) :
                   torch.nn.ModuleList(ch.Select(x => torch.nn.Sequential(torch.nn.Sequential(new YoloSharp.Modules.Modules.DWConv(x, x, 3), new YoloSharp.Modules.Modules.Conv(x, c3, 1)), torch.nn.Sequential(new YoloSharp.Modules.Modules.DWConv(c3, c3, 3), new YoloSharp.Modules.Modules.Conv(c3, c3, 1)), torch.nn.Conv2d(c3, this.nc, 1))).ToArray());

                this.dfl = this.reg_max > 1 ? new YoloSharp.Modules.Modules.DFL(this.reg_max) : torch.nn.Identity();
                if (end2end)
                {
                    this.one2one_cv2 = torch.nn.ModuleList(this.cv2.ToArray());
                    this.one2one_cv3 = torch.nn.ModuleList(this.cv3.ToArray());
                }
                this.end2end = end2end;

                // RegisterComponents();
            }

            private Dictionary<string, ModuleList<Sequential>> one2many()
            {
                Dictionary<string, ModuleList<Sequential>> result = new Dictionary<string, ModuleList<Sequential>>();
                result.Add("box_head", this.cv2);
                result.Add("cls_head", this.cv3);
                return result;
            }

            private Dictionary<string, ModuleList<Sequential>> one2one()
            {
                Dictionary<string, ModuleList<Sequential>> result = new Dictionary<string, ModuleList<Sequential>>();
                result.Add("box_head", this.one2one_cv2);
                result.Add("cls_head", this.one2one_cv3);
                return result;
            }

            private Dictionary<string, object> forward_head(torch.Tensor[] x, Dictionary<string, ModuleList<Sequential>> heads)
            {
                Dictionary<string, object> result = new Dictionary<string, object>();
                bool get_box_head = heads.TryGetValue("box_head", out ModuleList<Sequential>? box_head);
                bool get_cls_head = heads.TryGetValue("cls_head", out ModuleList<Sequential>? cls_head);
                if (!get_box_head || !get_cls_head)
                {
                    return result;
                }
                long bs = x[0].shape[0];  // batch size
                torch.Tensor boxes = torch.cat(Enumerable.Range(0, this.nl).Select(i => box_head[i].forward(x[i]).view(bs, 4 * this.reg_max, -1)).ToArray(), dim: -1);
                torch.Tensor scores = torch.cat(Enumerable.Range(0, this.nl).Select(i => cls_head[i].forward(x[i]).view(bs, this.nc, -1)).ToArray(), dim: -1);
                result.Add("feats", x);
                result.Add("boxes", boxes);
                result.Add("scores", scores);
                return result;
            }

            public override (torch.Tensor y, Dictionary<string, object> preds) forward(torch.Tensor[] x)
            {
                Dictionary<string, object> preds = this.forward_head(x, this.one2many());
                if (this.end2end)
                {
                    torch.Tensor[] x_detach = x.Select(xi => xi.detach()).ToArray();
                    Dictionary<string, object> one2one = this.forward_head(x_detach, this.one2one());
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
                return this.export ? (y, null) : (y, preds);
            }

            private torch.Tensor postprocess(torch.Tensor preds)
            {
                //boxes, scores

                torch.Tensor[] boxes_scores = preds.split(new long[] { 4, this.nc }, dim: -1);
                torch.Tensor boxes = boxes_scores[0];
                torch.Tensor scores = boxes_scores[1];

                (scores, torch.Tensor conf, torch.Tensor idx) = this.get_topk_index(scores, this.max_det);
                boxes = boxes.gather(dim: 1, index: idx.repeat(1, 1, 4));
                return torch.cat(new torch.Tensor[] { boxes, scores, conf }, dim: -1);
            }

            private void bias_init()
            {
                var one2many = this.one2many();
                for (int i = 0; i < one2many["cls_head"].Count; i++)
                {
                    one2many["box_head"][i].get_parameter("bias").fill_(2.0f);
                    one2many["cls_head"][i].get_parameter("bias")[..this.nc] = Math.Log(5 / this.nc / Math.Pow((640 / this.stride[i]), 2));
                }
                if (this.end2end)
                {
                    var one2one = this.one2one();
                    for (int i = 0; i < one2many["cls_head"].Count; i++)
                    {
                        one2one["box_head"][i].get_parameter("bias").fill_(2.0f);
                        one2one["cls_head"][i].get_parameter("bias")[..this.nc] = Math.Log(5 / this.nc / Math.Pow((640 / this.stride[i]), 2));
                    }
                }

            }

            private (torch.Tensor, torch.Tensor, torch.Tensor) get_topk_index(torch.Tensor scores, int max_det)
            {
                // i.e. shape(16,8400,84)
                long batch_size = scores.shape[0];
                long anchors = scores.shape[1];
                long nc = scores.shape[2];

                int k = this.export ? max_det : (int)Math.Min(max_det, anchors);
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
                torch.Tensor idx = ori_index[torch.arange(batch_size)[torch.TensorIndex.Ellipsis, torch.TensorIndex.None], index / nc]; // original index
                return (scores[torch.TensorIndex.Ellipsis, torch.TensorIndex.None], (index % nc)[torch.TensorIndex.Ellipsis, torch.TensorIndex.None].@float(), idx);

            }

            // Decode bounding boxes.
            protected virtual torch.Tensor decode_bboxes(torch.Tensor bboxes, torch.Tensor anchors, bool xywh = true)
            {
                return YoloSharp.Utils.Tal.dist2bbox(bboxes, anchors, xywh: (xywh && !this.end2end && !this.xyxy), dim: 1);
            }

            private torch.Tensor _inference(Dictionary<string, object> x)
            {
                torch.Tensor dbox = this._get_decode_boxes(x);
                return torch.cat(new torch.Tensor[] { dbox, ((torch.Tensor)x["scores"]).sigmoid() }, 1);
            }

            private torch.Tensor _get_decode_boxes(Dictionary<string, object> x)
            {
                long[] shape = ((torch.Tensor)x["feats"])[0].shape; // BCHW
                if (this.dynamic || this.shape != shape)
                {
                    (this.anchors, this.strides) = YoloSharp.Utils.Tal.make_anchors((torch.Tensor[])x["feats"], this.stride, 0.5f);
                    this.shape = shape;
                }
                torch.Tensor dbox = this.decode_bboxes(this.dfl.forward((torch.Tensor)x["boxes"]), this.anchors.unsqueeze(0)) * this.strides;
                return dbox;
            }

            public void fuse()
            {
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
            private readonly ModuleList<Sequential> cv4 = new ModuleList<Sequential>();

            internal Segment(int nc = 80, int nm = 32, int npr = 256, int reg_max = 16, bool end2end = false, int[] ch = null, Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, reg_max: reg_max, end2end: end2end, ch: ch)
            {
                this.nm = nm;  // number of masks
                this.npr = npr;  // number of protos
                proto = new Block.Proto(ch[0], this.npr, this.nm, device: device, dtype: dtype);  // protos
                c4 = Math.Max(ch[0] / 4, this.nm);

                cv4 = new ModuleList<Sequential>(ch.Select(x =>
                    nn.Sequential(
                        new Convs.Conv(x, c4, 3, device: device, dtype: dtype),
                        new Convs.Conv(c4, c4, 3, device: device, dtype: dtype),
                        nn.Conv2d(c4, this.nm, 1, device: device, dtype: dtype)
                    )).ToArray());

                RegisterComponents();
            }
        }


    }
}
