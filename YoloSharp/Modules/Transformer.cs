using TorchSharp;
using TorchSharp.Modules;

namespace YoloSharp.Modules
{
    internal class Transformer
    {
        internal class TransformerBlock : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly Convs.Conv conv;
            private readonly Linear linear;
            private readonly Sequential tr;
            private readonly int c2;

            internal TransformerBlock(int c1, int c2, int num_heads, int num_layers, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(TransformerBlock))
            {
                if (c1 != c2)
                {
                    this.conv = new Convs.Conv(c1, c2, device: device, dtype: dtype);
                }

                this.linear = torch.nn.Linear(c2, c2, device: device, dtype: dtype); // learnable position embedding
                this.tr = torch.nn.Sequential(Enumerable.Range(0, num_layers).Select(_ => new TransformerLayer(c2, num_heads, device, dtype)));
                this.c2 = c2;

                RegisterComponents();

            }

            /// <summary>
            /// Forward propagate the input through the transformer block.
            /// </summary>
            /// <param name="x">Input tensor with shape [b, c1, h, w].</param>
            /// <returns>Output tensor with shape [b, c2, h, w].</returns>
            public override torch.Tensor forward(torch.Tensor x)
            {
                if (this.conv is not null)
                {
                    x = this.conv.forward(x);
                }
                long b = x.shape[0];
                long h = x.shape[2];
                long w = x.shape[3];

                torch.Tensor p = x.flatten(2).permute(2, 0, 1);
                return this.tr.forward(p + this.linear.forward(p)).permute(1, 2, 0).reshape(b, this.c2, h, w);
            }
        }

        /// <summary>
        /// Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance).
        /// </summary>
        internal class TransformerLayer : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly Linear q;
            private readonly Linear k;
            private readonly Linear v;
            private readonly MultiheadAttention ma;
            private readonly Linear fc1;
            private readonly Linear fc2;

            /// <summary>
            /// Initialize a this-attention mechanism using linear transformations and multi-head attention.
            /// </summary>
            /// <param name="c">Input and output channel dimension.</param>
            /// <param name="num_heads">Number of attention heads.</param>
            /// <param name="device">Device type.</param>
            /// <param name="dtype">Scaler type.</param>
            internal TransformerLayer(int c, int num_heads, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(TransformerLayer))
            {
                this.q = torch.nn.Linear(c, c, hasBias: false, device: device, dtype: dtype);
                this.k = torch.nn.Linear(c, c, hasBias: false, device: device, dtype: dtype);
                this.v = torch.nn.Linear(c, c, hasBias: false, device: device, dtype: dtype);
                this.ma = torch.nn.MultiheadAttention(embedded_dim: c, num_heads: num_heads);
                this.fc1 = torch.nn.Linear(c, c, hasBias: false, device: device, dtype: dtype);
                this.fc2 = torch.nn.Linear(c, c, hasBias: false, device: device, dtype: dtype);

                RegisterComponents();
            }

            /// <summary>
            /// Apply a transformer block to the input x and return the output.
            /// </summary>
            /// <param name="x">Input tensor.</param>
            /// <returns>Output tensor after transformer layer.</returns>
            public override torch.Tensor forward(torch.Tensor x)
            {
                x = this.ma.forward(this.q.forward(x), this.k.forward(x), this.v.forward(x), null, true, null).Item1 + x;
                return this.fc2.forward(this.fc1.forward(x)) + x;
            }
        }
    }
}
