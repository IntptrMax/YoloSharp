using System;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;

namespace YoloSharp.Modules
{
    /// <summary>
    /// Transformer modules (mirrors ultralytics/nn/modules/transformer.py).
    /// </summary>
    internal class Transformer
    {
        /// <summary>
        /// A single layer of the transformer encoder.
        ///
        /// This class implements a standard transformer encoder layer with multi-head attention and feedforward
        /// network, supporting both pre-normalization and post-normalization configurations.
        ///
        /// Attributes:
        ///     ma (nn.MultiheadAttention): Multi-head attention module.
        ///     fc1 (nn.Linear): First linear layer in the feedforward network.
        ///     fc2 (nn.Linear): Second linear layer in the feedforward network.
        ///     norm1 (nn.LayerNorm): Layer normalization after attention.
        ///     norm2 (nn.LayerNorm): Layer normalization after feedforward network.
        ///     dropout (nn.Dropout): Dropout layer for the feedforward network.
        ///     dropout1 (nn.Dropout): Dropout layer after attention.
        ///     dropout2 (nn.Dropout): Dropout layer after feedforward network.
        ///     act (nn.Module): Activation function.
        ///     normalize_before (bool): Whether to apply normalization before attention and feedforward.
        /// </summary>
        internal class TransformerEncoderLayer : torch.nn.Module
        {
            protected readonly MultiheadAttention ma;
            protected readonly Linear fc1, fc2;
            protected readonly LayerNorm norm1, norm2;
            private readonly Dropout dropout, dropout1, dropout2;
            protected readonly torch.nn.Module<torch.Tensor, torch.Tensor> act;
            private readonly bool normalize_before;

            /// <summary>
            /// Initialize the TransformerEncoderLayer with specified parameters.
            ///
            /// Args:
            ///     c1 (int): Input dimension.
            ///     cm (int): Hidden dimension in the feedforward network.
            ///     num_heads (int): Number of attention heads.
            ///     dropout (float): Dropout probability.
            ///     act (nn.Module, optional): Activation function. Defaults to nn.GELU() if None.
            ///     normalize_before (bool): Whether to apply normalization before attention and feedforward.
            /// </summary>
            internal TransformerEncoderLayer(int c1, int cm = 2048, int num_heads = 8, double dropout = 0.0,
                torch.nn.Module<torch.Tensor, torch.Tensor>? act = null, bool normalize_before = false,
                torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(TransformerEncoderLayer))
            {
                this.ma = torch.nn.MultiheadAttention(embedded_dim: c1, num_heads: num_heads, dropout: dropout);
                // Implementation of Feedforward model
                this.fc1 = torch.nn.Linear(c1, cm, device: device, dtype: dtype);
                this.fc2 = torch.nn.Linear(cm, c1, device: device, dtype: dtype);
                this.norm1 = torch.nn.LayerNorm(c1, device: device, dtype: dtype);
                this.norm2 = torch.nn.LayerNorm(c1, device: device, dtype: dtype);
                this.dropout = torch.nn.Dropout(dropout);
                this.dropout1 = torch.nn.Dropout(dropout);
                this.dropout2 = torch.nn.Dropout(dropout);
                this.act = act ?? torch.nn.GELU();
                this.normalize_before = normalize_before;
                RegisterComponents();
            }

            /// <summary>Add position embeddings to the tensor if provided.</summary>
            protected static torch.Tensor with_pos_embed(torch.Tensor tensor, torch.Tensor? pos = null)
            {
                return pos is null ? tensor : tensor + pos;
            }

            /// <summary>
            /// Perform forward pass with post-normalization.
            ///
            /// Args:
            ///     src (torch.Tensor): Input tensor.
            ///     pos (torch.Tensor, optional): Positional encoding.
            ///
            /// Returns:
            ///     (torch.Tensor): Output tensor after attention and feedforward.
            /// </summary>
            protected torch.Tensor forward_post(torch.Tensor src, torch.Tensor? pos = null)
            {
                var qk = with_pos_embed(src, pos);
                var src2 = ma.forward(qk, qk, src, null, true, null).Item1;
                src = src + dropout1.forward(src2);
                src = norm1.forward(src);
                src2 = fc2.forward(dropout.forward(act.forward(fc1.forward(src))));
                src = src + dropout2.forward(src2);
                return norm2.forward(src);
            }

            /// <summary>
            /// Perform forward pass with pre-normalization.
            ///
            /// Args:
            ///     src (torch.Tensor): Input tensor.
            ///     pos (torch.Tensor, optional): Positional encoding.
            ///
            /// Returns:
            ///     (torch.Tensor): Output tensor after attention and feedforward.
            /// </summary>
            protected torch.Tensor forward_pre(torch.Tensor src, torch.Tensor? pos = null)
            {
                var src2 = norm1.forward(src);
                var qk = with_pos_embed(src2, pos);
                src2 = ma.forward(qk, qk, src2, null, true, null).Item1;
                src = src + dropout1.forward(src2);
                src2 = norm2.forward(src);
                src2 = fc2.forward(dropout.forward(act.forward(fc1.forward(src2))));
                return src + dropout2.forward(src2);
            }

            /// <summary>
            /// Forward propagate the input through the encoder module.
            ///
            /// Args:
            ///     src (torch.Tensor): Input tensor.
            ///     pos (torch.Tensor, optional): Positional encoding.
            ///
            /// Returns:
            ///     (torch.Tensor): Output tensor after transformer encoder layer.
            /// </summary>
            public virtual torch.Tensor forward(torch.Tensor src, torch.Tensor? pos = null)
            {
                return normalize_before ? forward_pre(src, pos) : forward_post(src, pos);
            }
        }

        /// <summary>
        /// AIFI transformer layer for 2D data with positional embeddings.
        ///
        /// This class extends TransformerEncoderLayer to work with 2D feature maps by adding 2D sine-cosine
        /// positional embeddings and handling the spatial dimensions appropriately.
        /// </summary>
        internal class AIFI : TransformerEncoderLayer
        {
            /// <summary>
            /// Initialize the AIFI instance with specified parameters.
            ///
            /// Args:
            ///     c1 (int): Input dimension.
            ///     cm (int): Hidden dimension in the feedforward network.
            ///     num_heads (int): Number of attention heads.
            ///     dropout (float): Dropout probability.
            ///     act (nn.Module, optional): Activation function. Defaults to nn.GELU() if None.
            ///     normalize_before (bool): Whether to apply normalization before attention and feedforward.
            /// </summary>
            internal AIFI(int c1, int cm = 2048, int num_heads = 8, double dropout = 0.0,
                torch.nn.Module<torch.Tensor, torch.Tensor>? act = null, bool normalize_before = false,
                torch.Device? device = null, torch.ScalarType? dtype = null)
                : base(c1, cm, num_heads, dropout, act, normalize_before, device, dtype)
            {
            }

            /// <summary>
            /// Forward pass for the AIFI transformer layer.
            ///
            /// Args:
            ///     x (torch.Tensor): Input tensor with shape [B, C, H, W].
            ///
            /// Returns:
            ///     (torch.Tensor): Output tensor with shape [B, C, H, W].
            /// </summary>
            public torch.Tensor forward(torch.Tensor x)
            {
                long c = x.shape[1], h = x.shape[2], w = x.shape[3];
                var pos_embed = build_2d_sincos_position_embedding((int)w, (int)h, (int)c, 10000.0, x);
                // Flatten [B, C, H, W] to [B, HxW, C]
                // NOTE: no `new` keyword — this does NOT hide a base member (TransformerEncoderLayer.forward
                // takes different parameters = overloading, not hiding). / 无new：与基类参数不同是重载不是遮蔽
                x = base.forward(x.flatten(2).permute(0, 2, 1), pos_embed.to(x.dtype));
                return x.permute(0, 2, 1).reshape(-1, c, h, w).contiguous();
            }

            /// <summary>
            /// Build 2D sine-cosine position embedding.
            ///
            /// Args:
            ///     w (int): Width of the feature map.
            ///     h (int): Height of the feature map.
            ///     embed_dim (int): Embedding dimension.
            ///     temperature (float): Temperature for the sine/cosine functions.
            ///     like (torch.Tensor): Tensor whose device the embedding grids are built on.
            ///
            /// Returns:
            ///     (torch.Tensor): Position embedding with shape [1, h*w, embed_dim].
            /// </summary>
            internal static torch.Tensor build_2d_sincos_position_embedding(int w, int h, int embed_dim = 256,
                double temperature = 10000.0, torch.Tensor? like = null)
            {
                if (embed_dim % 4 != 0)
                    throw new ArgumentException("Embed dimension must be divisible by 4 for 2D sin-cos position embedding");
                var dev = like?.device ?? torch.CPU;
                // fp32 seed keeps sin/cos exact
                var grid_w = torch.arange(w, device: dev, dtype: torch.ScalarType.Float32);
                var grid_h = torch.arange(h, device: dev, dtype: torch.ScalarType.Float32);
                // Tokens are flattened row-major from [B, C, H, W], so the grid must be (h, w)
                var gh = grid_h.reshape(h, 1).expand(h, w).reshape(-1);   // meshgrid(h,w) row-major flatten
                var gw = grid_w.reshape(1, w).expand(h, w).reshape(-1);
                int pos_dim = embed_dim / 4;
                var omega = torch.arange(pos_dim, device: dev, dtype: torch.ScalarType.Float32) / pos_dim;
                omega = 1.0 / torch.pow(temperature, omega);

                // Pin matmul to fp32: fp16 sin/cos on integer-derived positions accumulates visible error.
                var out_w = gw.reshape(-1, 1).matmul(omega.reshape(1, -1));
                var out_h = gh.reshape(-1, 1).matmul(omega.reshape(1, -1));

                // Row encoding first keeps square inputs identical to the original layout
                return torch.cat(new[] { out_h.sin(), out_h.cos(), out_w.sin(), out_w.cos() }, 1).unsqueeze(0);
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
            /// Initialize a self-attention mechanism using linear transformations and multi-head attention.
            ///
            /// Args:
            ///     c (int): Input and output channel dimension.
            ///     num_heads (int): Number of attention heads.
            /// </summary>
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
            ///
            /// Args:
            ///     x (torch.Tensor): Input tensor with shape (seq_len, batch, c).
            ///
            /// Returns:
            ///     (torch.Tensor): Output tensor after transformer layer.
            /// </summary>
            public override torch.Tensor forward(torch.Tensor x)
            {
                x = this.ma.forward(this.q.forward(x), this.k.forward(x), this.v.forward(x), null, true, null).Item1 + x;
                return this.fc2.forward(this.fc1.forward(x)) + x;
            }
        }

        /// <summary>
        /// Vision Transformer block based on https://arxiv.org/abs/2010.11929.
        ///
        /// This class implements a complete transformer block with optional convolution layer for channel
        /// adjustment, learnable position embedding, and multiple transformer layers.
        ///
        /// Attributes:
        ///     conv (Conv, optional): Convolution layer if input and output channels differ.
        ///     linear (nn.Linear): Learnable position embedding.
        ///     tr (nn.Sequential): Sequential container of transformer layers.
        ///     c2 (int): Output channel dimension.
        /// </summary>
        internal class TransformerBlock : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly Convs.Conv conv;
            private readonly Linear linear;
            private readonly Sequential tr;
            private readonly int c2;

            /// <summary>
            /// Initialize a Transformer module with position embedding and specified number of heads and layers.
            ///
            /// Args:
            ///     c1 (int): Input channel dimension.
            ///     c2 (int): Output channel dimension.
            ///     num_heads (int): Number of attention heads.
            ///     num_layers (int): Number of transformer layers.
            /// </summary>
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
            ///
            /// Args:
            ///     x (torch.Tensor): Input tensor with shape [b, c1, h, w].
            ///
            /// Returns:
            ///     (torch.Tensor): Output tensor with shape [b, c2, h, w].
            /// </summary>
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
        /// A single block of a multi-layer perceptron.
        /// </summary>
        internal class MLPBlock : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly Linear lin1, lin2;
            private readonly torch.nn.Module<torch.Tensor, torch.Tensor> act;

            /// <summary>
            /// Initialize the MLPBlock with specified embedding dimension, MLP dimension, and activation function.
            ///
            /// Args:
            ///     embedding_dim (int): Input and output dimension.
            ///     mlp_dim (int): Hidden dimension.
            ///     act (nn.Module, optional): Activation function. Defaults to nn.GELU().
            /// </summary>
            internal MLPBlock(int embedding_dim, int mlp_dim, torch.nn.Module<torch.Tensor, torch.Tensor>? act = null,
                torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(MLPBlock))
            {
                this.lin1 = torch.nn.Linear(embedding_dim, mlp_dim, device: device, dtype: dtype);
                this.lin2 = torch.nn.Linear(mlp_dim, embedding_dim, device: device, dtype: dtype);
                this.act = act ?? torch.nn.GELU();
                RegisterComponents();
            }

            /// <summary>
            /// Forward pass for the MLPBlock.
            ///
            /// Args:
            ///     x (torch.Tensor): Input tensor.
            ///
            /// Returns:
            ///     (torch.Tensor): Output tensor after MLP block.
            /// </summary>
            public override torch.Tensor forward(torch.Tensor x)
            {
                return lin2.forward(act.forward(lin1.forward(x)));
            }
        }

        /// <summary>
        /// A simple multi-layer perceptron (also called FFN).
        ///
        /// This class implements a configurable MLP with multiple linear layers, activation functions, and optional
        /// sigmoid output activation.
        ///
        /// Attributes:
        ///     num_layers (int): Number of layers in the MLP.
        ///     layers (ModuleList): List of linear layers.
        ///     sigmoid (bool): Whether to apply sigmoid to the output.
        ///     act (nn.Module): Activation function.
        ///     residual (bool): Whether to add the input to the output.
        ///     out_norm (nn.Module): Normalization layer applied to the output (nn.Identity if not provided).
        /// </summary>
        internal class MLP : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly int num_layers;
            private readonly ModuleList<Linear> layers;
            private readonly bool sigmoid;
            private readonly torch.nn.Module<torch.Tensor, torch.Tensor> act;
            private readonly bool residual;
            private readonly torch.nn.Module<torch.Tensor, torch.Tensor> out_norm;

            /// <summary>
            /// Initialize the MLP with specified input, hidden, output dimensions and number of layers.
            ///
            /// Args:
            ///     input_dim (int): Input dimension.
            ///     hidden_dim (int): Hidden dimension.
            ///     output_dim (int): Output dimension.
            ///     num_layers (int): Number of layers.
            ///     act (nn.Module, optional): Activation function. Defaults to nn.ReLU().
            ///     sigmoid (bool): Whether to apply sigmoid to the output.
            ///     residual (bool): Whether to use residual connections.
            ///     out_norm (nn.Module, optional): Normalization layer for the output.
            /// </summary>
            internal MLP(int input_dim, int hidden_dim, int output_dim, int num_layers,
                torch.nn.Module<torch.Tensor, torch.Tensor>? act = null, bool sigmoid = false, bool residual = false,
                torch.nn.Module<torch.Tensor, torch.Tensor>? out_norm = null,
                torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(MLP))
            {
                this.num_layers = num_layers;
                int[] h = Enumerable.Repeat(hidden_dim, num_layers - 1).ToArray();
                var dims = new int[num_layers + 1];
                dims[0] = input_dim;
                for (int i = 0; i < h.Length; i++) dims[i + 1] = h[i];
                dims[num_layers] = output_dim;
                var lins = new System.Collections.Generic.List<Linear>(num_layers);
                for (int i = 0; i < num_layers; i++)
                    lins.Add(torch.nn.Linear(dims[i], dims[i + 1], device: device, dtype: dtype));
                this.layers = new ModuleList<Linear>(lins.ToArray());
                this.sigmoid = sigmoid;
                this.act = act ?? torch.nn.ReLU();
                if (residual && input_dim != output_dim)
                    throw new ArgumentException("residual is only supported if input_dim == output_dim");
                this.residual = residual;
                this.out_norm = out_norm ?? torch.nn.Identity();   // whether to apply a normalization layer to the output
                RegisterComponents();
            }

            /// <summary>
            /// Forward pass for the entire MLP.
            ///
            /// Args:
            ///     x (torch.Tensor): Input tensor.
            ///
            /// Returns:
            ///     (torch.Tensor): Output tensor after MLP.
            /// </summary>
            public override torch.Tensor forward(torch.Tensor x)
            {
                var orig_x = x;
                for (int i = 0; i < num_layers; i++)
                    x = i < num_layers - 1 ? act.forward(layers[i].forward(x)) : layers[i].forward(x);
                if (residual)
                    x = x + orig_x;
                x = out_norm.forward(x);
                return sigmoid ? x.sigmoid() : x;
            }
        }

        /// <summary>
        /// 2D Layer Normalization module inspired by Detectron2 and ConvNeXt implementations.
        ///
        /// This class implements layer normalization for 2D feature maps, normalizing across the channel dimension
        /// while preserving spatial dimensions.
        ///
        /// Attributes:
        ///     weight (nn.Parameter): Learnable scale parameter.
        ///     bias (nn.Parameter): Learnable bias parameter.
        ///     eps (float): Small constant for numerical stability.
        ///
        /// References:
        ///     https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
        ///     https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
        /// </summary>
        internal class LayerNorm2d : torch.nn.Module<torch.Tensor, torch.Tensor>
        {
            private readonly torch.Tensor weight;
            private readonly torch.Tensor bias;
            private readonly double eps;

            /// <summary>
            /// Initialize LayerNorm2d with the given parameters.
            ///
            /// Args:
            ///     num_channels (int): Number of channels in the input.
            ///     eps (float): Small constant for numerical stability.
            /// </summary>
            internal LayerNorm2d(int num_channels, double eps = 1e-6, torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(LayerNorm2d))
            {
                this.weight = torch.ones(num_channels, device: device, dtype: dtype ?? torch.ScalarType.Float32);
                this.weight.requires_grad = true;
                this.bias = torch.zeros(num_channels, device: device, dtype: dtype ?? torch.ScalarType.Float32);
                this.bias.requires_grad = true;
                this.eps = eps;
                RegisterComponents();
            }

            /// <summary>
            /// Perform forward pass for 2D layer normalization.
            ///
            /// Args:
            ///     x (torch.Tensor): Input tensor.
            ///
            /// Returns:
            ///     (torch.Tensor): Normalized output.
            /// </summary>
            public override torch.Tensor forward(torch.Tensor x)
            {
                var c = x.shape[1];
                var y = torch.nn.functional.layer_norm(x.permute(0, 2, 3, 1),
                    new long[] { c }, weight, bias, eps);
                return y.permute(0, 3, 1, 2);
            }
        }

        /// <summary>
        /// Multiscale Deformable Attention Module based on Deformable-DETR and PaddleDetection implementations.
        ///
        /// This module implements multiscale deformable attention that can attend to features at multiple scales with
        /// learnable sampling locations and attention weights.
        ///
        /// Attributes:
        ///     im2col_step (int): Step size for im2col operations.
        ///     d_model (int): Model dimension.
        ///     n_levels (int): Number of feature levels.
        ///     n_heads (int): Number of attention heads.
        ///     n_points (int): Number of sampling points per attention head per feature level.
        ///     sampling_offsets (nn.Linear): Linear layer for generating sampling offsets.
        ///     attention_weights (nn.Linear): Linear layer for generating attention weights.
        ///     value_proj (nn.Linear): Linear layer for projecting values.
        ///     output_proj (nn.Linear): Linear layer for projecting output.
        ///
        /// References:
        ///     https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
        /// </summary>
        internal class MSDeformAttn : torch.nn.Module
        {
            internal const int im2col_step = 64;
            private readonly int d_model, n_levels, n_heads, n_points;
            private readonly Linear sampling_offsets, attention_weights, value_proj, output_proj;

            /// <summary>
            /// Initialize MSDeformAttn with the given parameters.
            ///
            /// Args:
            ///     d_model (int): Model dimension.
            ///     n_levels (int): Number of feature levels.
            ///     n_heads (int): Number of attention heads.
            ///     n_points (int): Number of sampling points per attention head per feature level.
            /// </summary>
            internal MSDeformAttn(int d_model = 256, int n_levels = 4, int n_heads = 8, int n_points = 4,
                torch.Device? device = null, torch.ScalarType? dtype = null) : base(nameof(MSDeformAttn))
            {
                if (d_model % n_heads != 0)
                    throw new ArgumentException($"d_model must be divisible by n_heads, but got {d_model} and {n_heads}");
                this.d_model = d_model;
                this.n_levels = n_levels;
                this.n_heads = n_heads;
                this.n_points = n_points;

                this.sampling_offsets = torch.nn.Linear(d_model, n_heads * n_levels * n_points * 2, device: device, dtype: dtype);
                this.attention_weights = torch.nn.Linear(d_model, n_heads * n_levels * n_points, device: device, dtype: dtype);
                this.value_proj = torch.nn.Linear(d_model, d_model, device: device, dtype: dtype);
                this.output_proj = torch.nn.Linear(d_model, d_model, device: device, dtype: dtype);
                RegisterComponents();
                _reset_parameters();
            }

            /// <summary>Reset module parameters.</summary>
            void _reset_parameters()
            {
                using (torch.no_grad())
                {
                    torch.nn.init.constant_(sampling_offsets.weight, 0.0);
                    var thetas = torch.arange(n_heads, dtype: torch.ScalarType.Float32) * (2.0 * Math.PI / n_heads);
                    var grid_init = torch.stack(new[] { thetas.cos(), thetas.sin() }, -1);
                    grid_init = (grid_init / grid_init.abs().max(-1, keepdim: true).values)
                        .view(n_heads, 1, 1, 2)
                        .repeat(1, n_levels, n_points, 1);
                    for (int i = 0; i < n_points; i++)
                    {
                        var scaled = grid_init[torch.TensorIndex.Ellipsis, torch.TensorIndex.Tensor(i), torch.TensorIndex.Ellipsis] * (i + 1);
                        grid_init[torch.TensorIndex.Ellipsis, torch.TensorIndex.Tensor(i), torch.TensorIndex.Ellipsis] = scaled;
                    }
                    sampling_offsets.bias.copy_(grid_init.reshape(-1));
                    torch.nn.init.constant_(attention_weights.weight, 0.0);
                    torch.nn.init.constant_(attention_weights.bias, 0.0);
                    torch.nn.init.xavier_uniform_(value_proj.weight);
                    torch.nn.init.constant_(value_proj.bias, 0.0);
                    torch.nn.init.xavier_uniform_(output_proj.weight);
                    torch.nn.init.constant_(output_proj.bias, 0.0);
                }
            }

            /// <summary>
            /// Perform forward pass for multiscale deformable attention.
            ///
            /// Args:
            ///     query (torch.Tensor): Query tensor with shape [bs, query_length, C].
            ///     refer_bbox (torch.Tensor): Reference bounding boxes with shape [bs, query_length, 1, 2 or 4], range
            ///         in [0, 1], top-left (0,0), bottom-right (1, 1). The size-1 axis broadcasts across n_levels.
            ///     value (torch.Tensor): Value tensor with shape [bs, value_length, C].
            ///     value_shapes (list): List with shape [n_levels, 2], [(H_0, W_0), (H_1, W_1), ...].
            ///     value_mask (torch.Tensor, optional): Mask tensor with shape [bs, value_length], True for padding
            ///         elements, False for non-padding elements.
            ///
            /// Returns:
            ///     (torch.Tensor): Output tensor with shape [bs, Length_query, C].
            ///
            /// References:
            ///     https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
            /// </summary>
            public torch.Tensor forward(torch.Tensor query, torch.Tensor refer_bbox, torch.Tensor value,
                (int h, int w)[] value_shapes, torch.Tensor? value_mask = null)
            {
                long bs = query.shape[0], len_q = query.shape[1];
                long len_v = value.shape[1];
                long total_hw = 0; foreach (var s in value_shapes) total_hw += (long)s.h * s.w;
                if (total_hw != len_v) throw new ArgumentException("sum(H_i*W_i) != len_v");

                value = value_proj.forward(value);
                if (value_mask is not null)
                    value = value.masked_fill(value_mask.unsqueeze(-1), 0f);
                value = value.view(bs, len_v, n_heads, d_model / n_heads);
                // Fold (n_levels, n_points) into one axis so every traced tensor stays at rank <= 5
                int n_total_points = n_levels * n_points;
                var sampling_offsets = this.sampling_offsets.forward(query).view(bs, len_q, n_heads, n_total_points, 2);
                var attention_weights = this.attention_weights.forward(query).view(bs, len_q, n_heads, n_total_points);
                attention_weights = torch.nn.functional.softmax(attention_weights, -1);
                int num_points = (int)refer_bbox.shape[refer_bbox.shape.Length - 1];   // python shape[-1] — C# long[] has no negative indexing / C#数组无负索引
                torch.Tensor sampling_locations;
                if (num_points == 2)
                {
                    var flip = new torch.Tensor[value_shapes.Length];
                    for (int i = 0; i < value_shapes.Length; i++)
                        flip[i] = torch.tensor(new double[] { value_shapes[i].w, value_shapes[i].h },
                            dtype: query.dtype, device: query.device);
                    var offset_normalizer = torch.stack(flip);   // flip(-1)
                    offset_normalizer = offset_normalizer.unsqueeze(1).expand(-1, n_points, -1).reshape(n_total_points, 2);
                    sampling_locations = refer_bbox.unsqueeze(2) + sampling_offsets / offset_normalizer;
                }
                else if (num_points == 4)
                {
                    var rb = refer_bbox.unsqueeze(2);
                    sampling_locations = rb[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(null, 2)]
                        + sampling_offsets / n_points
                        * rb[torch.TensorIndex.Ellipsis, torch.TensorIndex.Slice(2, null)] * 0.5;
                }
                else
                {
                    throw new ArgumentException($"Last dim of reference_points must be 2 or 4, but got {num_points}.");
                }
                var output = utils.multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights);
                return output_proj.forward(output);
            }
        }

        /// <summary>
        /// Deformable Transformer Decoder Layer inspired by PaddleDetection and Deformable-DETR implementations.
        ///
        /// This class implements a single decoder layer with self-attention, cross-attention using multiscale
        /// deformable attention, and a feedforward network.
        ///
        /// Attributes:
        ///     self_attn (nn.MultiheadAttention): Self-attention module.
        ///     dropout1 (nn.Dropout): Dropout after self-attention.
        ///     norm1 (nn.LayerNorm): Layer normalization after self-attention.
        ///     cross_attn (MSDeformAttn): Cross-attention module.
        ///     dropout2 (nn.Dropout): Dropout after cross-attention.
        ///     norm2 (nn.LayerNorm): Layer normalization after cross-attention.
        ///     linear1 (nn.Linear): First linear layer in the feedforward network.
        ///     act (nn.Module): Activation function.
        ///     dropout3 (nn.Dropout): Dropout in the feedforward network.
        ///     linear2 (nn.Linear): Second linear layer in the feedforward network.
        ///     dropout4 (nn.Dropout): Dropout after the feedforward network.
        ///     norm3 (nn.LayerNorm): Layer normalization after the feedforward network.
        ///
        /// References:
        ///     https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        ///     https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
        /// </summary>
        internal class DeformableTransformerDecoderLayer : torch.nn.Module
        {
            private readonly MultiheadAttention self_attn;
            private readonly Dropout dropout1;
            private readonly LayerNorm norm1;
            private readonly MSDeformAttn cross_attn;
            private readonly Dropout dropout2;
            private readonly LayerNorm norm2;
            private readonly Linear linear1, linear2;
            private readonly torch.nn.Module<torch.Tensor, torch.Tensor> act;
            private readonly Dropout dropout3, dropout4;
            private readonly LayerNorm norm3;

            /// <summary>
            /// Initialize the DeformableTransformerDecoderLayer with the given parameters.
            ///
            /// Args:
            ///     d_model (int): Model dimension.
            ///     n_heads (int): Number of attention heads.
            ///     d_ffn (int): Dimension of the feedforward network.
            ///     dropout (float): Dropout probability.
            ///     act (nn.Module, optional): Activation function. Defaults to nn.ReLU() if None.
            ///     n_levels (int): Number of feature levels.
            ///     n_points (int): Number of sampling points per attention head per feature level.
            /// </summary>
            internal DeformableTransformerDecoderLayer(int d_model = 256, int n_heads = 8, int d_ffn = 1024,
                double dropout = 0.0, torch.nn.Module<torch.Tensor, torch.Tensor>? act = null,
                int n_levels = 4, int n_points = 4, torch.Device? device = null, torch.ScalarType? dtype = null)
                : base(nameof(DeformableTransformerDecoderLayer))
            {
                // Self attention
                this.self_attn = torch.nn.MultiheadAttention(embedded_dim: d_model, num_heads: n_heads, dropout: dropout);
                this.dropout1 = torch.nn.Dropout(dropout);
                this.norm1 = torch.nn.LayerNorm(d_model, device: device, dtype: dtype);

                // Cross attention
                this.cross_attn = new MSDeformAttn(d_model, n_levels, n_heads, n_points, device, dtype);
                this.dropout2 = torch.nn.Dropout(dropout);
                this.norm2 = torch.nn.LayerNorm(d_model, device: device, dtype: dtype);

                // FFN
                this.linear1 = torch.nn.Linear(d_model, d_ffn, device: device, dtype: dtype);
                this.act = act ?? torch.nn.ReLU();
                this.dropout3 = torch.nn.Dropout(dropout);
                this.linear2 = torch.nn.Linear(d_ffn, d_model, device: device, dtype: dtype);
                this.dropout4 = torch.nn.Dropout(dropout);
                this.norm3 = torch.nn.LayerNorm(d_model, device: device, dtype: dtype);
                RegisterComponents();
            }

            /// <summary>Add positional embeddings to the input tensor, if provided.</summary>
            protected static torch.Tensor with_pos_embed(torch.Tensor tensor, torch.Tensor? pos)
            {
                return pos is null ? tensor : tensor + pos;
            }

            /// <summary>
            /// Perform forward pass through the Feed-Forward Network part of the layer.
            ///
            /// Args:
            ///     tgt (torch.Tensor): Input tensor.
            ///
            /// Returns:
            ///     (torch.Tensor): Output tensor after FFN.
            /// </summary>
            protected torch.Tensor forward_ffn(torch.Tensor tgt)
            {
                var tgt2 = linear2.forward(dropout3.forward(act.forward(linear1.forward(tgt))));
                tgt = tgt + dropout4.forward(tgt2);
                return norm3.forward(tgt);
            }

            /// <summary>
            /// Perform the forward pass through the entire decoder layer.
            ///
            /// Args:
            ///     embed (torch.Tensor): Input embeddings with shape (bs, num_queries, d_model).
            ///     refer_bbox (torch.Tensor): Normalized reference bounding boxes with shape (bs, num_queries, 4).
            ///     feats (torch.Tensor): Flattened multi-level feature maps with shape (bs, sum(H_i * W_i), d_model).
            ///     shapes (list): Feature shapes [(H_0, W_0), ..., (H_{L-1}, W_{L-1})].
            ///     padding_mask (torch.Tensor, optional): Padding mask.
            ///     attn_mask (torch.Tensor, optional): Attention mask.
            ///     query_pos (torch.Tensor, optional): Query position embeddings.
            ///
            /// Returns:
            ///     (torch.Tensor): Output tensor after decoder layer.
            /// </summary>
            public torch.Tensor forward(torch.Tensor embed, torch.Tensor refer_bbox, torch.Tensor feats,
                (int h, int w)[] shapes, torch.Tensor? padding_mask = null, torch.Tensor? attn_mask = null,
                torch.Tensor? query_pos = null)
            {
                // Self attention
                var qk = with_pos_embed(embed, query_pos);
                var tgt = self_attn.forward(qk.transpose(0, 1), qk.transpose(0, 1), embed.transpose(0, 1), null, true, null).Item1.transpose(0, 1);
                embed = embed + dropout1.forward(tgt);
                embed = norm1.forward(embed);

                // Cross attention
                tgt = cross_attn.forward(with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes, padding_mask);
                embed = embed + dropout2.forward(tgt);
                embed = norm2.forward(embed);

                // FFN
                return forward_ffn(embed);
            }
        }

        /// <summary>
        /// Deformable Transformer Decoder based on PaddleDetection implementation.
        ///
        /// This class implements a complete deformable transformer decoder with multiple decoder layers and
        /// prediction heads for bounding box regression and classification.
        ///
        /// Attributes:
        ///     layers (ModuleList): List of decoder layers.
        ///     num_layers (int): Number of decoder layers.
        ///     hidden_dim (int): Hidden dimension.
        ///     eval_idx (int): Index of the layer to use during evaluation.
        ///
        /// References:
        ///     https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        /// </summary>
        internal class DeformableTransformerDecoder : torch.nn.Module
        {
            internal bool Training { get; set; } = false;   // train/eval mode flag (Python self.training equivalent)
            private readonly ModuleList<DeformableTransformerDecoderLayer> layers;
            private readonly int num_layers, hidden_dim, eval_idx;

            /// <summary>
            /// Initialize the DeformableTransformerDecoder with the given parameters.
            ///
            /// Args:
            ///     hidden_dim (int): Hidden dimension.
            ///     decoder_layer_factory (Func&lt;DeformableTransformerDecoderLayer&gt;): Factory creating each decoder
            ///         layer (C# port of `_get_clones(decoder_layer, num_layers)`).
            ///     num_layers (int): Number of decoder layers.
            ///     eval_idx (int): Index of the layer to use during evaluation.
            /// </summary>
            internal DeformableTransformerDecoder(int hidden_dim,
                System.Func<DeformableTransformerDecoderLayer> decoder_layer_factory,
                int num_layers, int eval_idx = -1, torch.Device? device = null, torch.ScalarType? dtype = null)
                : base(nameof(DeformableTransformerDecoder))
            {
                var ls = new System.Collections.Generic.List<DeformableTransformerDecoderLayer>(num_layers);
                for (int i = 0; i < num_layers; i++) ls.Add(decoder_layer_factory());
                this.layers = new ModuleList<DeformableTransformerDecoderLayer>(ls.ToArray());
                this.num_layers = num_layers;
                this.hidden_dim = hidden_dim;
                this.eval_idx = eval_idx >= 0 ? eval_idx : num_layers + eval_idx;
                RegisterComponents();
            }

            /// <summary>
            /// Perform the forward pass through the entire decoder.
            ///
            /// Args:
            ///     embed (torch.Tensor): Decoder embeddings with shape (bs, num_queries, hidden_dim).
            ///     refer_bbox (torch.Tensor): Reference bounding boxes as unnormalized logits (sigmoid applied
            ///         internally).
            ///     feats (torch.Tensor): Flattened multi-level image features with shape (bs, sum(H_i*W_i), hidden_dim).
            ///     shapes (list): Feature shapes [(H_0, W_0), ..., (H_{L-1}, W_{L-1})].
            ///     bbox_head (ModuleList): Per-layer bounding box prediction heads.
            ///     score_head (ModuleList): Per-layer score prediction heads.
            ///     pos_mlp (nn.Module): MLP generating query position embeddings from reference boxes.
            ///     attn_mask (torch.Tensor, optional): Attention mask.
            ///     padding_mask (torch.Tensor, optional): Padding mask.
            ///
            /// Returns:
            ///     dec_bboxes (torch.Tensor): Decoded normalized bounding boxes with shape (L, bs, num_queries, 4).
            ///     dec_cls (torch.Tensor): Classification logits with shape (L, bs, num_queries, nc).
            /// </summary>
            public (torch.Tensor dec_bboxes, torch.Tensor dec_cls) forward(
                torch.Tensor embed, torch.Tensor refer_bbox, torch.Tensor feats, (int h, int w)[] shapes,
                ModuleList<Linear> bbox_head, ModuleList<Linear> score_head,
                torch.nn.Module<torch.Tensor, torch.Tensor> pos_mlp,
                torch.Tensor? attn_mask = null, torch.Tensor? padding_mask = null)
            {
                var output = embed;
                var dec_bboxes = new System.Collections.Generic.List<torch.Tensor>();
                var dec_cls = new System.Collections.Generic.List<torch.Tensor>();
                torch.Tensor? last_refined_bbox = null;
                refer_bbox = refer_bbox.sigmoid();
                bool training = this.Training;
                for (int i = 0; i < layers.Count; i++)
                {
                    output = layers[i].forward(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp.forward(refer_bbox));

                    var bbox = bbox_head[i].forward(output);
                    var refined_bbox = torch.sigmoid(bbox + utils.inverse_sigmoid(refer_bbox));

                    if (training)
                    {
                        dec_cls.Add(score_head[i].forward(output));
                        if (i == 0)
                        {
                            dec_bboxes.Add(refined_bbox);
                        }
                        else
                        {
                            dec_bboxes.Add(torch.sigmoid(bbox + utils.inverse_sigmoid(last_refined_bbox!)));
                        }
                    }
                    else if (i == eval_idx)
                    {
                        dec_cls.Add(score_head[i].forward(output));
                        dec_bboxes.Add(refined_bbox);
                        break;
                    }

                    last_refined_bbox = refined_bbox;
                    refer_bbox = training ? refined_bbox.detach() : refined_bbox;
                }

                return (torch.stack(dec_bboxes.ToArray()), torch.stack(dec_cls.ToArray()));
            }
        }
    }
}
