using System.Numerics;
using TorchSharp;
using TorchSharp.Modules;
using YoloSharp.Types;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace YoloSharp.Modules
{
	internal class Modules
	{
		internal class C3 : Module<Tensor, Tensor>
		{
			internal readonly Convs.Conv cv1;
			internal readonly Convs.Conv cv2;
			internal readonly Convs.Conv cv3;
			protected Sequential m;

			internal C3(int inChannels, int outChannels, int n = 1, bool shortcut = true, int groups = 1, float e = 0.5f, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(C3))
			{
				int c = (int)(outChannels * e);
				cv1 = new Convs.Conv(inChannels, c, 1, 1, device: device, dtype: dtype);
				cv2 = new Convs.Conv(inChannels, c, 1, 1, device: device, dtype: dtype);
				cv3 = new Convs.Conv(2 * c, outChannels, 1, device: device, dtype: dtype);

				m = Sequential(Enumerable.Range(0, n).Select(i => new Block.Bottleneck(c, c, k: new int[] { 1, 3 }, shortcut: shortcut, g: groups, e: 1.0f, device: device, dtype: dtype)).ToArray());
				//for (int i = 0; i < n; i++)
				//{
				//	m.append(new Bottleneck(c, c, (1, 3), shortcut, groups, e: 1.0f, device: device, dtype: dtype));
				//}
				//RegisterComponents();
			}

			public override Tensor forward(Tensor input)
			{
				return cv3.forward(cat(new Tensor[] { m.forward(cv1.forward(input)), cv2.forward(input) }, 1));
			}
		}

		internal class C3k : C3
		{
			internal C3k(int inChannels, int outChannels, int n = 1, bool shortcut = true, int groups = 1, float e = 0.5f, Device? device = null, torch.ScalarType? dtype = null) : base(inChannels, outChannels, n, shortcut, groups, e, device, dtype)
			{
				int c = (int)(outChannels * e);
				m = Sequential(Enumerable.Range(0, n).Select(_ => new Block.Bottleneck(c, c, k: new int[] { 3, 3 }, shortcut: shortcut, g: groups, e: 1.0f, device: device, dtype: dtype)).ToArray());
			}
		}

		internal class C2f : Module<Tensor, Tensor>
		{
			internal readonly Convs.Conv cv1;
			internal readonly Convs.Conv cv2;
			internal readonly int c;
			internal Sequential m;
			internal C2f(int inChannels, int outChannels, int n = 1, bool shortcut = false, int groups = 1, float e = 0.5f, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(C2f))
			{
				c = (int)(outChannels * e);
				cv1 = new Convs.Conv(inChannels, 2 * c, 1, 1, device: device, dtype: dtype);
				cv2 = new Convs.Conv((2 + n) * c, outChannels, 1, device: device, dtype: dtype);  // optional act=FReLU(outChannels)
				m = Sequential();
				for (int i = 0; i < n; i++)
				{
					m = m.append(new Block.Bottleneck(c, c, k: new int[] { 3, 3 }, shortcut: shortcut, g: groups, e: 1, device: device, dtype: dtype));
				}
				RegisterComponents();
			}

			public override Tensor forward(Tensor input)
			{
				using var _ = NewDisposeScope();
				var y = cv1.forward(input).chunk(2, 1).ToList();
				for (int i = 0; i < m.Count; i++)
				{
					y.Add(m[i].call(y.Last()));
				}
				Tensor result = cv2.forward(cat(y, 1));
				return result.MoveToOuterDisposeScope();
			}
		}

		internal class C3k2 : Module<Tensor, Tensor>
		{
			internal readonly Convs.Conv cv1;
			internal readonly Convs.Conv cv2;
			internal readonly ModuleList<Module> m;
			internal readonly int c;
			internal C3k2(int inChannels, int outChannels, int n = 1, bool c3k = false, float e = 0.5f, int groups = 1, bool shortcut = true, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(C3k2))
			{
				c = (int)(outChannels * e);
				cv1 = new Convs.Conv(inChannels, 2 * c, 1, 1, device: device, dtype: dtype);
				cv2 = new Convs.Conv((2 + n) * c, outChannels, 1, device: device, dtype: dtype);  // optional act=FReLU(outChannels)
				m = new ModuleList<Module>();
				for (int i = 0; i < n; i++)
				{
					if (c3k)
					{
						m.append(new C3k(c, c, 2, shortcut, groups, device: device, dtype: dtype));
					}
					else
					{
						m.append(new Block.Bottleneck(c, c, k: new int[] { 3, 3 }, shortcut: shortcut, g: groups, device: device, dtype: dtype));
					}
				}
				RegisterComponents();
			}

			public override Tensor forward(Tensor input)
			{
				using (NewDisposeScope())
				{
					List<Tensor> y = cv1.forward(input).chunk(2, 1).ToList();
					for (int i = 0; i < m.Count; i++)
					{
						y.Add(((Module<Tensor, Tensor>)m[i]).forward(y.Last()));
					}
					Tensor result = cv2.forward(cat(y, 1));
					return result.MoveToOuterDisposeScope();
				}
			}
		}

		internal class C2PSA : Module<Tensor, Tensor>
		{
			private readonly int c;
			private readonly Convs.Conv cv1;
			private readonly Convs.Conv cv2;
			private readonly Sequential m;

			internal C2PSA(int inChannel, int outChannel, int n = 1, float e = 0.5f, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(C2PSA))
			{
				if (inChannel != outChannel)
				{
					throw new ArgumentException("in channel not equals to out channel");
				}
				c = (int)(inChannel * e);
				cv1 = new Convs.Conv(inChannel, 2 * c, 1, 1, device: device, dtype: dtype);
				cv2 = new Convs.Conv(2 * c, outChannel, 1, device: device, dtype: dtype);

				m = Sequential(Enumerable.Range(0, n).Select(_ => new PSABlock(c, attn_ratio: 0.5f, num_heads: c / 64, device: device, dtype: dtype)).ToArray());

				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				using var _ = NewDisposeScope();

				Tensor[] ab = cv1.forward(x).split(new long[] { c, c }, dim: 1);
				Tensor a = ab[0];
				Tensor b = ab[1];
				b = m.forward(b);
				return cv2.forward(cat(new Tensor[] { a, b }, 1)).MoveToOuterDisposeScope();
			}
		}

		internal class PSABlock : Module<Tensor, Tensor>
		{
			private readonly Attention attn; // can use ScaledDotProductAttention instead
			private readonly Sequential ffn;
			private readonly bool add;

			internal PSABlock(int c, float attn_ratio = 0.5f, int num_heads = 8, bool shortcut = true, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(PSABlock))
			{
				attn = new Attention(c, num_heads, attn_ratio, attentionType: AttentionType.SelfAttention, device: device, dtype: dtype);
				ffn = Sequential(new Convs.Conv(c, c * 2, 1, device: device, dtype: dtype), new Convs.Conv(c * 2, c, 1, device: device, dtype: dtype));
				add = shortcut;
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				x = add ? x + attn.forward(x) : attn.forward(x);
				x = add ? x + ffn.forward(x) : ffn.forward(x);
				return x;
			}
		}

		internal class Attention : Module<Tensor, Tensor>
		{
			private readonly int num_heads;
			private readonly int head_dim;
			private readonly int key_dim;
			private readonly float scale;

			private readonly Convs.Conv qkv;
			private readonly Convs.Conv proj;
			private readonly Convs.Conv pe;

			private AttentionType attentionType;

			internal Attention(int dim, int num_heads = 8, float attn_ratio = 0.5f, AttentionType attentionType = AttentionType.SelfAttention, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(Attention))
			{
				this.num_heads = num_heads;
				head_dim = dim / num_heads;
				key_dim = (int)(head_dim * attn_ratio);
				scale = (float)Math.Pow(key_dim, -0.5);

				int nh_kd = key_dim * num_heads;
				int h = dim + nh_kd * 2;

				qkv = new Convs.Conv(dim, h, 1, device: device, dtype: dtype);
				proj = new Convs.Conv(dim, dim, 1, device: device, dtype: dtype);
				pe = new Convs.Conv(dim, dim, 3, 1, g: dim, device: device, dtype: dtype);

				this.attentionType = attentionType;
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				using var _ = NewDisposeScope();

				long B = x.shape[0];
				long C = x.shape[1];
				long H = x.shape[2];
				long W = x.shape[3];

				long N = H * W;

				Tensor qkv = this.qkv.forward(x);

				Tensor[] qkv_mix = qkv.view(B, num_heads, key_dim * 2 + head_dim, N).split(new long[] { key_dim, key_dim, head_dim }, dim: 2);
				Tensor q = qkv_mix[0];
				Tensor k = qkv_mix[1];
				Tensor v = qkv_mix[2];

				switch (attentionType)
				{
					case AttentionType.SelfAttention:
						{
							Tensor attn = q.transpose(-2, -1).matmul(k) * scale;
							attn = attn.softmax(dim: -1);
							x = v.matmul(attn.transpose(-2, -1)).view(B, C, H, W) + pe.forward(v.reshape(B, C, H, W));
							break;
						}
					case AttentionType.ScaledDotProductAttention:
						{
							q = q.transpose(-2, -1); // [B, num_heads, N, key_dim]
							k = k.transpose(-2, -1); // [B, num_heads, N, key_dim]
							v = v.transpose(-2, -1); // [B, num_heads, N, head_dim]

							Tensor attn_output = functional.scaled_dot_product_attention(q, k, v, is_casual: false);

							attn_output = attn_output.transpose(-2, -1); // [B, num_heads, N, head_dim]
							attn_output = attn_output.contiguous();

							if (B * num_heads * N * head_dim != B * C * H * W)
							{
								throw new InvalidOperationException("Shape mismatch: Cannot reshape attn_output to [B, C, H, W].");
							}

							attn_output = attn_output.view(B, C, H, W);
							x = attn_output + pe.forward(attn_output);
							break;
						}
					default:
						{
							throw new NotImplementedException($"Attention type {attentionType} is not implemented.");
						}
				}

				x = proj.forward(x);

				return x.MoveToOuterDisposeScope();
			}
		}

		internal class SCDown : Module<Tensor, Tensor>
		{
			private readonly Convs.Conv cv1;
			private readonly Convs.Conv cv2;
			internal SCDown(int inChannel, int outChannel, int k, int s, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(SCDown))
			{
				cv1 = new Convs.Conv(inChannel, outChannel, 1, 1, device: device, dtype: dtype);
				cv2 = new Convs.Conv(outChannel, outChannel, k: k, s: s, g: outChannel, device: device, dtype: dtype);
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				return cv2.forward(cv1.forward(x));
			}
		}

		internal class C2fCIB : Module<Tensor, Tensor>
		{
			private readonly Convs.Conv cv1;
			private readonly Convs.Conv cv2;
			internal readonly Sequential m;
			internal C2fCIB(int inChannels, int outChannels, int n = 1, bool shortcut = false, bool lk = false, int g = 1, float e = 0.5f, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(C2fCIB))
			{
				int c = (int)(outChannels * e);
				cv1 = new Convs.Conv(inChannels, 2 * c, 1, 1, device: device, dtype: dtype);
				cv2 = new Convs.Conv((2 + n) * c, outChannels, 1, device: device, dtype: dtype);  // optional act=FReLU(outChannels)
				m = Sequential();
				for (int i = 0; i < n; i++)
				{
					m = m.append(new CIB(c, c, shortcut, e: 1.0f, lk: lk, device: device, dtype: dtype));
				}
				RegisterComponents();
			}

			public override Tensor forward(Tensor input)
			{
				using var _ = NewDisposeScope();

				var y = cv1.forward(input).chunk(2, 1).ToList();
				for (int i = 0; i < m.Count; i++)
				{
					y.Add(m[i].call(y.Last()));
				}
				return cv2.forward(cat(y, 1)).MoveToOuterDisposeScope();
			}
		}

		internal class CIB : Module<Tensor, Tensor>
		{
			private readonly Sequential cv1;
			private readonly bool add;
			internal CIB(int inChannels, int outChannels, bool shortcut = true, float e = 0.5f, bool lk = false, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(CIB))
			{
				int c = (int)(outChannels * e);  // hidden channels
				cv1 = Sequential(
					new Convs.Conv(inChannels, inChannels, 3, g: inChannels, device: device, dtype: dtype),
					new Convs.Conv(inChannels, 2 * c, 1, device: device, dtype: dtype),
					lk ? new RepVGGDW(2 * c, device: device, dtype: dtype) : new Convs.Conv(2 * c, 2 * c, 3, g: 2 * c, device: device, dtype: dtype),
					new Convs.Conv(2 * c, outChannels, 1, device: device, dtype: dtype),
					new Convs.Conv(outChannels, outChannels, 3, g: outChannels, device: device, dtype: dtype));
				add = shortcut && inChannels == outChannels;

				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				return add ? x + cv1.forward(x) : cv1.forward(x);
			}
		}


		/// <summary>
		/// Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.
		/// This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
		/// processing.It supports both area-attention and standard convolution modes.
		/// </summary>
		internal class A2C2f : Module<Tensor, Tensor>
		{
			/// <summary>
			/// Initial 1x1 convolution layer that reduces x channels to hidden channels.
			/// </summary>
			private readonly Convs.Conv cv1;

			/// <summary>
			/// Final 1x1 convolution layer that processes concatenated features.
			/// </summary>
			private readonly Convs.Conv cv2;

			/// <summary>
			/// Learnable parameter for residual scaling when using area attention.
			/// </summary>
			private readonly Parameter? gamma;

			/// <summary>
			/// List of either ABlock or C3k modules for feature processing.
			/// </summary>
			private readonly Sequential m;

			/// <summary>
			/// Initialize Area-Attention C2f module.
			/// </summary>
			/// <param name="c1">Number of x channels.</param>
			/// <param name="c2">Number of output channels.</param>
			/// <param name="n">Number of ABlock or C3k modules to stack.</param>
			/// <param name="a2">Whether to use area attention blocks. If False, uses C3k blocks instead.</param>
			/// <param name="area">Number of areas the feature map is divided.</param>
			/// <param name="residual">Whether to use residual connections with learnable gamma parameter.</param>
			/// <param name="mlp_ratio">Expansion ratio for MLP hidden dimension.</param>
			/// <param name="e">Channel expansion ratio for hidden channels.</param>
			/// <param name="g">Number of groups for grouped convolutions.</param>
			/// <param name="shortcut">Whether to use shortcut connections in C3k blocks.</param>
			/// <param name="device"></param>
			/// <param name="dtype"></param>
			/// <exception cref="Exception"></exception>
			internal A2C2f(int c1, int c2, int n = 1, bool a2 = true, int area = 1, bool residual = false, float mlp_ratio = 2.0f, float e = 0.5f, int g = 1, bool shortcut = true, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(A2C2f))
			{
				int c_ = (int)(c2 * e);
				if (c_ % 32 != 0)
				{
					throw new Exception("Dimension of ABlock be a multiple of 32.");
				}
				cv1 = new Convs.Conv(c1, c_, 1, 1, device: device, dtype: dtype);
				cv2 = new Convs.Conv((1 + n) * c_, c2, 1, device: device, dtype: dtype);

				gamma = a2 && residual ? Parameter(0.01 * ones(c2, device: device, dtype: dtype), requires_grad: true) : null;
				m = Sequential();
				for (int i = 0; i < n; i++)
				{
					if (a2)
					{
						var seq = Sequential();
						for (int j = 0; j < 2; j++)
						{
							seq.append(new ABlock(c_, c_ / 32, mlp_ratio, area, device: device, dtype: dtype));
						}
						m.append(seq);
					}
					else
					{
						C3k c3k = new C3k(c_, c_, 2, shortcut, g, device: device, dtype: dtype);
						m.append(c3k);
					}
				}
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				using var _ = NewDisposeScope();

				List<Tensor> y = new List<Tensor> { cv1.forward(x) };

				foreach (var module in m.children())
				{
					y.Add(((Module<Tensor, Tensor>)module).forward(y.Last()));
				}

				Tensor y_cat = cat(y.ToArray(), 1);
				Tensor output = cv2.forward(y_cat);

				if (gamma is not null)
				{
					Tensor gamma_view = gamma.view(new long[] { -1, gamma.shape[0], 1, 1 });
					return (x + gamma_view * output).MoveToOuterDisposeScope();
				}
				return output.MoveToOuterDisposeScope();
			}
		}

		/// <summary>
		/// Area-attention block module for efficient feature extraction in YOLO models.
		/// This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
		/// It uses a novel area-based attention approach that is more efficient than traditional self-attention while
		/// maintaining effectiveness
		/// </summary>
		internal class ABlock : Module<Tensor, Tensor>
		{
			private readonly AAttn attn;
			private readonly Sequential mlp;
			private readonly Action<Module> initWeights;  // Weight initialization function
			internal ABlock(int dim, int num_heads, float mlp_ratio = 1.2f, int area = 1, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(ABlock))
			{
				attn = new AAttn(dim, num_heads: num_heads, area: area, attentionType: AttentionType.SelfAttention, device: device, dtype: dtype);
				int mlp_hidden_dim = (int)(dim * mlp_ratio);
				mlp = Sequential(new Convs.Conv(dim, mlp_hidden_dim, 1, device: device, dtype: dtype), new Convs.Conv(mlp_hidden_dim, dim, 1, device: device, dtype: dtype));
				// Initialize weights
				initWeights = m =>
				{
					if (m is Conv2d conv)
					{
						init.trunc_normal_(conv.weight, std: 0.02);
						if (conv.bias is not null)
							init.constant_(conv.bias, 0);
					}
				};
				apply(initWeights);
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				using var _ = NewDisposeScope();

				x = x + attn.forward(x);
				return (x + mlp.forward(x)).MoveToOuterDisposeScope();
			}
		}


		/// <summary>
		/// Area-attention module for YOLO models, providing efficient attention mechanisms.
		/// This module implements an area-based attention mechanism that processes x features in a spatially-aware manner,
		/// making it particularly effective for object detection tasks.

		/// </summary>
		internal class AAttn : Module<Tensor, Tensor>
		{
			private readonly int area;
			private readonly int num_heads;
			private readonly int head_dim;

			private readonly Convs.Conv qkv;
			private readonly Convs.Conv proj;
			private readonly Convs.Conv pe;
			private readonly AttentionType attentionType;

			internal AAttn(int dim, int num_heads, int area = 1, AttentionType attentionType = AttentionType.SelfAttention, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(AAttn))
			{
				this.attentionType = attentionType;
				this.area = area;
				this.num_heads = num_heads;
				head_dim = dim / num_heads;
				int all_head_dim = head_dim * this.num_heads;

				qkv = new Convs.Conv(dim, all_head_dim * 3, 1, device: device, dtype: dtype);
				proj = new Convs.Conv(all_head_dim, dim, 1, device: device, dtype: dtype);
				pe = new Convs.Conv(all_head_dim, dim, 7, 1, 3, g: dim, bias: true, device: device, dtype: dtype);
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				using var _ = NewDisposeScope();

				long B = x.shape[0];
				long C = x.shape[1];
				long H = x.shape[2];
				long W = x.shape[3];
				long N = H * W;

				Tensor qkv = this.qkv.forward(x).flatten(2).transpose(1, 2);

				if (area > 1)
				{
					qkv = qkv.reshape(B * area, N / area, C * 3);
					B = qkv.shape[0];
					N = qkv.shape[1];
				}
				Tensor[] qkv_mix = qkv.view(B, N, num_heads, head_dim * 3).permute(0, 2, 3, 1).split(new long[] { head_dim, head_dim, head_dim }, dim: 2);
				Tensor q = qkv_mix[0];
				Tensor k = qkv_mix[1];
				Tensor v = qkv_mix[2];
				if (attentionType == AttentionType.SelfAttention)
				{
					Tensor attn = q.transpose(-2, -1).matmul(k) * (float)Math.Pow(head_dim, -0.5);
					attn = attn.softmax(dim: -1);
					x = v.matmul(attn.transpose(-2, -1));
					x = x.permute(0, 3, 1, 2);
					v = v.permute(0, 3, 1, 2);
				}
				else if (attentionType == AttentionType.ScaledDotProductAttention)
				{
					q = q.permute(0, 1, 3, 2); // [B, nh, N, hd]
					k = k.permute(0, 1, 3, 2);
					v = v.permute(0, 1, 3, 2);

					x = functional.scaled_dot_product_attention(q, k, v);

					x = x.permute(0, 2, 1, 3)  // [B, N, nh, hd]
						.reshape(B, N, -1);     // [B, N, C]

					v = v.permute(0, 2, 1, 3)  // [B, N, nh, hd]
						.reshape(B, N, -1);     // [B, N, C]
				}
				else
				{
					throw new NotImplementedException($"Attention type {attentionType} is not implemented.");
				}

				if (area > 1)
				{
					x = x.reshape(B / area, N * area, C);
					v = v.reshape(B / area, N * area, C);
					B = x.shape[0];
					N = x.shape[1];
				}

				x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous();
				v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous();
				x = x + pe.forward(v);
				return proj.forward(x).MoveToOuterDisposeScope();

			}
		}

		internal class RepVGGDW : Module<Tensor, Tensor>
		{
			private readonly Convs.Conv conv;
			private readonly Convs.Conv conv1;
			private readonly int dim;
			private readonly Module<Tensor, Tensor> act;
			internal RepVGGDW(int ed, Device? device = null, Module<Tensor, Tensor>? act = null, torch.ScalarType? dtype = null) : base(nameof(RepVGGDW))
			{
				conv = new Convs.Conv(ed, ed, 7, 1, 3, g: ed, act: act, device: device, dtype: dtype);
				conv1 = new Convs.Conv(ed, ed, 3, 1, 1, g: ed, act: act, device: device, dtype: dtype);
				dim = ed;
				this.act = act ?? SiLU();

				RegisterComponents();
			}
			public override Tensor forward(Tensor x)
			{
				return act.forward(conv.forward(x) + conv1.forward(x));
			}
		}

		
		internal class Concat : Module<Tensor[], Tensor>
		{
			private readonly int dim;
			internal Concat(int dim = 1) : base(nameof(Concat))
			{
				this.dim = dim;
			}

			public override Tensor forward(Tensor[] input)
			{
				return concat(input, dim: dim);
			}
		}


	}
}
