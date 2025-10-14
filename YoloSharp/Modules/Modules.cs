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
		internal class Conv : Module<Tensor, Tensor>
		{
			private readonly Conv2d conv;
			private readonly BatchNorm2d bn;
			private readonly Module<Tensor, Tensor> act;
			private readonly double eps = 0.001;
			private readonly double momentum = 0.03;

			internal Conv(int in_channels, int out_channels, int kernel_size, int stride = 1, int? padding = null, int groups = 1, int d = 1, bool bias = false, Func<Module<Tensor, Tensor>>? act = null, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(Conv))
			{
				padding = padding ?? kernel_size / 2;
				conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding.Value, groups: groups, bias: bias, dilation: d, device: device, dtype: dtype);
				bn = BatchNorm2d(out_channels, eps: eps, momentum: momentum, track_running_stats: true, device: device, dtype: dtype);
				this.act = (act ?? SiLU)();
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				if (training)
				{
					return act.forward(bn.forward(conv.forward(x))).MoveToOuterDisposeScope();
				}
				else
				{
					using (NewDisposeScope())
					using (no_grad())
					{
						// Prepare filters
						Conv2d fusedconv = Conv2d(conv.in_channels, conv.out_channels, kernel_size: (conv.kernel_size[0], conv.kernel_size[1]), stride: (conv.stride[0], conv.stride[1]), padding: (conv.padding[0], conv.padding[1]), dilation: (conv.dilation[0], conv.dilation[1]), groups: conv.groups, bias: true, device: conv.weight.device, dtype: conv.weight.dtype);
						Tensor w_conv = conv.weight.view(conv.out_channels, -1);
						Tensor w_bn = diag(bn.weight.div(sqrt(bn.eps + bn.running_var)));
						fusedconv.weight.copy_(mm(w_bn, w_conv).view(fusedconv.weight.shape));

						// Prepare spatial bias
						Tensor b_conv = conv.bias is null ? zeros(conv.weight.shape[0], dtype: conv.weight.dtype, device: conv.weight.device) : conv.bias;
						Tensor b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(sqrt(bn.running_var + bn.eps));
						fusedconv.bias.copy_(mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn);

						return act.forward(fusedconv.forward(x)).MoveToOuterDisposeScope();
					}

				}
			}
		}

		internal class DWConv : Conv
		{
			internal DWConv(int in_channels, int out_channels, int kernel_size = 1, int stride = 1, int d = 1, Func<Module<Tensor, Tensor>>? act = null, bool bias = false, Device? device = null, torch.ScalarType? dtype = null) : base(in_channels, out_channels, kernel_size, stride, groups: (int)BigInteger.GreatestCommonDivisor(in_channels, out_channels), d: d, bias: bias, act: act, device: device, dtype: dtype)
			{

			}
		}

		internal class Bottleneck : Module<Tensor, Tensor>
		{
			private readonly Conv cv1;
			private readonly Conv cv2;
			private readonly bool add;

			internal Bottleneck(int inChannels, int outChannels, (int, int) kernal, bool shortcut = true, int groups = 1, float e = 0.5f, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(Bottleneck))
			{
				int c = (int)(outChannels * e);
				cv1 = new Conv(inChannels, c, kernal.Item1, device: device, dtype: dtype);
				cv2 = new Conv(c, outChannels, kernal.Item2, groups: groups, device: device, dtype: dtype);
				add = shortcut && inChannels == outChannels;
				RegisterComponents();
			}

			public override Tensor forward(Tensor input)
			{
				return add ? cv2.forward(cv1.forward(input)) + input : cv2.forward(cv1.forward(input));
			}
		}

		internal class C3 : Module<Tensor, Tensor>
		{
			internal readonly Conv cv1;
			internal readonly Conv cv2;
			internal readonly Conv cv3;
			protected Sequential m;

			internal C3(int inChannels, int outChannels, int n = 1, bool shortcut = true, int groups = 1, float e = 0.5f, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(C3))
			{
				int c = (int)(outChannels * e);
				cv1 = new Conv(inChannels, c, 1, 1, device: device, dtype: dtype);
				cv2 = new Conv(inChannels, c, 1, 1, device: device, dtype: dtype);
				cv3 = new Conv(2 * c, outChannels, 1, device: device, dtype: dtype);

				m = Sequential(Enumerable.Range(0, n).Select(i => new Bottleneck(c, c, (1, 3), shortcut, groups, e: 1.0f, device: device, dtype: dtype)).ToArray());
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
				m = Sequential(Enumerable.Range(0, n).Select(_ => new Bottleneck(c, c, (3, 3), shortcut, groups, e: 1.0f, device: device, dtype: dtype)).ToArray());
			}
		}

		internal class C2f : Module<Tensor, Tensor>
		{
			internal readonly Conv cv1;
			internal readonly Conv cv2;
			internal readonly int c;
			internal Sequential m;
			internal C2f(int inChannels, int outChannels, int n = 1, bool shortcut = false, int groups = 1, float e = 0.5f, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(C2f))
			{
				c = (int)(outChannels * e);
				cv1 = new Conv(inChannels, 2 * c, 1, 1, device: device, dtype: dtype);
				cv2 = new Conv((2 + n) * c, outChannels, 1, device: device, dtype: dtype);  // optional act=FReLU(outChannels)
				m = Sequential();
				for (int i = 0; i < n; i++)
				{
					m = m.append(new Bottleneck(c, c, (3, 3), shortcut, groups, 1, device: device, dtype: dtype));
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
			internal readonly Conv cv1;
			internal readonly Conv cv2;
			internal readonly ModuleList<Module> m;
			internal readonly int c;
			internal C3k2(int inChannels, int outChannels, int n = 1, bool c3k = false, float e = 0.5f, int groups = 1, bool shortcut = true, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(C3k2))
			{
				c = (int)(outChannels * e);
				cv1 = new Conv(inChannels, 2 * c, 1, 1, device: device, dtype: dtype);
				cv2 = new Conv((2 + n) * c, outChannels, 1, device: device, dtype: dtype);  // optional act=FReLU(outChannels)
				m = new ModuleList<Module>();
				for (int i = 0; i < n; i++)
				{
					if (c3k)
					{
						m.append(new C3k(c, c, 2, shortcut, groups, device: device, dtype: dtype));
					}
					else
					{
						m.append(new Bottleneck(c, c, (3, 3), shortcut, groups, device: device, dtype: dtype));
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

		internal class SPPF : Module<Tensor, Tensor>
		{
			private readonly Conv cv1;
			private readonly Conv cv2;
			private readonly MaxPool2d m;

			internal SPPF(int inChannels, int outChannels, int kernalSize = 5, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(SPPF))
			{
				int c = inChannels / 2;
				cv1 = new Conv(inChannels, c, 1, 1, device: device, dtype: dtype);
				cv2 = new Conv(c * 4, outChannels, 1, 1, device: device, dtype: dtype);
				m = MaxPool2d(kernalSize, stride: 1, padding: kernalSize / 2);
				RegisterComponents();
			}

			public override Tensor forward(Tensor input)
			{
				using var _ = NewDisposeScope();

				Tensor x = cv1.forward(input);
				Tensor y1 = m.forward(x);
				Tensor y2 = m.forward(y1);
				Tensor result = cv2.forward(cat(new[] { x, y1, y2, m.forward(y2) }, 1));
				return result.MoveToOuterDisposeScope();

			}
		}

		internal class C2PSA : Module<Tensor, Tensor>
		{
			private readonly int c;
			private readonly Conv cv1;
			private readonly Conv cv2;
			private readonly Sequential m;

			internal C2PSA(int inChannel, int outChannel, int n = 1, float e = 0.5f, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(C2PSA))
			{
				if (inChannel != outChannel)
				{
					throw new ArgumentException("in channel not equals to out channel");
				}
				c = (int)(inChannel * e);
				cv1 = new Conv(inChannel, 2 * c, 1, 1, device: device, dtype: dtype);
				cv2 = new Conv(2 * c, outChannel, 1, device: device, dtype: dtype);
				m = Sequential();

				for (int i = 0; i < n; i++)
				{
					m = m.append(new PSABlock(c, attn_ratio: 0.5f, num_heads: c / 64, device: device, dtype: dtype));
				}
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
				ffn = Sequential(new Conv(c, c * 2, 1, device: device, dtype: dtype), new Conv(c * 2, c, 1, device: device, dtype: dtype));
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

			private readonly Conv qkv;
			private readonly Conv proj;
			private readonly Conv pe;

			private AttentionType attentionType;

			internal Attention(int dim, int num_heads = 8, float attn_ratio = 0.5f, AttentionType attentionType = AttentionType.SelfAttention, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(Attention))
			{
				this.num_heads = num_heads;
				head_dim = dim / num_heads;
				key_dim = (int)(head_dim * attn_ratio);
				scale = (float)Math.Pow(key_dim, -0.5);

				int nh_kd = key_dim * num_heads;
				int h = dim + nh_kd * 2;

				qkv = new Conv(dim, h, 1, device: device, dtype: dtype);
				proj = new Conv(dim, dim, 1, device: device, dtype: dtype);
				pe = new Conv(dim, dim, 3, 1, groups: dim, device: device, dtype: dtype);

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
			private readonly Conv cv1;
			private readonly Conv cv2;
			internal SCDown(int inChannel, int outChannel, int k, int s, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(SCDown))
			{
				cv1 = new Conv(inChannel, outChannel, 1, 1, device: device, dtype: dtype);
				cv2 = new Conv(outChannel, outChannel, kernel_size: k, stride: s, groups: outChannel, device: device, dtype: dtype);
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				return cv2.forward(cv1.forward(x));
			}
		}

		internal class C2fCIB : Module<Tensor, Tensor>
		{
			private readonly Conv cv1;
			private readonly Conv cv2;
			internal readonly Sequential m;
			internal C2fCIB(int inChannels, int outChannels, int n = 1, bool shortcut = false, bool lk = false, int g = 1, float e = 0.5f, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(C2fCIB))
			{
				int c = (int)(outChannels * e);
				cv1 = new Conv(inChannels, 2 * c, 1, 1, device: device, dtype: dtype);
				cv2 = new Conv((2 + n) * c, outChannels, 1, device: device, dtype: dtype);  // optional act=FReLU(outChannels)
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
					new Conv(inChannels, inChannels, 3, groups: inChannels, device: device, dtype: dtype),
					new Conv(inChannels, 2 * c, 1, device: device, dtype: dtype),
					lk ? new RepVGGDW(2 * c, device: device, dtype: dtype) : new Conv(2 * c, 2 * c, 3, groups: 2 * c, device: device, dtype: dtype),
					new Conv(2 * c, outChannels, 1, device: device, dtype: dtype),
					new Conv(outChannels, outChannels, 3, groups: outChannels, device: device, dtype: dtype));
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
			private readonly Conv cv1;

			/// <summary>
			/// Final 1x1 convolution layer that processes concatenated features.
			/// </summary>
			private readonly Conv cv2;

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
				cv1 = new Conv(c1, c_, 1, 1, device: device, dtype: dtype);
				cv2 = new Conv((1 + n) * c_, c2, 1, device: device, dtype: dtype);

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
				mlp = Sequential(new Conv(dim, mlp_hidden_dim, 1, device: device, dtype: dtype), new Conv(mlp_hidden_dim, dim, 1, device: device, dtype: dtype));
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

			private readonly Conv qkv;
			private readonly Conv proj;
			private readonly Conv pe;
			private readonly AttentionType attentionType;

			internal AAttn(int dim, int num_heads, int area = 1, AttentionType attentionType = AttentionType.SelfAttention, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(AAttn))
			{
				this.attentionType = attentionType;
				this.area = area;
				this.num_heads = num_heads;
				head_dim = dim / num_heads;
				int all_head_dim = head_dim * this.num_heads;

				qkv = new Conv(dim, all_head_dim * 3, 1, device: device, dtype: dtype);
				proj = new Conv(all_head_dim, dim, 1, device: device, dtype: dtype);
				pe = new Conv(all_head_dim, dim, 7, 1, 3, groups: dim, bias: true, device: device, dtype: dtype);
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
					// 调整维度为 (B, num_heads, seq_len, head_dim)
					q = q.permute(0, 1, 3, 2); // [B, nh, N, hd]
					k = k.permute(0, 1, 3, 2);
					v = v.permute(0, 1, 3, 2);

					// 使用内置的scaled_dot_product_attention
					x = functional.scaled_dot_product_attention(q, k, v);

					// 调整输出维度与原始实现一致
					x = x.permute(0, 2, 1, 3)  // [B, N, nh, hd]
						.reshape(B, N, -1);     // [B, N, C]

					// 处理v的维度与原始实现一致
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
			private readonly Conv conv;
			private readonly Conv conv1;
			private readonly int dim;
			private readonly Module<Tensor, Tensor> act;
			internal RepVGGDW(int ed, Device? device = null, Func<Module<Tensor, Tensor>>? act = null, torch.ScalarType? dtype = null) : base(nameof(RepVGGDW))
			{
				conv = new Conv(ed, ed, 7, 1, 3, groups: ed, act: act, device: device, dtype: dtype);
				conv1 = new Conv(ed, ed, 3, 1, 1, groups: ed, act: act, device: device, dtype: dtype);
				dim = ed;
				this.act = (act ?? SiLU)();

				RegisterComponents();
			}
			public override Tensor forward(Tensor x)
			{
				return act.forward(conv.forward(x) + conv1.forward(x));
			}
		}

		internal class DFL : Module<Tensor, Tensor>
		{
			private readonly Conv2d conv;
			private readonly int c1;
			internal DFL(int c1 = 16, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(DFL))
			{
				conv = Conv2d(c1, 1, 1, bias: false, device: device, dtype: dtype);
				Tensor x = arange(c1, device: device, dtype: dtype);
				conv.weight = Parameter(x.view(1, c1, 1, 1));
				this.c1 = c1;

				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				long b = x.shape[0];  // batch, channels, anchors
				long a = x.shape[2];
				return conv.forward(x.view(b, 4, c1, a).transpose(2, 1).softmax(1)).view(b, 4, a);

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

		internal class Yolov5Detect : Module<Tensor[], Tensor[]>
		{
			bool dynamic = false;  // force grid reconstruction
			bool export = false;// export mode

			private readonly int nc;
			private readonly int no;
			private readonly int nl;
			private readonly int na;
			//private List<Tensor> grid; // 存储网格坐标的列表

			//private readonly Tensor anchors;
			private readonly Sequential m;
			private float[][] anchors;
			private readonly int[] ch;

			private Device device;
			private torch.ScalarType scalarType;

			internal Yolov5Detect(int nc, int[] ch, float[][] anchors, bool inplace = true, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(Yolov5Detect))
			{
				this.nc = nc;
				no = nc + 5;// =85 每个类别需添加位置与置信度
				nl = anchors.Length;
				na = anchors[0].Length / 2; // =3 获得每个grid的anchor数量
				this.anchors = anchors;
				this.ch = ch;

				m = Sequential(ch.Select(x => Conv2d(x, no * na, 1, device: device, dtype: dtype)).ToArray());

				RegisterComponents();
			}

			public override Tensor[] forward(Tensor[] x)
			{
				using var _ = NewDisposeScope();

				device = x[0].device;
				scalarType = x[0].dtype;

				List<Tensor> z = new List<Tensor>();
				Tensor stride = tensor(new int[] { 8, 16, 32 }, dtype: scalarType, device: device);  //strides computed during build
				for (int i = 0; i < nl; i++)
				{
					x[i] = ((Module<Tensor, Tensor>)m[i]).forward(x[i]);
					long bs = x[i].shape[0];
					int ny = (int)x[i].shape[2];
					int nx = (int)x[i].shape[3];
					x[i] = x[i].view(bs, na, no, ny, nx).permute(0, 1, 3, 4, 2).contiguous();
					if (!training)
					{
						var (grid, anchor_grid) = _make_grid(nx, ny, i);
						Tensor[] re = x[i].sigmoid().split(new long[] { 2, 2, nc + 1 }, 4);
						Tensor xy = re[0];
						Tensor wh = re[1];
						Tensor conf = re[2];

						xy = (xy * 2 + grid) * stride[i];  // xy
						wh = (wh * 2).pow(2) * anchor_grid;  // wh
						Tensor y = cat(new Tensor[] { xy, wh, conf }, 4);
						z.Add(y.view(bs, na * nx * ny, no));
					}
				}

				if (training)
				{
					return x.Select(tensor => tensor.MoveToOuterDisposeScope()).ToArray();
				}
				else
				{
					List<Tensor> list = new List<Tensor>() { cat(z, 1) };
					return list.Select(tensor => tensor.MoveToOuterDisposeScope()).ToArray();
				}

			}

			private (Tensor, Tensor) _make_grid(int nx = 20, int ny = 20, int i = 0)
			{
				using (no_grad())
				{
					float[] an = new float[this.anchors.Length * this.anchors[0].Length];
					for (int ii = 0; ii < this.anchors.Length; ii++)
					{
						for (int j = 0; j < this.anchors[1].Length; j++)
						{
							an[ii * this.anchors[0].Length + j] = this.anchors[ii][j];
						}
					}
					Tensor anchors = tensor(an, new long[] { this.anchors.Length, this.anchors[0].Length / 2, 2 }, dtype: scalarType, device: device);
					Tensor stride = tensor(new int[] { 8, 16, 32 }, dtype: scalarType, device: device);  //strides computed during build
																										 //Tensor stride = tensor(ch, dtype: scalarType, device: device) / 8;  //strides computed during build
					var d = anchors[i].device;
					var t = anchors[i].dtype;

					long[] shape = new long[] { 1, na, ny, nx, 2 }; // grid shape
					Tensor y = arange(ny, t, d);
					Tensor x = arange(nx, t, d);
					Tensor[] xy = meshgrid(new Tensor[] { y, x }, indexing: "ij");
					Tensor yv = xy[0];
					Tensor xv = xy[1];
					Tensor grid = stack(new Tensor[] { xv, yv }, 2).expand(shape) - 0.5f;  // add grid offset, i.e. y = 2.0 * x - 0.5

					Tensor anchor_grid = (anchors[i] * stride[i]).view(new long[] { 1, na, 1, 1, 2 }).expand(shape);

					return (grid, anchor_grid);
				}

			}
		}

		internal class Yolov8Detect : Module<Tensor[], Tensor[]>
		{
			private int max_det = 300; // max_det
			private long[] shape = null;
			protected Tensor anchors = empty(0); // init
			protected Tensor strides = empty(0); // init

			private readonly int nc;
			protected readonly int nl;
			private readonly int reg_max;
			private readonly int no;
			private readonly int[] stride;
			private readonly ModuleList<Sequential> cv2 = new ModuleList<Sequential>();
			private readonly ModuleList<Sequential> cv3 = new ModuleList<Sequential>();
			private readonly Module<Tensor, Tensor> dfl;

			internal Yolov8Detect(int nc, int[] ch, bool legacy = true, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(Yolov8Detect))
			{
				this.nc = nc; // number of classes
				nl = ch.Length;// number of detection layers
				reg_max = 16; // DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
				no = nc + reg_max * 4; // number of outputs per anchor
				stride = new int[] { 8, 16, 32 }; // strides computed during build

				int c2 = Math.Max(Math.Max(16, ch[0] / 4), reg_max * 4);
				int c3 = Math.Max(ch[0], Math.Min(this.nc, 100));// channels


				cv2 = new ModuleList<Sequential>(ch.Select(x =>
					Sequential(
						new Conv(x, c2, 3, device: device, dtype: dtype),
						new Conv(c2, c2, 3, device: device, dtype: dtype),
						Conv2d(c2, 4 * reg_max, 1, device: device, dtype: dtype)
					)).ToArray());

				if (legacy)
				{
					cv3 = new ModuleList<Sequential>(ch.Select(x =>
						Sequential(
							new Conv(x, c3, 3, device: device, dtype: dtype),
							new Conv(c3, c3, 3, device: device, dtype: dtype),
							Conv2d(c3, this.nc, 1, device: device, dtype: dtype)
						)).ToArray());
				}
				else
				{
					cv3 = new ModuleList<Sequential>(ch.Select(x =>
						Sequential(
								Sequential(new DWConv(x, x, 3, device: device, dtype: dtype), new Conv(x, c3, 1, device: device, dtype: dtype)),
								Sequential(new DWConv(c3, c3, 3, device: device, dtype: dtype), new Conv(c3, c3, 1, device: device, dtype: dtype)),
								Conv2d(c3, this.nc, 1, device: device, dtype: dtype)
						)).ToArray());
				}

				dfl = reg_max > 1 ? new DFL(reg_max, device: device, dtype: dtype) : nn.Identity();
				// RegisterComponents();
			}

			public override Tensor[] forward(Tensor[] x)
			{
				using var _ = NewDisposeScope();

				for (int i = 0; i < nl; i++)
				{
					x[i] = cat(new Tensor[] { cv2[i].forward(x[i]), cv3[i].forward(x[i]) }, 1).MoveToOuterDisposeScope();
				}

				if (training)
				{
					return x;
				}
				else
				{
					Tensor y = _inference(x);
					Tensor[] results = new Tensor[] { y }.Concat(x).ToArray();
					results = results.Select(tensor => tensor.MoveToOuterDisposeScope()).ToArray();

					return results;
				}
			}

			//Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.
			private Tensor _inference(Tensor[] x)
			{
				long[] shape = x[0].shape;  // BCHW

				//List<Tensor> xi_mix = new List<Tensor>();
				//foreach (var xi in x)
				//{
				//	xi_mix.Add(xi.view(shape[0], this.no, -1));
				//}
				Tensor x_cat = cat(x.Select(xi => xi.view(shape[0], no, -1)).ToArray(), 2);

				if (this.shape != shape)
				{
					(Tensor anchors, Tensor strides) = make_anchors(x, stride, 0.5f);
					this.anchors = anchors.transpose(0, 1);
					this.strides = strides.transpose(0, 1);
					this.shape = shape;
				}

				Tensor[] box_cls = x_cat.split(new long[] { reg_max * 4, nc }, 1);
				Tensor box = box_cls[0];
				Tensor cls = box_cls[1];
				Tensor dbox = decode_bboxes(dfl.forward(box), anchors.unsqueeze(0)) * strides;
				return cat(new Tensor[] { dbox, cls.sigmoid() }, 1);

			}

			// Decode bounding boxes.
			protected virtual Tensor decode_bboxes(Tensor bboxes, Tensor anchors)
			{
				return dist2bbox(bboxes, anchors, xywh: true, dim: 1);
			}

			// Transform distance(ltrb) to box(xywh or xyxy).
			private Tensor dist2bbox(Tensor distance, Tensor anchor_points, bool xywh = true, int dim = -1)
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
				return cat(new Tensor[] { x1y1, x2y2 }, dim); // xyxy bbox

			}

			private (Tensor, Tensor) make_anchors(Tensor[] feats, int[] strides, float grid_cell_offset = 0.5f)
			{
				torch.ScalarType dtype = feats[0].dtype;
				Device device = feats[0].device;
				List<Tensor> anchor_points = new List<Tensor>();
				List<Tensor> stride_tensor = new List<Tensor>();
				for (int i = 0; i < strides.Length; i++)
				{
					long h = feats[i].shape[2];
					long w = feats[i].shape[3];
					Tensor sx = arange(w, device: device, dtype: dtype) + grid_cell_offset;  // shift x
					Tensor sy = arange(h, device: device, dtype: dtype) + grid_cell_offset;  // shift y
					Tensor[] sy_sx = meshgrid(new Tensor[] { sy, sx }, indexing: "ij");
					sy = sy_sx[0];
					sx = sy_sx[1];
					anchor_points.Add(stack(new Tensor[] { sx, sy }, -1).view(-1, 2));
					stride_tensor.Add(full(new long[] { h * w, 1 }, strides[i], dtype: dtype, device: device));
				}
				return (cat(anchor_points), cat(stride_tensor));

			}

		}

		internal class Proto : Module<Tensor, Tensor>
		{
			private readonly Conv cv1;
			private readonly Conv cv2;
			private readonly Conv cv3;
			private readonly ConvTranspose2d upsample;
			internal Proto(int c1, int c_ = 256, int c2 = 32, Device? device = null, torch.ScalarType? dtype = null) : base(nameof(Proto))
			{
				cv1 = new Conv(c1, c_, kernel_size: 3, device: device, dtype: dtype);
				upsample = ConvTranspose2d(c_, c_, 2, 2, 0, bias: true, device: device, dtype: dtype);  // nn.Upsample(scale_factor=2, mode='nearest')
				cv2 = new Conv(c_, c_, kernel_size: 3, device: device, dtype: dtype);
				cv3 = new Conv(c_, c2, kernel_size: 1, device: device, dtype: dtype);
				RegisterComponents();
			}

			public override Tensor forward(Tensor x)
			{
				return cv3.forward(cv2.forward(upsample.forward(cv1.forward(x))));
			}
		}

		public class Segment : Yolov8Detect
		{
			private readonly int nm;
			private readonly int npr;
			private readonly Proto proto;
			private readonly int c4;
			private readonly ModuleList<Sequential> cv4 = new ModuleList<Sequential>();

			public Segment(int[] ch, int nc = 80, int nm = 32, int npr = 256, bool legacy = true, Device? device = null, torch.ScalarType? dtype = null) : base(nc, ch, legacy, device, dtype)
			{
				this.nm = nm; // number of masks
				this.npr = npr;  // number of protos
				proto = new Proto(ch[0], this.npr, this.nm, device: device, dtype: dtype);  // protos
				c4 = Math.Max(ch[0] / 4, this.nm);

				cv4 = new ModuleList<Sequential>(ch.Select(x =>
					Sequential(
						new Conv(x, c4, 3, device: device, dtype: dtype),
						new Conv(c4, c4, 3, device: device, dtype: dtype),
						Conv2d(c4, this.nm, 1, device: device, dtype: dtype)
					)).ToArray());

				//foreach (int x in ch)
				//{
				//	cv4.append(Sequential(new Conv(x, c4, 3, device: device, dtype: dtype), new Conv(c4, c4, 3, device: device, dtype: dtype), nn.Conv2d(c4, this.nm, 1, device: device, dtype: dtype)));
				//}
				RegisterComponents();
			}

			public override Tensor[] forward(Tensor[] x)
			{
				using var _ = NewDisposeScope();

				Tensor p = proto.forward(x[0]); // mask protos
				long bs = p.shape[0]; //batch size

				var mc = cat(cv4.Select((module, i) => module.forward(x[i]).view(bs, nm, -1)).ToArray(), dim: 2); // mask coefficients				x = base.forward(x);
				x = base.forward(x);
				if (training)
				{
					x = x.Append(mc).Append(p).ToArray();

					return x.Select(tensor => tensor.MoveToOuterDisposeScope()).ToArray();
				}
				else
				{
					return new Tensor[] { cat(new Tensor[] { x[0], mc }, dim: 1).MoveToOuterDisposeScope(), x[1].MoveToOuterDisposeScope(), x[2].MoveToOuterDisposeScope(), x[3].MoveToOuterDisposeScope(), p.MoveToOuterDisposeScope() };
				}
			}
		}

		/// <summary>
		/// YOLO OBB detection head for detection with rotation models.
		/// </summary>
		public class OBB : Yolov8Detect
		{
			private readonly int ne;
			private readonly ModuleList<Sequential> cv4 = new ModuleList<Sequential>();
			private Tensor angle;

			public OBB(int[] ch, int nc = 80, int ne = 1, bool legacy = true, Device? device = null, torch.ScalarType? dtype = null) : base(nc, ch, legacy: legacy, device: device, dtype: dtype)
			{
				this.ne = ne;  // number of extra parameters
				int c4 = Math.Max(ch[0] / 4, this.ne);

				cv4 = new ModuleList<Sequential>(ch.Select(x =>
					Sequential(
						new Conv(x, c4, 3, device: device, dtype: dtype),
						new Conv(c4, c4, 3, device: device, dtype: dtype),
						Conv2d(c4, this.ne, 1, device: device, dtype: dtype)
					)).ToArray());

				//foreach (int x in ch)
				//{
				//	cv4.append(nn.Sequential(new Conv(x, c4, 3, device: device, dtype: dtype), new Conv(c4, c4, 3, device: device, dtype: dtype), nn.Conv2d(c4, this.ne, 1, device: device, dtype: dtype)));
				//}
				RegisterComponents();
			}

			public override Tensor[] forward(Tensor[] x)
			{
				long bs = x[0].shape[0];  // batch size

				// NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
				Tensor angle = cat(cv4.Zip(x, (layer, input) => layer.forward(input).view(bs, ne, -1)).ToArray(), 2);  // OBB theta logits	
				angle = (angle.sigmoid() - 0.25f) * MathF.PI;  // [-pi/4, 3pi/4]
															   // angle = angle.sigmoid() * math.pi / 2  // [0, pi/2]

				if (!training)
				{
					this.angle = angle;
				}

				x = base.forward(x);

				if (training)
				{
					return x.Concat(new Tensor[] { angle }).ToArray();
				}
				return new Tensor[] { cat(new Tensor[] { x[0], angle }, 1), x[1], x[2], x[3], angle };

			}

			//Decode rotated bounding boxes.
			protected override Tensor decode_bboxes(Tensor bboxes, Tensor anchors)
			{
				return YoloSharp.Utils.Tal.dist2rbox(bboxes, angle, anchors, dim: 1);
			}
		}

		public class Pose : Yolov8Detect
		{
			private readonly int[] kpt_shape;
			private readonly int nk;
			private readonly ModuleList<Sequential> cv4 = new ModuleList<Sequential>();

			/// <summary>
			/// YOLO Pose head for keypoints models.
			/// </summary>
			/// <remarks>This class extends the Detect head to include keypoint prediction capabilities for pose estimation tasks.</remarks>
			/// <param name="nc">Number of classes.</param>
			/// <param name="kpt_shape">Number of keypoints, number of dims (2 for x,y or 3 for x,y,visible).</param>
			/// <param name="ch">Tuple of channel sizes from backbone feature maps.</param>
			/// <param name="legacy"></param>
			/// <param name="device"></param>
			/// <param name="dtype"></param>
			public Pose(int nc = 80, int[] kpt_shape = null, int[] ch = null, bool legacy = true, Device? device = null, torch.ScalarType? dtype = null) : base(nc: nc, ch: ch, legacy: legacy, device: device, dtype: dtype)
			{
				this.kpt_shape = kpt_shape ?? new int[] { 17, 3 }; // number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
				nk = kpt_shape[0] * kpt_shape[1];  // number of keypoints total
				int c4 = Math.Max(ch[0] / 4, nk);

				cv4 = new ModuleList<Sequential>(ch.Select(x =>
					Sequential(
						new Conv(x, c4, 3, device: device, dtype: dtype),
						new Conv(c4, c4, 3, device: device, dtype: dtype),
						Conv2d(c4, nk, 1, device: device, dtype: dtype)
					)).ToArray());

				RegisterComponents();
			}

			public override Tensor[] forward(Tensor[] x)
			{
				long bs = x[0].shape[0];  // batch size
				Tensor kpt = cat(cv4.Select((module, i) => module.forward(x[i]).view(bs, nk, -1)).ToArray(), dim: -1);  // (bs, 17*3, h*w)
				x = base.forward(x);

				if (training)
				{
					return x.Append(kpt).ToArray();
				}

				Tensor pred_kpt = kpts_decode(bs, kpt);
				return new Tensor[] { cat(new Tensor[] { x[0], pred_kpt }, 1), x[1], kpt };
			}

			private Tensor kpts_decode(long bs, Tensor kpts)
			{
				Tensor y = kpts.clone();
				int ndim = kpt_shape[1];
				if (ndim == 3)
				{
					y[.., TensorIndex.Slice(2, step: ndim)] = y[.., TensorIndex.Slice(2, step: ndim)].sigmoid();  // sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
				}
				y[.., TensorIndex.Slice(0, step: ndim)] = (y[.., TensorIndex.Slice(0, step: ndim)] * 2.0 + (anchors[0] - 0.5)) * strides;
				y[.., TensorIndex.Slice(1, step: ndim)] = (y[.., TensorIndex.Slice(1, step: ndim)] * 2.0 + (anchors[1] - 0.5)) * strides;

				return y;
			}

		}

		public class Classify : Module<Tensor[], Tensor[]>
		{
			private readonly Conv conv;
			private readonly AdaptiveAvgPool2d pool;
			private readonly Linear linear;
			private readonly Dropout drop;

			/// <summary>
			/// YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2).
			/// </summary>
			/// <remarks>
			/// This class implements a classification head that transforms feature maps into class predictions.
			///</remarks>
			/// <param name="c1">Number of input channels.</param>
			/// <param name="c2">Number of output classes.</param>
			/// <param name="k">Kernel size.</param>
			/// <param name="s">Stride.</param>
			/// <param name="p">Padding.</param>
			/// <param name="g">Groups.</param>
			public Classify(int c1, int c2, int k = 1, int s = 1, int? p = null, int g = 1) : base(nameof(Classify))
			{
				int c_ = 1280;  // efficientnet_b0 size
				conv = new Conv(c1, c_, k, s, p, g);
				pool = AdaptiveAvgPool2d(1);  // to x(b,c_,1,1)
				drop = Dropout(p: 0.0, inplace: true);
				linear = Linear(c_, c2);  // to x(b,c2)
				RegisterComponents();
			}

			public override Tensor[] forward(Tensor[] inputs)
			{
				Tensor x = cat(inputs, 1);
				x = linear.forward(drop.forward(pool.forward(conv.forward(x)).flatten(1)));
				if (training)
				{
					return new Tensor[] { x };
				}
				else
				{
					Tensor y = x.softmax(1);  // get final output
					return new Tensor[] { y, x };
				}
			}

		}

	}
}
