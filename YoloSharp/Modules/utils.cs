using System.Collections.Generic;
using TorchSharp;

namespace YoloSharp.Modules
{
    /// <summary>
    /// Utility functions for neural network modules (mirrors ultralytics/nn/modules/utils.py).
    /// </summary>
    internal static class utils
    {
        /// <summary>
        /// Create a list of cloned modules from the given module.
        ///
        /// Python uses `copy.deepcopy(module)`; TorchSharp has no generic module deep-copy, so this port takes a
        /// factory that constructs a fresh instance per call (callers pass e.g. `() =&gt; new MSDeformAttn(...)`).
        ///
        /// Args:
        ///     module_factory (Func&lt;nn.Module&gt;): Factory creating a fresh module instance per clone.
        ///     n (int): Number of clones to create.
        ///
        /// Returns:
        ///     (List&lt;nn.Module&gt;): A list containing n clones of the input module.
        ///
        /// Examples:
        ///     >>> var clones = _get_clones(() =&gt; new MSDeformAttn(d_model, n_levels, n_heads, n_points), 3);
        ///     >>> clones.Count
        ///     3
        /// </summary>
        internal static List<torch.nn.Module> _get_clones(System.Func<torch.nn.Module> module_factory, int n)
        {
            var clones = new List<torch.nn.Module>(n);
            for (int i = 0; i < n; i++)
                clones.Add(module_factory());   // copy.deepcopy(module)
            return clones;
        }

        /// <summary>
        /// Initialize conv/fc bias value according to a given probability value.
        ///
        /// This function calculates the bias initialization value based on a prior probability using the inverse
        /// sigmoid (logit) function. It's commonly used in object detection models to initialize classification
        /// layers with a specific positive prediction probability.
        ///
        /// Args:
        ///     prior_prob (float, optional): Prior probability for bias initialization.
        ///
        /// Returns:
        ///     (float): Bias initialization value calculated from the prior probability.
        ///
        /// Examples:
        ///     >>> bias = bias_init_with_prob(0.01)
        ///     >>> print(f"Bias initialization value: {bias:.4f}")
        ///     Bias initialization value: -4.5951
        /// </summary>
        internal static double bias_init_with_prob(double prior_prob = 0.01)
        {
            return -System.Math.Log((1 - prior_prob) / prior_prob);   // return bias_init
        }

        /// <summary>
        /// Calculate the inverse sigmoid function for a tensor.
        ///
        /// This function applies the inverse of the sigmoid function to a tensor, which is useful in various neural
        /// network operations, particularly in attention mechanisms and coordinate transformations.
        ///
        /// Args:
        ///     x (torch.Tensor): Input tensor with values in range [0, 1]; values outside this range are clamped.
        ///     eps (float, optional): Small epsilon value to prevent numerical instability.
        ///
        /// Returns:
        ///     (torch.Tensor): Tensor after applying the inverse sigmoid function.
        ///
        /// Examples:
        ///     >>> x = torch.tensor([0.2, 0.5, 0.8])
        ///     inverse_sigmoid(x)
        ///     tensor([-1.3863,  0.0000,  1.3863])
        /// </summary>
        internal static torch.Tensor inverse_sigmoid(torch.Tensor x, double eps = 1e-5)
        {
            x = x.clamp(min: 0, max: 1);
            var x1 = x.clamp(min: eps);
            var x2 = (1 - x).clamp(min: eps);
            return torch.log(x1 / x2);
        }

        /// <summary>
        /// Implement multi-scale deformable attention in PyTorch.
        ///
        /// Folds the (num_levels, num_points) axes into a single num_total_points axis so every traced tensor stays
        /// at rank &lt;= 5, the maximum rank supported by CoreML's MIL converter. Numerically equivalent to the rank-6
        /// reference implementation on CUDA and CPU.
        ///
        /// Args:
        ///     value (torch.Tensor): Value tensor with shape (bs, num_keys, num_heads, embed_dims).
        ///     value_spatial_shapes ((int h, int w)[]): Per-level spatial shapes.
        ///     sampling_locations (torch.Tensor): Sampling locations with shape (bs, num_queries, num_heads,
        ///         num_levels * num_points, 2).
        ///     attention_weights (torch.Tensor): Attention weights with shape (bs, num_queries, num_heads,
        ///         num_levels * num_points).
        ///
        /// Returns:
        ///     (torch.Tensor): Output tensor with shape (bs, num_queries, num_heads * embed_dims).
        ///
        /// References:
        ///     https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py
        /// </summary>
        internal static torch.Tensor multi_scale_deformable_attn_pytorch(
            torch.Tensor value,
            (int h, int w)[] value_spatial_shapes,
            torch.Tensor sampling_locations,
            torch.Tensor attention_weights)
        {
            long bs = (long)value.shape[0];
            long num_heads = (long)value.shape[2];
            long embed_dims = (long)value.shape[3];
            long num_queries = (long)sampling_locations.shape[1];
            long num_total_points = (long)sampling_locations.shape[3];
            int num_levels = value_spatial_shapes.Length;
            long num_points = num_total_points / num_levels;

            // (bs, num_keys, num_heads, embed_dims) -> (bs*num_heads, embed_dims, num_keys)
            var valuePerm = value.permute(0, 2, 3, 1).flatten(0, 1);
            // Map to grid_sample coords in [-1, 1]: -> (bs*num_heads, num_queries, num_total_points, 2)
            var sampling_grids_all = (2 * sampling_locations - 1).permute(0, 2, 1, 3, 4).flatten(0, 1);

            var sampling_value_list = new List<torch.Tensor>(num_levels);
            long valueOffset = 0, pointOffset = 0;
            for (int level = 0; level < num_levels; level++)
            {
                int h = value_spatial_shapes[level].h, w = value_spatial_shapes[level].w;
                long hw = (long)h * w;
                var value_l = valuePerm
                    .narrow(-1, valueOffset, hw)                    // (bs*num_heads, embed_dims, H*W) of this level
                    .reshape(bs * num_heads, embed_dims, h, w);
                var grid_l = sampling_grids_all
                    .narrow(-2, pointOffset, num_points);           // (bs*num_heads, num_queries, num_points, 2)
                sampling_value_list.Add(
                    torch.nn.functional.grid_sample(value_l, grid_l,
                        mode: torch.GridSampleMode.Bilinear,
                        padding_mode: torch.GridSamplePaddingMode.Zeros, align_corners: false));
                valueOffset += hw;
                pointOffset += num_points;
            }

            // (bs, num_queries, num_heads, num_total_points) -> (bs*num_heads, 1, num_queries, num_total_points)
            var weights = attention_weights.permute(0, 2, 1, 3).reshape(bs * num_heads, 1, num_queries, num_total_points);

            var output = (torch.cat(sampling_value_list, dim: -1) * weights)
                .sum(-1)
                .view(bs, num_heads * embed_dims, num_queries);
            return output.transpose(1, 2).contiguous();
        }
    }
}
