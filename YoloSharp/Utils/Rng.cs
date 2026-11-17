using System;

namespace YoloSharp.Utils
{
    /// <summary>
    /// [Add 2026-11-09] Shared deterministic RNG for training reproducibility.
    /// ultralytics fixes seed=0 (init_seeds) so reruns are identical; the platform had no
    /// seeding — every training run drew fresh randomness (dataloader shuffle, augmentation,
    /// mosaic sampling), making results irreproducible and adding noise to python-vs-platform
    /// comparisons. Train() resets this once per run; all stochastic components share it.
    /// 训练可复现共享随机源：对齐ultralytics seed=0语义——Train()开头Reset(0)，
    /// dataloader洗牌/增强/mosaic采样共用同一个种子化实例。
    /// </summary>
    public static class Rng
    {
        private static readonly object _lock = new object();
        private static Random _shared = new Random();

        /// <summary>Reset to a fixed seed (called at Train() start).</summary>
        public static void Reset(int seed)
        {
            lock (_lock) _shared = new Random(seed);
        }

        /// <summary>Next double in [0,1) — thread-safe.</summary>
        public static double NextDouble()
        {
            lock (_lock) return _shared.NextDouble();
        }

        /// <summary>Next int in [minValue, maxValue) — thread-safe.</summary>
        public static int Next(int minValue, int maxValue)
        {
            lock (_lock) return _shared.Next(minValue, maxValue);
        }

        /// <summary>Next non-negative int — thread-safe.</summary>
        public static int Next()
        {
            lock (_lock) return _shared.Next();
        }
    }
}
