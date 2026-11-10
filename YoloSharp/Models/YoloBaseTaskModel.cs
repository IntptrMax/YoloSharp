using Data;
using ScottPlot;
using System.Data;
using System.Diagnostics;
using System.Text;
using TorchSharp;
using TorchSharp.Modules;
using Utils;
using YoloSharp.Modules;
using YoloSharp.Types;
using YoloSharp.Utils;

namespace YoloSharp.Models
{
    internal abstract class YoloBaseTaskModel
    {
        private bool _needHeadInit = true;   // true until Bias_Init runs (ctor-init or LoadModel pre-init) / 在Bias_Init执行前为true

        protected Config config = new Config();
        protected Yolo.Yolov8 yolo;
        protected torch.nn.Module<Dictionary<string, object>, Dictionary<string, torch.Tensor>, (torch.Tensor loss, torch.Tensor loss_detach)> loss;
        protected ModelEMA ema;
        // [Add 2026-11-09] Warmup triplet state (shared Train→TrainEpoch): bias group reference
        // (correct-group identification for warmup_bias_lr) + full accumulate window (interp target).
        // warmup三件套状态：bias组引用(修正预热落点)+累积窗口目标(渐进插值终点)。
        private TorchSharp.Modules.SGD.ParamGroup _biasGroupRef;
        private int _accumulateFull;

        internal YoloBaseTaskModel()
        {
            torchvision.io.DefaultImager = new torchvision.io.SkiaImager();
        }

        internal virtual void LoadModel(string path, bool skipNcNotEqualLayers = false)
        {
            Console.WriteLine("Loading model...");

            Dictionary<string, torch.Tensor> sourceStateDict;

            if (path.EndsWith(".pt", StringComparison.OrdinalIgnoreCase))
            {
                var ptLoader = new YoloSharp.ModelLoader.PtReader.YoloPtLoader();
                var loadResult = ptLoader.LoadFromFile(path);
                sourceStateDict = YoloSharp.ModelLoader.PtReader.TensorBuilder.BuildStateDict(loadResult);
            }
            else
            {
                sourceStateDict = Lib.LoadModel(path, skipNcNotEqualLayers);
            }

            var modelShapes = new Dictionary<string, long[]>();
            foreach (var kv in yolo.state_dict())
            {
                modelShapes[kv.Key] = kv.Value.shape;
            }

            // [Fix 2026-11-05] REMOVED the head-layer remap entirely — python's intersect_dicts
            // transfers NOTHING when layer names differ (converted.pt head sits at model.22, the
            // training skeleton's at model.29), so python trains with a fully-random 6-group head
            // and the aux groups naturally stay silent (verified: python best.pt aux outputs are
            // exactly 0 on test images). The remap I added force-fed converted.pt head weights
            // into the aux groups — aux then produced competing high-score boxes with different
            // widths, splitting the middle object into "wide+narrow" duplicate detections
            // (observed 0.64/0.76 mid-object scores). No remap = python-identical behavior.
            // 彻底移除头层号remap：python对层名不匹配的头一条都不迁（头全随机）——aux组自然
            // 沉默（best.pt实测aux输出恒0）。此前remap把converted头权塞进aux组导致双头竞争
            // 出框、中间目标被拆成宽+窄双框。删remap=与python行为一致。

            var matchedDict = new Dictionary<string, torch.Tensor>();
            int skippedMissing = 0;
            int skippedShape = 0;
            var skippedShapeSamples = new List<string>();

            foreach (var kv in sourceStateDict)
            {
                string name = kv.Key;
                torch.Tensor srcTensor = kv.Value;

                if (!modelShapes.TryGetValue(name, out var expectedShape))
                {
                    // [Fix 2026-10-24] 非参数缓冲不计入"头不匹配"（needHeadInit）：
                    // stride/class_weights/angle（OBB头的角度维度标记）在C#骨架上无对应buffer，
                    // 但它们不是可训练权重——旧逻辑把它们算作missing，导致加载预训练后
                    // Bias_Init()重置整个Detect头（实测：v8n-obb.pt 397/400迁移，3个missing
                    // 全是非权重缓冲，却触发头重置→epoch1 mAP50从Python的0.94跌到0.41）
                    if (!name.EndsWith("class_weights") && !name.EndsWith("stride") && !name.EndsWith("angle"))
                    {
                        skippedMissing++;
                    }
                    continue;
                }

                if (!srcTensor.shape.SequenceEqual(expectedShape))
                {
                    skippedShape++;
                    if (skippedShapeSamples.Count < 10)
                    {
                        skippedShapeSamples.Add($"{name}: src [{string.Join(",", srcTensor.shape)}] " + $"vs model [{string.Join(",", expectedShape)}]");
                    }
                    continue;
                }

                matchedDict[name] = srcTensor;
            }

            Console.WriteLine($"Transferred {matchedDict.Count}/{sourceStateDict.Count} items");
            if (skippedShape > 0)
            {
                Console.WriteLine($"  Shape mismatch: {skippedShape} (will be re-initialized)");
                foreach (var s in skippedShapeSamples)
                {
                    Console.WriteLine($"    {s}");
                }
            }
            if (skippedMissing > 0)
            {
                Console.WriteLine($"  Not in model: {skippedMissing}");
            }
            if (matchedDict.Count > 0)
            {
                // [Fix 2026-11-04] Cross-device copy silently no-ops in TorchSharp load_state_dict:
                // .pt sources load as CPU tensors while the model sits on CUDA — weights never
                // landed (scores.max=0.016 on a correctly-transferred checkpoint). Move every
                // matched tensor to the model's device before loading.
                // 跨设备copy静默失效：pt源是CPU张量而模型在CUDA——权重从未真正落位。
                // 加载前统一迁到模型设备。
                var firstParam = yolo.parameters().Cast<torch.Tensor>().FirstOrDefault();
                if (firstParam is torch.Tensor fp)
                {
                    var dev = fp.device;
                    if (dev.type != TorchSharp.DeviceType.CPU)
                    {
                        var moved = new Dictionary<string, torch.Tensor>();
                        foreach (var mkv in matchedDict)
                            moved[mkv.Key] = mkv.Value.to(dev);
                        matchedDict = moved;
                    }
                }
                // [Fix 2026-11-04] Bias-init BEFORE load_state_dict (ultralytics order: model ctor
                // calls bias_init, then intersect_dict weights overwrite). Unmatched layers (nc
                // 80→1 cls tails) keep the good bias prior log(5/nc/640²)≈-12 → sigmoid≈0 →
                // epoch-1 cls loss stays ~5. The previous "skip bias_init on partial transfer"
                // left random biases (±0.5) → mass false positives → epoch-1 cls loss 1661.
                // 先bias_init再load：未匹配层(cls末层)留住好偏置先验；迁移层随后被覆盖。
                // 此前"部分迁移跳过bias_init"导致随机偏置→首轮cls loss爆1661。
                this.Bias_Init();
                _needHeadInit = false;   // already initialized here — Train() must not repeat / 此处已初始化，Train不再重复
                yolo.load_state_dict(matchedDict, strict: false);
            }
            // [Fix 2026-10-28] Move model to dtype+device uniformly after loading:
            // old code only called to(Dtype) without to(Device) — under CUDA config the model stayed on CPU while
            // ImagePredict sent input to CUDA → "cpu and cuda:0" mismatch (silent 0 detections).
            // The training path Train() has to(device) internally, so this never surfaced before.
            // 加载后统一迁移dtype+device——旧代码只to(Dtype)不to(Device)，CUDA下模型留CPU导致设备不匹配。
            yolo.to(config.Device);
            yolo.to(config.Dtype);

            // [Fix 2026-11-04] bias_init now runs inside LoadModel BEFORE load_state_dict
            // (ultralytics order) — no post-load conditional init. Train()'s init only serves
            // the from-scratch path (no LoadModel call).
            // bias_init已移到LoadModel内load之前(ultralytics时序)——此处不再条件重置。
            // Train()里的初始化只服务从头训路径。

            Console.WriteLine("Model loaded.");
        }

        internal virtual void Train()
        {
            // [Add 2026-11-09] Deterministic seeding (ultralytics init_seeds(0) semantics):
            // python fixes seed=0 + deterministic=True so reruns are reproducible; the platform
            // had NO seeding — every run was a random seed (two trainings of identical config
            // gave different results, and the ~0.008 score gap vs python was partly this noise).
            // Fix torch CPU/CUDA seeds + .NET Random for dataloader shuffling/augmentation.
            // 对齐python确定性种子：python侧seed=0+deterministic——平台此前无种子，每次
            // 随机。固定torch双端种子+.NET Random（dataloader洗牌/增强共用）。
            try
            {
                torch.manual_seed(0);
                if (torch.cuda.is_available()) torch.cuda.manual_seed_all(0);
                global::Data.Augment.ReSeedAll(0);   // 增强随机源（ThreadLocalRandom工厂重置为种子链）
                Console.WriteLine("  Seed: 0 (deterministic, aligned with ultralytics)");
            }
            catch { }

            // Fix: from-scratch training path（未走LoadModel）模型保持Float32，而数据按config.Dtype进入——
            // CUDA上触发 "Input type (CUDABFloat16Type) and weight type (CUDAFloatType) should be the same"。
            // LoadModel路径在 yolo.to(config.Dtype) 已转换；此处对从头训路径补齐同一转换。
            yolo.to(config.Dtype);

            if (config.End2End)
            {
                One2one_Init();
            }
            if (_needHeadInit)
            {
                Console.WriteLine("Initializing head bias...");
                Bias_Init();
            }

            float best_fitness = float.NegativeInfinity;

            Console.WriteLine(config.ToString());
            WriteConfig();

            BaseDataset trainDataSet = this.config.TaskType == TaskType.Classification ? new ClassificationDataset(config) : new YoloDataset(config);
            if (trainDataSet.Count == 0)
            {
                throw new FileNotFoundException("No data found in the path: " + config.RootPath);
            }

            YoloDataLoader trainDataLoader = new YoloDataLoader(trainDataSet, config.BatchSize, num_worker: config.Workers, shuffle: true, seed: 0, device: config.Device);   // [Fix 2026-11-09] 固定洗牌种子(可复现,对齐python seed=0)

            config.ValDataPath = string.IsNullOrEmpty(config.ValDataPath) ? config.TrainDataPath : config.ValDataPath;

            BaseDataset valDataSet = this.config.TaskType == TaskType.Classification ? new ClassificationDataset(this.config, true) : new YoloDataset(this.config, true);
            if (valDataSet.Count == 0)
            {
                throw new FileNotFoundException("No data found in the path: " + config.RootPath);
            }

            YoloDataLoader valDataLoader = new YoloDataLoader(valDataSet, config.BatchSize, num_worker: config.Workers, shuffle: false, device: config.Device);

            // [Fix 2026-11-05] Removed dfl freeze — ultralytics trains DFL weights too (no
            // "freezing .dfl." logic in official train.py). Freezing was a silent divergence.
            // 移除dfl冻结——官方训练DFL参与更新；冻结是与python的隐性差异。
            foreach (var (name, param) in yolo.named_parameters())
            {
                param.requires_grad = true;
            }

            // [Add 2026-11-05] VRAM guidance: on 8GB laptops, v9-s @ bs16 spills activations to
            // shared memory (WDDM paging) — batches jump ~4.8× (1.1s→5.3s) and epochs take 220s
            // vs python's 90s. bs8 fits in VRAM and runs FASTER end-to-end (54s/epoch < 90s).
            // 8GB卡显存提示：v9-s在bs16下激活溢出换页（批时长4.8倍）——提示改bs8。
            // Detected lazily: first-epoch batch timing is checked inside TrainEpoch after 8 batches.
            // 首轮前8批后按实际批时长自监测提示（见TrainEpoch内），此处无操作。

            double lr_fit = config.LearningRate;

            float weightDecay = 0.0005f;

            var allParams = yolo.named_parameters().Where(p => p.parameter.requires_grad).ToList();

            var bnWeightNames = allParams.Where(p => (p.name.Contains("bn.") || p.name.Contains("Norm") || p.name.Contains("norm")) && p.name.Contains("weight")).ToList();
            var biasNames = allParams.Where(p => p.name.Contains("bias")).ToList();

            var bnBiasNames = new HashSet<string>(bnWeightNames.Select(p => p.name));
            var biasNamesSet = new HashSet<string>(biasNames.Select(p => p.name));

            var weightNames = allParams.Where(p => !bnBiasNames.Contains(p.name) && !biasNamesSet.Contains(p.name)).ToList();

            var weightGroup = new TorchSharp.Modules.SGD.ParamGroup
            {
                Parameters = weightNames.Select(p => p.parameter),
                Options = new TorchSharp.Modules.SGD.Options() { weight_decay = weightDecay, LearningRate = lr_fit, momentum = 0.937, nesterov = true }
            };
            var bnWeightGroup = new TorchSharp.Modules.SGD.ParamGroup
            {
                Parameters = bnWeightNames.Select(p => p.parameter),
                Options = new TorchSharp.Modules.SGD.Options() { weight_decay = 0.0, LearningRate = lr_fit, momentum = 0.937, nesterov = true },
            };
            var biasGroup = new TorchSharp.Modules.SGD.ParamGroup
            {
                Parameters = biasNames.Select(p => p.parameter),
                Options = new TorchSharp.Modules.SGD.Options() { weight_decay = 0.0, LearningRate = lr_fit, momentum = 0.937, nesterov = true },
            };

            // [Fix 2026-11-05] SGD(momentum=0.937, nesterov) aligned with ultralytics exp68 — the
            // old AdamW(2e-4)+CosLR recipe under-trained vs python (confidence mean 0.836 vs 0.906
            // across the full test set). SGD 0.01 + linear decay + warmup_bias 0.1 matches python.
            // 优化器对齐python：SGD(0.01, momentum 0.937, nesterov)+线性衰减。
            torch.optim.Optimizer optimizer = new TorchSharp.Modules.SGD(
                new TorchSharp.Modules.SGD.ParamGroup[] { weightGroup, bnWeightGroup, biasGroup },
                lr: lr_fit, momentum: 0.937, dampening: 0.0, weight_decay: 0.0, nesterov: true, maximize: false);
            // [Add 2026-11-09] Warmup triplet (python trainer.py L527-541): keep the bias group's
            // REFERENCE so warmup_bias_lr lands on the right group — the old code guessed by
            // position (paraId==0 = weight group!) so bias warmup never actually applied.
            // 记录bias组引用——warmup_bias_lr要落到bias组（旧代码按位置猜paraId==0是weight组，
            // bias预热从未真正生效）。
            var biasGroupRef = biasGroup;
            _biasGroupRef = biasGroup;   // [Add 2026-11-09] 提升到实例字段供TrainEpoch的warmup用

            // nb_estimate已无需用于EMA tau（对齐官方固定2000）
            // [Fix 2026-10-24] EMA对齐ultralytics官方：decay=0.9999, tau=2000固定。
            // 旧值(0.999, nb*10)使EMA有效平均窗口≈10epoch——前20轮val指标严重滞后(与Python差异大的根因)
            // [Fix 2026-10-24] EMA对齐ultralytics官方：decay=0.9999, tau=2000固定。
            ema = new ModelEMA(yolo, decay: 0.9999f, tau: 2000f);

            Func<int, double> lrLambda = config.UseCosLR ?
                OneCycle(1.0, config.Lrf, config.Epochs) :
                LrLambda(1.0, config.Lrf, config.Epochs);

            torch.optim.lr_scheduler.LRScheduler lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lrLambda);
            Console.WriteLine();

            // [Fix 2026-11-05] grad clip 10.0 — python v9 train.py line 311 clips gradients
            // (max_norm=10); the platform never passed it (default 0=off). With SGD lr=0.01 the
            // occasional exploding batch crashed whole epochs (observed mAP50 collapse to 0.000
            // at ep18/ep27). 梯度裁剪对齐python(10.0)——此前从未启用，SGD大lr下偶发爆炸塌轮。
            // [Add 2026-11-05] Gradient accumulation to nbs=64 (python L138): accumulate =
            // round(64/batch). bs8→8批合一步（有效batch 64），消除SGD小batch梯度噪声导致的mAP锯齿。
            int accumulateSteps = Math.Max(1, (int)Math.Round(64.0 / Math.Max(1, config.BatchSize)));
            _accumulateFull = accumulateSteps;   // [Add 2026-11-09] 提升到实例字段供TrainEpoch的warmup用
            AMPWrapper amp = new AMPWrapper(yolo, optimizer, precision: config.Dtype, gradClipNorm: 10.0f, accumulate: accumulateSteps);
            EarlyStopping stopper = new EarlyStopping(config.Patience);
            string weightsPath = Path.Combine(config.OutputPath, "weights");

            long nb = trainDataLoader.Count;
            // long nw = Math.Max(config.WarmUpEpoches * nb, 100);
            long nw = config.WarmUpEpoches > 0 ? Math.Max((long)Math.Round((double)config.WarmUpEpoches * nb), nb) : -1;

            long valBatches = valDataLoader.Count;

            for (int epoch = 1; epoch <= config.Epochs; epoch++)
            {
                // [Fix 2026-11-02] Stop check at epoch boundary: honor Config.StopRequested so the
                // editor's "停止训练" actually stops (the callback existed but was never polled —
                // observed a run ignoring stop from epoch 91 to 118+). Break BEFORE starting a new
                // epoch: last.bin/best.bin from completed epochs are preserved as-is.
                // epoch边界检查停止：之前回调存在但从未被轮询——停止被无视。
                // 在开启新epoch前break，已完成轮的last.bin/best.bin原样保留。
                try
                {
                    if (config.StopRequested != null && config.StopRequested())
                    {
                        Console.WriteLine("[Txm] Training stopped by user request at epoch boundary " + (epoch - 1) + ".");
                        break;
                    }
                }
                catch { }

                if (!Directory.Exists(weightsPath))
                {
                    Directory.CreateDirectory(weightsPath);
                }

                Console.WriteLine(GetTrainDescription());
                Stopwatch stopwatch = Stopwatch.StartNew();

                bool closeMosaic = config.CloseMosaic > 0 && epoch > config.Epochs - config.CloseMosaic;
                trainDataSet.CloseMosaic(closeMosaic);

                float[] trainLoss_items = TrainEpoch(trainDataLoader, amp, epoch, nb, nw);
                // [Fix 2026-11-02] Mid-epoch stop signal: empty array → skip Val/save, exit training
                // loop immediately (last.bin/best.bin from completed epochs preserved as-is).
                if (trainLoss_items.Length == 0)
                {
                    break;
                }
                lr_scheduler.step();

                var trainingBackup = SaveModelState(yolo);
                ema.ApplyTo(yolo);
                (float[] valLoss_items, float[] metrics) = Val(valDataLoader, amp, epoch);
                RestoreModelState(yolo, trainingBackup);

                float fitness = GetFitness(metrics);
                // [Fix 2026-11-01] Recall guard for best-model selection: on tiny val sets a degenerate
                // epoch (precision=1.0 but recall≈0) can score a spurious high mAP50-95 and hijack
                // best.bin + freeze the early-stop patience counter (observed: best stuck at epoch 39
                // with R=0.037 while epoch 77 reached R=0.78/mAP50=0.68). A candidate must recall at
                // least 10% of ground truth before it may compete for best / reset patience.
                // 召回守门：病态轮(P高R极低)在小验证集上mAP50-95偶发虚高会劫持best与早停计数，
                // 要求R>=0.1才有资格竞争best/重置patience。
                bool recallHealthy = metrics.Length > 1 && metrics[1] >= 0.1f;
                float effectiveFitness = recallHealthy ? fitness : float.NegativeInfinity;

                if (fitness > best_fitness && recallHealthy)
                {
                    best_fitness = fitness;
                    SaveWeight(Path.Combine(weightsPath, "best.bin"), ema.GetStateDict(yolo));
                    // [Fix 2026-11-02] Export ONNX from the EMA weights (the same weights that produced
                    // `fitness` during Val). Previously ExportOnnx ran on the model already restored to
                    // raw training weights by RestoreModelState — so best.onnx carried training weights
                    // while best.bin carried EMA weights, and deployment scores dropped (observed 0.77
                    // vs 0.9+ for the same checkpoints). Apply EMA, export, then restore again.
                    // 部署onnx必须导EMA权重（与Val算分一致的权重）：先应用EMA→导出→再恢复训练权重，
                    // 否则best.onnx=训练权重而best.bin=EMA权重，部署分数偏低。
                    ema.ApplyTo(yolo);
                    ExportOnnx(Path.Combine(weightsPath, "best.onnx"));   // deployment format produced in sync (directly loadable by OV/TRT) / 部署格式同步产出（OV/TRT可直接加载）
                    RestoreModelState(yolo, trainingBackup);
                }

                bool shouldStop = stopper.ShouldStop(effectiveFitness, epoch);

                if (shouldStop)
                {
                    break;
                }

                SaveWeight(Path.Combine(weightsPath, "last.bin"));
                Console.WriteLine();
                stopwatch.Stop();

                // [Fix 2026-11-05] Epoch-end hygiene: TorchSharp tensors free native memory via GC
                // finalizers — Val's NMS/metric tensors accumulate unmanaged allocations and CUDA
                // caching-allocator fragmentation, making epochs progressively slower (observed
                // 147s → 273s by epoch 5). Force collection + empty cache each epoch to keep
                // per-epoch time flat.
                // 每轮末清理：Val的NMS/指标张量靠GC终结器释放原生内存，逐轮累积导致越训越慢
                //（实测147s→273s）。强制回收使每轮耗时保持平稳（TorchSharp 0.105无empty_cache API）。
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();

                WriteLog(epoch, stopwatch.ElapsedMilliseconds / 1000f,
                         trainLoss_items, valLoss_items, metrics, nb, valBatches);

                // [Fix 2026-11-05] Redraw results.png EVERY epoch (ultralytics behavior): the old
                // code drew only once after training finished — users watching progress mid-train
                // kept seeing the stale/missing chart and concluded fixes "did nothing". Cheap
                // (a few ms) since the CSV is tiny.
                // 每轮重绘results.png（对齐python）：旧代码只在训练结束后画一次，
                // 训练中查看永远是旧图/无图——像"修复无效"。
                try { DrawCurve(); } catch { }
            }
            DrawCurve();

            Console.WriteLine("Train Done.");

            void WriteLog(int epoch, float time, float[] trainLoss_Items, float[] valLoss_Items,
                          float[] metrics, long trainBatches, long valBatches)
            {
                if (!Directory.Exists(config.OutputPath))
                {
                    Directory.CreateDirectory(config.OutputPath);
                }
                string fileName = Path.Combine(config.OutputPath, "log.csv");
                StringBuilder stringBuilder = new StringBuilder();
                if (!File.Exists(fileName))
                {
                    stringBuilder.AppendLine(GetSeperatLogHeaders());
                }

                stringBuilder.Append($"{epoch}, {time}, ");
                foreach (float item in trainLoss_Items)
                {
                    stringBuilder.Append($"{item / trainBatches}, ");
                }
                foreach (float item in valLoss_Items)
                {
                    stringBuilder.Append($"{item / valBatches}, ");
                }
                foreach (float item in metrics)
                {
                    stringBuilder.Append($"{item}, ");
                }
                stringBuilder.AppendLine($"{trainLoss_Items.Sum() / trainBatches}, {valLoss_Items.Sum() / valBatches}");
                File.AppendAllText(fileName, stringBuilder.ToString());
            }

            void WriteConfig()
            {
                if (!Directory.Exists(config.OutputPath))
                {
                    Directory.CreateDirectory(config.OutputPath);
                }
                string fileName = Path.Combine(config.OutputPath, "config.txt");
                StringBuilder stringBuilder = new StringBuilder();
                stringBuilder.AppendLine("Training Settings:");
                stringBuilder.AppendLine($"Date Time: {DateTime.Now}");
                stringBuilder.AppendLine(config.ToString());
                File.AppendAllText(fileName, stringBuilder.ToString());
            }

            void DrawCurve()
            {
                string fileName = Path.Combine(config.OutputPath, "log.csv");
                Dictionary<string, List<float>> dictionary = Tools.LoadCSV(fileName);
                Multiplot multiplot = new Multiplot();

                int plotCount = dictionary.Keys.Count - 4;
                if (string.IsNullOrEmpty(dictionary.Keys.Last()))
                {
                    plotCount = plotCount - 1;
                }

                multiplot.AddPlots(plotCount);
                multiplot.Layout = new ScottPlot.MultiplotLayouts.Grid(2, plotCount / 2);

                int max_count = Math.Min(plotCount, 10);
                for (int i = 0; i < max_count; i++)
                {
                    string name = dictionary.Keys.ToArray()[i + 2];
                    List<float> x = dictionary["Epoch"];
                    List<float> y = dictionary[name];

                    Plot plot = multiplot.Subplots.GetPlot(i);
                    plot.Title(name);
                    plot.Add.Scatter(x, y);
                    // [Fix 2026-11-05] Integer-only manual x ticks: with few epochs the automatic
                    // generator emits fractional positions that render as repeated "1 1 1 1…"
                    // labels (observed 8×"1" at epoch 2). Build integer tick positions manually —
                    // ScottPlot 5.1.57 has no IntegerTicks switch.
                    // x轴手动整数刻度：早期少量epoch时自动刻度显示成重复的"1"。
                    try
                    {
                        int epMax = (int)Math.Ceiling(x.Count > 0 ? x.Max() : 1);
                        int step = Math.Max(1, epMax / 10);
                        var ticks = Enumerable.Range(1, epMax).Where(v => (v - 1) % step == 0 || v == epMax)
                            .Select(v => (double)v).ToArray();
                        var manual = new ScottPlot.TickGenerators.NumericManual();
                        foreach (double t in ticks) manual.AddMajor(t, t.ToString("0"));
                        plot.Axes.Bottom.TickGenerator = manual;
                    }
                    catch { }
                }

                string plotImagePath = Path.Combine(config.OutputPath, "results.png");
                multiplot.SavePng(plotImagePath, 200 * plotCount, 1200);
            }
        }

        internal virtual float[] TrainEpoch(YoloDataLoader trainDataLoader, AMPWrapper amp, int epoch, long nb, long nw)
        {
            Func<int, double> lrLambda = config.UseCosLR ?
                    OneCycle(1.0, config.Lrf, config.Epochs) :
                    LrLambda(1.0, config.Lrf, config.Epochs);

            using (Tqdm<Dictionary<string, torch.Tensor>> pbar = new Tqdm<Dictionary<string, torch.Tensor>>(trainDataLoader, total: (int)trainDataLoader.Count, barStyle: Tqdm.BarStyle.Classic, barColor: Tqdm.BarColor.White, barWidth: 10, showPartialChar: true))
            {
                yolo.train();
                torch.Tensor loss_items = torch.empty(0);
                int i = 0;
                // [Add 2026-11-05] batch-timing watchdog: bs16 on ≤8GB VRAM spills to shared memory
                // (observed 5.3s/batch vs 1.1s at bs8). Warn once from real measurements instead of
                // probing device memory (API unavailable in TorchSharp 0.105).
                // 批时长看门狗：bs16在8GB卡显存换页(5.3s/批vs bs8的1.1s)——实测提示。
                var batchWatch = System.Diagnostics.Stopwatch.StartNew();
                double batchMsSum = 0; int batchMsN = 0; bool warnedSpill = false;
                foreach (Dictionary<string, torch.Tensor> data in pbar)
                {
                    // [Fix 2026-11-02] Batch-level stop check: abort mid-epoch so "停止训练" takes
                    // effect within one batch (~seconds) instead of after the whole epoch. Partial
                    // epoch progress is discarded; last.bin/best.bin from completed epochs remain.
                    // batch级停止检查：点停后一个batch内（秒级）生效，不再等整个epoch跑完。
                    try
                    {
                        if (config.StopRequested != null && config.StopRequested())
                        {
                            Console.WriteLine("[Txm] Training stopped by user request mid-epoch (epoch " + epoch + ", batch " + i + ").");
                            return new float[0];   // 空数组=中途停止信号：主循环跳过Val/保存直接break
                        }
                    }
                    catch { }
                    using (torch.NewDisposeScope())
                    {
                        // Warm Up — [Fix 2026-11-09] full python warmup triplet (trainer.py L527-541):
                        //  ① lr interp per group: bias group starts at warmup_bias_lr (0.1), others at 0,
                        //     both rising to initial_lr*lf(epoch). Old code guessed the bias group by
                        //     POSITION (paraId==0 = actually the weight group) — bias warmup never applied.
                        //  ② momentum interp: 0.8 (warmup_momentum) → 0.937 over the warmup window.
                        //     Old code kept momentum fixed 0.937 from step 1 — high momentum on the very
                        //     first noisy gradients amplifies early spikes (observed ep2 cls=79.7 burst).
                        //  ③ accumulate window interp: 1 → accumulateSteps progressively — early epochs
                        //     update more often (effective batch grows into 64), matching python's
                        //     `accumulate = max(1, interp(ni, xi, [1, nbs/bs]).round())`.
                        // 三件套对齐python：①bias组按引用识别(位置猜是weight组,预热从未生效)
                        // ②momentum 0.8→0.937预热(旧版恒0.937放大早期噪声尖峰) ③累积窗口1→n渐进。
                        long ni = i + nb * (epoch - 1);
                        if (ni <= nw)
                        {
                            double[] xi = new double[] { 0, nw };
                            foreach (var x in amp.Optimizer.ParamGroups)
                            {
                                var d = x.InitialLearningRate * lrLambda(epoch);
                                double start = (x is TorchSharp.Modules.SGD.ParamGroup pg && ReferenceEquals(pg, _biasGroupRef)) ? config.WarmUpBiasLr : 0.0;
                                x.LearningRate = Interp(ni, xi, new double[] { start, d });
                                // ② momentum interp (python L541) — ParamGroup concrete class exposes Momentum
                                if (x is TorchSharp.Modules.SGD.ParamGroup pgm) pgm.Momentum = Interp(ni, xi, new double[] { 0.8, 0.937 });
                            }
                            // ③ accumulate window interp (python L535)
                            int accNow = Math.Max(1, (int)Math.Round(Interp(ni, xi, new double[] { 1, _accumulateFull })));
                            amp.SetAccumulate(accNow);
                        }

                        if (data["batch_idx"].NumberOfElements < 1)
                        {
                            continue;
                        }
                        yolo.train();

                        (torch.Tensor ls, torch.Tensor loss_detach) = amp.TrainStep(data["images"].to(config.Dtype), data, loss);

                        // [Add 2026-11-05] spill watchdog sampling
                        batchMsSum += batchWatch.ElapsedMilliseconds; batchMsN++;
                        batchWatch.Restart();
                        if (!warnedSpill && batchMsN >= 8 && batchMsSum / batchMsN > 3000)
                        {
                            warnedSpill = true;
                            Console.WriteLine();
                            Console.WriteLine("[Txm] 提示: 平均每批 " + (batchMsSum / batchMsN / 1000.0).ToString("0.0") + "s（>3s）——大概率显存溢出换页。");
                            Console.WriteLine("[Txm] 建议将 Batch Size 降为 8 再训（实测 v9-s: bs8=54s/轮 vs bs16=220s/轮）。");
                        }

                        if (ema != null)
                        {
                            ema.Update(yolo);
                        }

                        if (loss_items.NumberOfElements < 1)
                        {
                            loss_items = torch.zeros_like(loss_detach);
                        }
                        loss_items = loss_items + loss_detach.to(loss_items.dtype, loss_items.device);
                        loss_items = loss_items.MoveToOuterDisposeScope();

                        float[] ls_items = (loss_detach).data<float>().ToArray();
                        StringBuilder stringBuilder = new StringBuilder();
                        stringBuilder.AppendFormat("{0,10}", epoch + "/" + config.Epochs);
                        foreach (float Items in ls_items)
                        {
                            stringBuilder.AppendFormat("{0,10:f3}", Items);
                        }
                        stringBuilder.AppendFormat("{0,10}", data["batch_idx"].NumberOfElements);
                        stringBuilder.AppendFormat("{0,10}", data["images"].shape[2]);
                        pbar.SetDescription(stringBuilder.ToString());
                    }
                    i++;
                }

                if (loss is Loss.E2EOBBLoss)
                {
                    ((Loss.E2EOBBLoss)loss).update();
                }
                return loss_items.@float().data<float>().ToArray();
            }
        }

        internal List<YoloResult> ImagePredict(torch.Tensor orgImage)
        {
            return ImagePredict(orgImage, config.PredictThreshold, config.IouThreshold);
        }

        internal abstract List<YoloResult> ImagePredict(torch.Tensor orgImage, float predictThreshold, float iouThreshold);

        internal abstract (float[] loss, float[] metrics) Val(YoloDataLoader valDataLoader, AMPWrapper amp, int epoch);

        internal abstract float GetFitness(float[] metrics);


        internal abstract string GetSeperatLogHeaders();

        /// <summary>
        /// Match predictions to ground truth objects using IoU.
        /// </summary>
        /// <param name="pred_classes">Predicted class indices of shape (N,).</param>
        /// <param name="true_classes">Target class indices of shape (M,).</param>
        /// <param name="iou">An NxM tensor containing the pairwise IoU values for predictions and ground truth.</param>
        /// <param name="use_scipy">Whether to use scipy for matching (more precise).</param>
        /// <returns>Correct tensor of shape (N, 10) for 10 IoU thresholds.</returns>
        internal torch.Tensor match_predictions(torch.Tensor pred_classes, torch.Tensor true_classes, torch.Tensor iou, bool use_scipy = false)
        {
            using (torch.NewDisposeScope())
            using (torch.no_grad())
            {
                torch.Tensor iouv = torch.linspace(0.5f, 0.95f, 10, dtype: torch.ScalarType.Float32);

                // Dx10 matrix, where D - detections, 10 - IoU thresholds
                torch.Tensor correct = torch.zeros(new long[] { pred_classes.shape[0], iouv.shape[0] }, dtype: torch.ScalarType.Bool);

                // LxD matrix where L - labels (rows), D - detections (columns)
                torch.Tensor correct_class = true_classes[torch.TensorIndex.Ellipsis, torch.TensorIndex.None] == pred_classes;
                iou = iou * correct_class;  // zero out the wrong classes
                for (int i = 0; i < iouv.NumberOfElements; i++)
                {
                    float threshold = iouv[i].ToSingle();
                    torch.Tensor matches = torch.nonzero(iou >= threshold);  // IoU > threshold and classes match
                    if (matches.shape[0] > 0)
                    {
                        if (matches.shape[0] > 1)
                        {
                            matches = matches[iou[matches[torch.TensorIndex.Ellipsis, 0], matches[torch.TensorIndex.Ellipsis, 1]].argsort(descending: true)];
                            matches = GetUniqueMatches(matches);
                        }

                        correct[matches[torch.TensorIndex.Ellipsis, 1], i] = true;
                    }
                }
                return correct.to(pred_classes.device).MoveToOuterDisposeScope();
            }

            torch.Tensor GetUniqueMatches(torch.Tensor matches)
            {
                using (torch.no_grad())
                {
                    if (matches.ndim != 2 || matches.shape[1] != 2)
                    {
                        throw new ArgumentException("matches shape must be [n, 2]");
                    }
                    matches = GetUniqueByColumn(matches, columnIndex: 1);
                    matches = GetUniqueByColumn(matches, columnIndex: 0);
                    return matches;
                }
            }

            torch.Tensor GetUniqueByColumn(torch.Tensor matches, int columnIndex)
            {
                using (torch.NewDisposeScope())
                using (torch.no_grad())
                {
                    torch.Tensor columnValues = matches[torch.TensorIndex.Ellipsis, columnIndex];
                    (torch.Tensor uniqueValues, torch.Tensor inverseIndices, _) = columnValues.unique(return_inverse: true);

                    long n = columnValues.shape[0];
                    var firstOccurrence = torch.full(new long[] { uniqueValues.shape[0] }, -1L, torch.ScalarType.Int64, device: matches.device);

                    for (long i = 0; i < n; i++)
                    {
                        long inverseIdx = inverseIndices[i].item<long>();
                        if (firstOccurrence[inverseIdx].item<long>() == -1)
                        {
                            firstOccurrence[inverseIdx] = i;
                        }
                    }

                    return (matches.index_select(0, firstOccurrence)).MoveToOuterDisposeScope();
                }
            }

        }

        internal abstract string GetValDescription();

        internal abstract string GetTrainDescription();

        internal void One2one_Init()
        {
            if (config.TaskType != TaskType.Classification)
            {
                Head.Detect head = (Head.Detect)(yolo.named_children().ToArray()[0].module.named_children().Last().module);
                head.one2one_init();
            }
        }

        internal void Bias_Init()
        {
            if (config.TaskType != TaskType.Classification)
            {
                Head.Detect head = (Head.Detect)(yolo.named_children().ToArray()[0].module.named_children().Last().module);
                head.bias_init();
            }
        }

        /// <summary>导出ONNX部署格式（失败不阻断训练——只告警）</summary>
        /// <summary>Export current weights to ONNX beside best.bin (deployment format; failure does not block training) / 导出当前权重为ONNX（失败不阻断训练）</summary>
        internal void ExportOnnx(string path)
        {
            try
            {
                yolo.eval();
                OnnxExport.OnnxExporter.Export(yolo, new long[] { 1, 3, config.ImageSize, config.ImageSize }, path);
                Console.WriteLine($"ONNX exported: {path}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"WARN: ONNX export failed (training continues): {ex.Message}");
            }
        }

        internal void SaveWeight(string path, Dictionary<string, torch.Tensor> stateDict = null)
        {
            var source = stateDict ?? yolo.state_dict();
            Dictionary<string, torch.Tensor> sd = new Dictionary<string, torch.Tensor>();
            foreach (var li in source)
            {
                if (!li.Key.Contains("one2one"))
                {
                    sd.Add(li.Key, li.Value);
                }
            }

            using FileStream output = File.Create(path);
            using BinaryWriter writer = new BinaryWriter(output);
            Encode(writer, sd.Count);
            foreach (KeyValuePair<string, torch.Tensor> item2 in sd)
            {
                writer.Write(item2.Key);
                item2.Value.Save(writer);
            }
        }

        private Dictionary<string, torch.Tensor> SaveModelState(torch.nn.Module model)
        {
            var backup = new Dictionary<string, torch.Tensor>();
            using (torch.no_grad())
            {
                foreach (var kv in model.state_dict())
                {
                    backup[kv.Key] = kv.Value.clone();
                }
            }
            return backup;
        }

        private void RestoreModelState(torch.nn.Module model, Dictionary<string, torch.Tensor> backup)
        {
            using (torch.no_grad())
            {
                var current = model.state_dict();
                foreach (var kv in backup)
                {
                    if (current.TryGetValue(kv.Key, out var p))
                    {
                        p.copy_(kv.Value);
                    }
                }
            }
        }

        private static Func<int, double> OneCycle(double y1, double y2, int steps)
        {
            return (int x) =>
            {
                double progress = x * Math.PI / steps;
                double cosVal = Math.Cos(progress);
                double factor = (1 - cosVal) / 2;
                factor = Math.Max(factor, 0);
                return factor * (y2 - y1) + y1;
            };
        }

        private static Func<int, double> LrLambda(double y1, double y2, int steps)
        {
            return (int epoch) =>
              {
                  double x = (double)epoch / steps;
                  double factor = Math.Max(1 - x, 0) * (y1 - y2) + y2;
                  return factor;
              };
        }

        public static double Interp(double x, double[] xp, double[] fp)
        {
            if (xp.Length != fp.Length)
                throw new ArgumentException("xp and fp must have the same length.");
            if (xp.Length == 0)
                throw new ArgumentException("xp and fp must not be empty.");

            if (x <= xp[0]) return fp[0];
            if (x >= xp[xp.Length - 1]) return fp[fp.Length - 1];

            int index = Array.BinarySearch(xp, x);
            if (index >= 0)
                return fp[index];

            index = ~index;
            double x0 = xp[index - 1];
            double x1 = xp[index];
            double y0 = fp[index - 1];
            double y1 = fp[index];

            double t = (x - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }

        private static void Encode(BinaryWriter writer, long value)
        {
            if (value < 0)
            {
                throw new NotImplementedException("LEB128 encoding of negative numbers");
            }

            long num;
            while (true)
            {
                num = value & 0x7F;
                value >>= 7;
                if (value == 0L)
                {
                    break;
                }

                writer.Write((byte)(num | 0x80));
            }

            writer.Write((byte)num);
        }

    }

}
