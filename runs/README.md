# 运行结果目录

每个子目录是一次训练运行，名字的前缀表示它属于哪一批实验；分析都在 [`docs/experiments.md`](../docs/experiments.md) 对应的小节里。
权重文件 `ckpt.pt` 超过 GitHub 的文件大小上限，没有放进仓库（只有最早的模型留了 `main/model.pt`）。

## 目录名

| 前缀 | 是什么 | 数据 | 结果在 |
|---|---|---|---|
| `main/` | 第一个正式训练的模型（12.6M，CPU），带权重 `model.pt` | TinyStories | 第 4 节 |
| `abl_*` | CPU 上的短消融 | TinyStories | 第 3 节 |
| `p1_*` … `p7_*` | CPU 上的第 1–7 阶段 | TinyStories | 5.2（p1）、5.3（p2）、5.4（p3）、5.8（p4）、5.9（p5）、5.10（p6）、5.11（p7） |
| `tf_full/` | 和第二、三阶段同样 1.15 亿 token 的 Transformer 对照 | TinyStories | 5.7 |
| `g1_*` … `g10_*` | GPU 上的第 1–10 批 | TinyStories | 5.11–5.19、5.21、5.23（对照见 `experiments/README.md`） |
| `g11_*` | 第十一批：长文本 | PG-19 | 5.22 |
| `g12_*`、`g14_*`、`g15_*`、`g16_*` | 第十二、十四、十五、十六批：在 TinyStories 上快速筛选的改动 | TinyStories | 5.26、5.27、5.29、5.24 |
| `g13_*`、`g17_*` | 第十三、十七批：通用文本 | FineWeb-Edu | 5.28、5.30、5.24 |
| `ref_*` | 参考模型（GPT-2、Pythia）的零样本基准 | | 5.24 |
| `bench_gpu/` | Transformer 在 GPU 上的推理实测（vLLM） | | 5.25 |

名字里常见的缩写：`snn` 脉冲网络、`tf` Transformer，后面的数字是参数量（如 `snn212` 是 2.12 亿参数）；
`_ema` 表示用滑动平均后的权重；`T2` 每个 token 模拟 2 步；`lr6e-4` 学习率。

## 每个运行目录里的文件

| 文件 | 内容 |
|---|---|
| `args.json` | 全部训练参数。`python3 slurm/args_of.py runs/X` 打印能原样重复这次运行的命令行参数 |
| `log.jsonl` | 训练日志：每隔几百步的训练 loss、梯度范数、速度、各层发放率；每次评估的验证 loss（40 个随机窗口） |
| `stdout.log` | 训练时的屏幕输出（含生成样例） |
| `final_eval.json`、`final_eval_ema.json` | 完整验证集上的 loss（后者用滑动平均后的权重） |
| `zeroshot*.json` | 零样本基准（通用文本上的运行） |
| `probe_memory*.json`、`recall*.json` | TinyStories 上的两种回看测量（5.1 节） |
| `hw_cost.json`、`quant.json`、`fpga_est.json` | 硬件代价、低精度模拟、FPGA 估算 |
| `long_*.json` | 按位置的 loss（`eval_long.py`） |
| `*.log`、`done` | 评测脚本的输出；`done` 表示训练和评测都已完成 |

## 顶层的文件

- `scaling_fineweb.png/.json`：FineWeb-Edu 上验证 loss 对训练算力的图（5.28）。
- `ref_loss.json`：参考模型在 FineWeb-Edu 验证集上的 loss；`lambada_probe.json`：LAMBADA 诊断（5.30）。
- `gpu_setup_report.md`：集群的显卡、分区、限额和实测训练速度。
- `phase*.log`、`ablate*.log`、`after_phase*.log`：CPU 阶段排队脚本的输出；`probe_*.json`、`recall_*.json`、`drift_*.json`、
  `leak_*.json`：CPU 阶段的回看测量和诊断（5.1–5.11）。
