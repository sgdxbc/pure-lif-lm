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
| `g12_*`、`g14_*`、`g15_*`、`g16_*` | 第十二、十四、十五、十六批：在 TinyStories 上快速筛选的改动 | TinyStories | 5.26、5.27、5.29、5.31 |
| `g13_*`、`g17_*` | 第十三、十七批：通用文本 | FineWeb-Edu | 5.28、5.30、5.32 |
| `g18_*` | 第十八批：记忆的大小、学习率、更强的老师 | FineWeb-Edu | 5.33 |
| `g19_*`、`g20_*` | 第十九、二十批：记忆的容量，放在哪几层 | FineWeb-Edu | 5.35 |
| `g21_*`、`g22_*`、`g23_*` | 第二十一到二十三批：梯度尖峰、按 (1 − λ) 缩放写入、长训练前的检查 | FineWeb-Edu | 5.34、5.35 |
| `g24_*` | 第二十四批：wsd 学习率表 | FineWeb-Edu | 5.24 |
| `g25_*`、`g27_*` | 第二十五、二十七批：词的组合输入和两级读出 | FineWeb-Edu | 5.36 |
| `g26_*` | 第二十六批：长训练（98 亿 token）和它的分支，以及同样训练的 Transformer | FineWeb-Edu | 5.37 |
| `g28_*`、`g28x_*` | 第二十八批：按 4 位权重训练 | FineWeb-Edu | 5.38 |
| `g29_*` | 第二十九批：硬件网络 | FineWeb-Edu | 5.39 |
| `g30_*` | 第三十批：输入行不做权重衰减 | FineWeb-Edu | 5.40 |
| `g31_*` | 第三十一批：Transformer 的平均权重 | FineWeb-Edu | 5.41 |
| `g32_*`、`g34_*` | 第三十二、三十四批：长训练后期换老师（GPT-2 XL、Medium） | FineWeb-Edu | 5.42、5.45 |
| `g33_*` | 第三十三批：和硬件网络质量相当的 Transformer | FineWeb-Edu | 5.43 |
| `g35_*` | 第三十五批：少做稠密乘加（`--in_rank`、`--mem_src spikes`） | FineWeb-Edu | 5.46 |
| `g36_*` | 第三十六批：线性注意力的对照组 | FineWeb-Edu | 5.47 |
| `g37_*` | 第三十七批：实数输出的孪生网络、先软后硬 | FineWeb-Edu | 5.48 |
| `g38_*`、`g39_*` | 第三十八、三十九批：总线归一化，以及线性注意力和脉冲网络之间的"梯子" | FineWeb-Edu | 5.49、5.50 |
| `g40_*` | 第四十批：小规模排名（77M 的形状） | FineWeb-Edu | 5.51 |
| `g41_*` | 第四十一批：一个词里可以发多个脉冲的中继（77M） | FineWeb-Edu | 5.52 |
| `g42_*` | 第四十二批：脉冲网络里的门控线性注意力逐项去掉特点 | FineWeb-Edu | 5.53 |
| `g43_*` | 第四十三批：中继、三值权重和两者合起来，77M 和 GPT-2 small 的形状 | FineWeb-Edu | 5.54 |
| `g44_*` | 第四十四批：门控线性注意力的长处加回可塑突触；膜电位的学习率；换种子 | FineWeb-Edu | 5.55 |
| `g45_*` | 第四十五批：77M 上脉冲的稀疏度、更宽更稀疏的结构、稀疏脉冲的训练方式 | FineWeb-Edu | 5.56 |
| `g46_*` | 第四十六批：换种子确认蒸馏；同样短配方的小 Transformer；两级读出本身的质量代价 | FineWeb-Edu | 5.58 |
| `g47_*` | 第四十七批：可塑突触三样一起的哪些部分值得它的硬件开销（0/1 键查询、隔层、键查询不带膜电位） | FineWeb-Edu | 5.59 |
| `g48_*` | 第四十八批：稠密线性注意力对照（31M、50M 形状）；学习率降到 0 | FineWeb-Edu | 5.60 |
| `g49_*` | 第四十九批：从 `g46_kd25s1` 第 15,000 步分出的"蒸馏一直加"、"后期减弱蒸馏"和"平滑的脉冲数梯度"（`g49_kds`，5.62） | FineWeb-Edu | 5.61 |
| `g50_*` | 第五十批：蒸馏过的 77M 稠密线性注意力和 0/1 MLP 对照；320 维窄网络的 GELU / 0/1 MLP 一对 | FineWeb-Edu | 5.62 |
| `classes/` | 两级读出和组合输入用的词的分类（`experiments/word_classes.py`） | FineWeb-Edu | 5.36 |
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
