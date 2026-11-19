# 运行说明

怎样准备数据、训练、评测、估算硬件代价，以及 `train.py` 的全部选项。项目的目标和现状见 [README](../README.md)，
每批实验的记录见 [`docs/experiments.md`](experiments.md)。所有命令都在仓库根目录下运行。

## 环境

- Python 3.10+，PyTorch 2.x：`pip install -r requirements.txt`。
- **CPU**：LIF 神经元的时间循环用 C++/OpenMP 内核（`lif_kernel.cpp`），需要 g++ 和 OpenMP，第一次 import 时自动编译。
- **NVIDIA GPU**：用 Triton 内核（`lif_triton.py`，PyTorch 自带 Triton）。需要支持 bf16 的显卡（Ampere 及以后）。
  两种内核按张量所在的设备自动选择，方程和梯度完全相同（`test_lif.py`、`test_lif_gpu.py`；无 GPU 时可用 `TRITON_INTERPRET=1`
  在 CPU 上验证 Triton 内核：24 种设置下脉冲完全一致，梯度和参考实现的最大相对误差 2.8e-6；
  整个模型在两种内核下 float32 梯度相差 1e-6 量级）。
- `bash gpu_setup.sh`：在一台有 GPU 的机器上一次做完：检查显卡、安装依赖、下载并处理 TinyStories（和已有结果用的数据逐字节相同，
  见 `data/checksums.md5`）、内核正确性测试、各规模的训练速度。集群上的实测速度见 `runs/gpu_setup_report.md`。
- 零样本基准另需 `lm_eval`、`transformers`、`datasets`；GPU 推理实测（`bench_gpu.py`）需要 `vllm`，建议单独的环境。

## 数据

| 数据集 | 用途 | 准备 | 训练时的参数 |
|---|---|---|---|
| TinyStories | 早期实验、快速试验结构改动 | 下载两个 txt 到 `data/`（见下），`python3 prepare_data.py` | 默认（`--data data --tokenizer tok4096.json`） |
| PG-19 | 长文本实验线（docs/experiments.md 5.22） | `slurm/pg19_download.sh`，然后 `sbatch slurm/prepare_pg19.sbatch` | `--data data/pg19 --tokenizer tok8192.json --vocab 8192` |
| FineWeb-Edu（100 亿 token 样本） | 现在的主线：通用文本 | `slurm/fineweb_download.sh`，然后 `sbatch slurm/prepare_fineweb.sbatch` | `--data data/fineweb --tokenizer gpt2.json --vocab 50304` |

分词器文件（`data/tok4096.json`、`data/pg19/tok8192.json`、`data/fineweb/gpt2.json`）都在仓库里；token 文件（`*.bin`）要自己生成。

```bash
mkdir -p data
curl -L -o data/TinyStoriesV2-GPT4-train.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl -L -o data/TinyStoriesV2-GPT4-valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
python3 prepare_data.py
```

## 训练

`train.py` 训练脉冲网络（默认）或 Transformer 对照组（`--arch tf`），每次评估时把 `ckpt.pt`、`log.jsonl`、`args.json` 写进 `--out` 目录；
加 `--resume` 从上次的 `ckpt.pt` 接着训练。

**现在通用文本上用的设置**（第十三批的 212M，`experiments/gpu13.sh`）：

```bash
python3 train.py --device cuda --out runs/my_snn \
  --data data/fineweb --tokenizer gpt2.json --vocab 50304 \
  --d_model 1024 --d_hidden 4096 --n_layers 12 --leak_rank 128 \
  --hard_reset true --sg fast_sigmoid --detach_reset false --reset_learn true --gamma_init 0.999 --gamma_init_min 0.05 \
  --ema 0.999 --neuron_lr_mult 10 \
  --batch 32 --seq 1024 --steps 76300 --warmup 1000 --lr 6e-4 --eval_every 5000 --eval_batches 40 --log_every 500
```

- 第二行是数据；第三行是网络大小（总线 1024，每层 4096 个 LIF 神经元，12 层，调节漏电的突触秩 128）；
- 第四行是神经元的设置：乘法复位 `V ← V·(1 − γ·s)`，γ 每个神经元可学习、初始值在 0.05–0.999 之间均匀分布，fast-sigmoid 代理梯度；
- 第五行是训练方法：权重的滑动平均、神经元自身参数 10 倍学习率。在 TinyStories 上还加了蒸馏（`--teacher ... --distill_weight 0.9`），
  这三项合称"三种方法"（docs/experiments.md 5.12–5.15）。
- 同样设置的 Transformer 对照组：`--arch tf --d_model 768 --n_layers 12 --n_heads 12`（GPT-2 small 的形状），之后的对照组用 `--optim muon`。

一张 H200 上这个 212M 网络约每秒 29 万 token，25 亿 token 约 2.5 小时。

## 评测

| 脚本 | 做什么 | 输出 |
|---|---|---|
| `evaluate.py --ckpt runs/X/ckpt.pt [--ema] --out ...` | 完整验证集的 loss、生成样例、流式生成速度 | `final_eval[_ema].json` |
| `eval_lm.py [--ema] runs/X`，`eval_lm.py --hf 名字` | 零样本基准（lm-evaluation-harness：HellaSwag、LAMBADA、PIQA、ARC、WinoGrande），也可测 Hugging Face 上的参考模型 | `zeroshot[_ema].json` |
| `ref_loss.py 名字...` | 用 GPT-2 分词器的 Hugging Face 模型在 FineWeb-Edu 验证集上的 loss（和 `evaluate.py` 同样的算法） | `runs/ref_loss.json` |
| `probe_lambada.py [--ema] runs/X 名字...` | LAMBADA 诊断：最后一个词读完整段 / 只读最后 16 个 token 时的 loss，按是否在前文出现过分组 | `runs/lambada_probe.json` |
| `eval_long.py [--ema] [--length N] [--window W] runs/X` | 从书或文档的开头读起，按位置区间报告 loss；`--window` 让 Transformer 只看最近 W 个 token | `long_valid*.json` |
| `probe_memory.py`、`probe_recall.py` | TinyStories 上的两种回看测量（截断上文；重复出现的词按距离分组，docs/experiments.md 5.1） | `probe_memory*.json`、`recall*.json` |
| `scaling.py runs/g13_*` | 验证 loss 对训练算力的图 | `runs/scaling*.png/.json` |
| `sample.py --ckpt ... [-i]` | 文本补全（`-i` 交互） | |
| `analyze.py` | loss 曲线、脉冲发放图、时间常数分布 | `figs/` |

## 硬件代价

| 脚本 | 做什么 |
|---|---|
| `hw_cost.py [--ema] runs/X` | 每个 token 的稠密乘加、可以跳过零的加法、要读的数据量、每路状态大小（按实测发放率；docs/experiments.md 5.20） |
| `quant_sim.py [--ema] runs/X` | 低精度模拟：权重、总线、膜电位、可塑突触按硬件的整数位宽取整后的验证 loss（只看权重时用 `evaluate.py --wbits 4`；5.20、5.44） |
| `hw_export.py runs/X --ema [--windows 256]` | 把网络导出成 FPGA 原型用的整数表格（`runs/X/hw/`，约 126 MB，不进 git），可同时写出验证片段（5.44） |
| `hw_ref.cpp` | 整数参考实现（C++）：`g++ -O3 -march=native -fopenmp -ffp-contract=off -o hw_ref hw_ref.cpp`，`./hw_ref runs/X/hw runs/X/hw/windows.bin 256 256` 打印验证 loss；FPGA 程序要和它逐位一致（5.44） |
| `experiments/wbits_sens.py runs/X` | 哪类权重矩阵取整到 4 位损失最大；把最敏感的几类留在 8 位时的损失和多用的存储（5.38） |
| `fpga_est.py runs/X [--device u55c --cards 2]` | 脉冲网络在 FPGA 上的延迟、吞吐量和能耗估算（默认 AMD Alveo V80 的参数，5.25；U55C 和多卡流水线见 5.44） |
| `bench_gpu.py --model 名字` | Transformer 在 GPU 上用 vLLM 实测：吞吐量、每 token 延迟、每 token 能耗（5.25） |
| `bench_gpu_lean.py --shape D,L,H` | 同样的量，但用最简的生成程序（固定大小的键值缓存、CUDA graph），给出 GPU 本身能做到多少的界限（5.43）；`--check` 在 CPU 上核对 |

## 在 SLURM 集群上

- `slurm/train.sbatch runs/NAME [train.py 的参数]`：一个作业训练一个网络，训练完自动做完整验证；TinyStories 上再做两种回看测量，
  通用文本上再做零样本基准。`--export=ALL,RELAY=<显卡型号>` 时每段 3 小时自动接力直到训练完；`HW=1` 再算硬件代价和低精度模拟。
- 一批实验由 `experiments/gpuN.sh` 提交，结果在 `runs/gN_*`。`python3 slurm/args_of.py runs/X` 打印能原样重复某个运行的参数。
- 其他作业脚本：`slurm/zeroshot.sbatch`（零样本基准）、`slurm/hw.sbatch`（硬件代价）、`slurm/diag.sbatch`（通用文本上的诊断）、
  `slurm/ref_loss.sbatch`、`slurm/long_eval.sbatch`、`slurm/bench_gpu.sbatch`、`slurm/probe_long.sbatch` 等，用法写在各自开头。
- 集群本身的说明（分区、显卡、限额）在 `runs/gpu_setup_report.md`；`slurm/env.sh` 是每个作业都要载入的环境；
  `slurm/claude_session.sbatch` 和 `slurm/notes.txt` 是集群上 Claude 会话的运维记录，和实验本身无关。

## 全部选项

**网络结构**（`model.py` 开头有每一项的完整说明）：
- 大小：`--d_model`（总线宽度）、`--d_hidden`（每层神经元数）、`--n_layers`、`--vocab`。
- 神经元：`--hard_reset`（发放后清零，否则减去阈值）、`--reset_learn true --gamma_init a --gamma_init_min b`（乘法复位
  `V ← V·(1 − γ·s)`，γ 每个神经元可学习）、`--sg fast_sigmoid|sigmoid|triangle --alpha`（代理梯度的形状和宽度）、`--detach_reset`（复位处是否截断梯度）、
  `--tau_min/--tau_max`（时间常数的初始范围，单位 token）、`--in_norm`（慢神经元的初始化方式）、`--learn_beta/--learn_theta`。
- 初始化：`--out_init`（W_out 的初始标准差，默认 0.02）、`--head_init`（读出的初始标准差，默认 0.02）。
- `--leak_rank r`：调节漏电的突触（秩 r）。同一批脉冲除了注入电流，还通过另一组突触改变神经元漏电的快慢，
  `β[t] = sigmoid(β_logit + U·总线[t])`，网络可以按读到的内容决定一个神经元"保持"还是"清空"。现在的标准设置都用它（5.8 节起）。
- `--steps_per_token T`：每个 token 模拟 T 个时间步（5.21 节）。
- `--out_classes C [--class_file f.npy]`：两级读出（5.36 节）：C 个类读出神经元，每类最多 ⌈词表 / C⌉ 个词读出神经元；
  P(词) = P(类) × P(词 | 类)，训练和评测照常（模型输出的是 log P(词)），生成时每个词只算 C 加一类的词读出神经元。
  `--class_file` 是每个词所属的类（`experiments/word_classes.py` 生成：按词频，或对训练好的网络的读出行做均衡 k-means）；默认按编号分类。
  加 `--out_short S`：S 个词（一个"短名单"，`word_classes.py --short S` 取训练数据里最常见的 S 个，类号记为 −1）不分类，
  各有自己的读出神经元，和 C 个类读出神经元一起做一次 softmax；其余的词再分成 C 类。生成时每个词算 S + C 加一类的词读出神经元。
- `--in_groups k [--in_class_file f.npy]`：组合输入（5.36 节）：k 群各 n = ⌈词表^(1/k)⌉ 个输入神经元，每个词在每群里各让一个发放，
  注入它们的突触之和。默认按编号的 k 位数字；k = 2 时 `--in_class_file` 按类分配（第一群是类，第二群是类里的序号；用
  `word_classes.py --rows embed` 对训练好的网络的输入行聚类）。加 `--in_short S`：短名单里的 S 个词（类号 −1）各有自己的一个输入神经元，
  其余的词照常用 k 个脉冲；此时 n = ⌈(词表 − S)^(1/k)⌉，分类时类数和每类的词数都不能超过 n。
- `--mem_heads h [--mem_keys m --mem_dv dv --mem_every k]`：**可塑突触**（5.32 节）。每隔 k 层（默认每层），在神经元群之前加 h 组
  键神经元和查询神经元（每组各 m 个，时间常数短，减法复位）和一个 m × dv 的突触矩阵 S：
  `S[t] = λ·S[t-1] + k[t]·v[t]ᵀ`（k 是键神经元的脉冲，v = W_v·总线，λ 每组一个、可学习，初始时间常数在
  `--mem_tau_min` 到 `--mem_tau_max` 个 token 之间按对数均匀分布；键和查询神经元的初始阈值 `--mem_theta`，默认 1.5），读出 `r[t] = S[t]ᵀ·q[t]`（q 是查询神经元的脉冲），
  每组按 `r / sqrt(mean(r²) + 1)` 缩放（读出强时均方根约为 1，读出弱时近似原样）后经 W_o 加回总线。W_o 初始为 0，所以初始化时和不加时是同一个网络。
  - 缩放式里的常数（`--mem_eps`，默认 1）原来是 1e-6，CPU 上的小规模试训发现训练会发散：初始化时约三分之一的位置没有查询神经元碰上
    写过的键，r 正好是 0，这时缩放把梯度放大约 1000 倍，经替代梯度传到键和查询神经元的输入突触上，梯度范数几步之内涨到 10⁴–10⁵。
    改成 1 以后梯度范数和不加可塑突触时相当。
  - `--mem_write_avg true`（5.34 节，试验中）：写入 (1 − λ)·v 而不是 v，S 是过去写入的加权平均，不再随时间常数累加到很大
    （时间常数长的组里 S 的一行能到 5 万，查询神经元的梯度正比于它，这是梯度尖峰的来源）。λ 训练完就固定了，这个系数可以并进 W_v，
    所以网络和硬件不变。
  - **为什么偏离纯 LIF**：膜电位是每个神经元一个数，它能记住"最近大致发生了什么"，但很难把几十、几百个 token 之前的某个具体的词
    原样找回来（5.5、5.6 节，5.30 节的诊断）；突触可塑性是神经形态硬件里常见的机制，这里用它做一个联想记忆。
  - **硬件代价**：键、查询、值的突触和 W_o 是稠密乘加（和 W_in 同类）；写入和读出是加法（每个发放的键或查询神经元 dv 次）；
    每步 S 整体衰减一次（每个突触一次乘法）；状态多出 h·m·dv 个数（每层），固定大小。`hw_cost.py`、`fpga_est.py`、`scaling.py`、
    `quant_sim.py`（5.44 节起）都已计入。训练时按 `--mem_chunk` 个位置一块并行计算（默认 128，只影响速度）。
- `--bus_norm true`（5.49 节，试验中）：各层神经元群（W_in 和调节漏电的突触）、可塑突触（键、查询、值的突触）和读出读总线时，
  先把每个 token 的总线除以它在 d_model 个通道上的均方根（像 Transformer 的 pre-norm，没有增益）；总线本身不变。
  每个读者每个 token 多一次平方和与一次开方倒数，`hw_cost.py` 已计入；`quant_sim.py`、`hw_export.py` 还不支持（会直接报错）。
- `--mem_kind gla`（5.50 节，诊断用）：每一组可塑突触换成 `model_tf.py` 的门控线性注意力（`--mem_heads` 个头），神经元群不变。
  不是脉冲网络，没有硬件路径（`hw_cost.py` 跳过它，`quant_sim.py`、`hw_export.py` 会报错）；状态（每个头的矩阵 S 和短卷积的最后 3 个输入）
  和可塑突触一样在两次调用之间带着走，所以逐段的评测和生成都照常。

**已删除的选项**：试过、没有带来改善的结构选项已从代码里删掉，以免每次改动都要照顾它们：同层连接 `--rec_rank`（5.2 节）、
带延迟的突触 `--delay_k`（5.9 节）、脉冲化总线 `--spike_bus`（5.23 节）、输入电流按 (1 − β) 缩放 `--leak_gate`（5.26 节）；
可塑突触的纠错式写入 `--mem_rule delta`（5.32 节，代码在提交 c138343）；按位置写入的可塑突触 `--mem_clock`（5.33 节，代码在提交 c0868f9）；按写入次数归一化的读出 `--mem_count_norm`（5.34 节，代码在提交 bf7fed7）；
低秩的 W_in `--in_rank`、可塑突触读上一层的脉冲 `--mem_src spikes`（5.46 节）、输出实数的神经元 `--spike_soft` 和先软后硬的
`--soft_hold`/`--soft_end`（5.48 节；LIF 内核里的 soft 参数一起删了），代码在提交 5fcacfd。
要重复用到前四个的实验，请检出提交 490e9e6（那里的代码和文档都还在）。

**训练方法**（不改变网络结构，默认都关闭）：
- `--teacher 路径 --distill_weight a --distill_temp T`：知识蒸馏。损失 = (1−a)·交叉熵 + a·T²·KL(老师 ‖ 学生)，老师是已训练好的模型
  在同一个 batch 上的输出。验证 loss 仍只算交叉熵。`--teacher hf:名字` 用 Hugging Face 上的模型当老师（需要 transformers），
  例如 `hf:openai-community/gpt2-large`；老师的词表比学生小时，多出来的词概率记为 0。有老师时，训练开始前先报告老师自己的验证 loss。
- `--ema d`：同时保存权重的滑动平均（例如 0.999），评估时两种权重都报告；`evaluate.py --ema`、`sample.py --ema` 等使用平均后的权重。
- `--neuron_lr_mult m`：神经元自身参数（β、θ、γ）的学习率倍数。
- `--carry_state`：每个 batch 的每一行沿着训练文本连续往下读，上一段结束时的膜电位作为下一段的起点（梯度在段与段之间截断）。
  不加时每段都从静息状态开始。
- `--optim muon`：层内的二维权重用 Muon（动量 `--muon_momentum`，默认 0.95；更新乘以 0.2·√max(行数, 列数)，
  所以 AdamW 的学习率可以直接沿用），输入和读出的突触、偏置和神经元自身的参数仍用 AdamW。对 Transformer 有效，对脉冲网络无效（5.27 节）。
  `--muon_skip leak_` 让名字里含 `leak_` 的矩阵（调节漏电的突触）仍用 AdamW。
- `--hint_weight w [--hint_layers K]`：中间层蒸馏（要有老师，学生是脉冲网络）。按相对深度配成 K 对层（默认老师每层一对），
  每一对用一个只在训练时存在的线性映射，从学生这一层之后的总线（每个位置先缩放到均方根为 1）预测老师这一层之后的残差流
  （每个通道在这一批里标准化），均方误差乘以 w 加进 loss（5.29 节）。
- 其他：`--lr`、`--min_lr_frac`、`--warmup`、`--wd`、`--clip`、`--batch`、`--seq`、`--steps`、`--seed`。
- `--embed_wd w`：输入行（词向量）的权重衰减，默认（-1）和 `--wd` 相同。脉冲网络里没有归一化，长训练中权重衰减会把输入行缩小到初始的几十分之一（docs/experiments.md 5.40 节），`0` 表示输入行不衰减。
- `--schedule wsd --decay_steps D`：学习率先预热，然后保持 `--lr` 不变，只在最后 D 步按 1 − √进度 降到 `--min_lr_frac` 倍
  （默认 `cosine`：预热后按余弦从 `--lr` 降到 `--min_lr_frac` 倍）。这种学习率表下，`--steps` 更大的训练在开始降之前和更短的训练完全一样，
  所以一次长训练可以在中途分出几个"降学习率"的分支，各自得到一个训练完的结果（看质量随训练量的变化）：
  主训练加 `--keep_ckpt 50000,100000`（必须是 `--eval_every` 的倍数）在这几步另存 `ckpt_step<N>.pt`；
  分支用新的目录，加 `--init_ckpt runs/主训练/ckpt_step50000.pt --steps 60000 --decay_steps 10000`，其余参数和主训练相同，
  从那一步的权重、优化器状态和平均权重接着训练（目录里已经有 `ckpt.pt` 时，`--resume` 优先用它，所以分段接力也没问题）。
- `--spike_log x`（默认 5）：梯度范数超过最近 200 步中位数 x 倍的每一步，把最大的几个参数梯度和这批数据的位置写进 stdout 和 log.jsonl（`spike_gn` 记录；5.34 节）。只记录，不改变训练；0 表示关掉。
- `--spike_skip x`（默认 0，关）：梯度范数超过最近 200 步中位数 x 倍的一步，整步不更新（优化器和平均权重都不动，这一步仍计入学习率表和数据）。梯度裁剪只是把更新缩小，方向仍被出尖峰的那几处参数主导；跳过则这一批完全不起作用。
- `--wbits b [--wgroup g --wbits_from s]`：**按低精度权重训练**（5.38 节）。所有权重矩阵（输入行、W_in、W_out、调节漏电的突触、
  可塑突触的输入和输出、读出）在每一步的前向和反向计算前取整为 b 位对称整数，每 g 个输入（默认 128）共用一个缩放系数
  （和 `quant_sim.py` 的 `W4g128` 相同）；更新加在全精度的权重上（直通估计），所以检查点里存的仍是全精度权重。
  `--wbits_from s`：从第 s 步起才取整训练，之前照常训练。评估一律用取整后的网络；有平均权重时另外报告它不取整时的验证 loss。
  b 记在网络的配置里，`sample.load`（因此 `evaluate.py`、`eval_lm.py`、`hw_cost.py`）载入时自动取整；
  `evaluate.py --wbits 4` 和 `eval_lm.py --wbits 4` 可以把任何检查点按 4 位取整后评估（`--wbits 0` 不取整）。
  `--wbits8 '^head\.,^embed\.'`：名字和其中某个正则表达式匹配的矩阵改用 8 位（混合精度；用 `experiments/wbits_sens.py` 找最敏感的几类）。
  `hw_cost.py` 按训练时的位数（含每组 16 位的系数）算权重的存储量，`fpga_est.py` 用它判断能否整个放在片上。
- `--threads n`：CPU 上的线程数（默认 4）；`--sample_every n`：每隔几次评估生成一次样例（0 = 每次）；
  `--log_file`：训练日志的文件名（默认 `log.jsonl`；`slurm/train.sbatch` 先写到不受 git 管的文件，再复制过来）。
- `--accum n`：把每个批次分成 n 份依次计算、梯度相加（显存约降为 1/n，结果和不分时相同，只是慢一些）。不能和 `--carry_state` 同用。

**数据**：`--data 目录 --tokenizer 文件`（见上面的表）。Transformer 另有 `--rope`（旋转位置编码），最长长度取 1024 和 `--seq` 中较大的一个。
`--attn linear`（5.47 节）：把每一层的 softmax 注意力换成带门控的线性注意力（状态大小固定、不随文本变长，没有位置编码），
作为"非脉冲、但同样只有固定大小状态"的对照组。`--mlp spike`（5.49 节）：MLP 的隐藏单元换成 0/1 的阶跃（每个单元一个可学习的阈值，
用脉冲网络的替代梯度训练，没有膜电位），量"只把隐藏单元换成 0/1"在一个有归一化的普通网络里的代价。
`--relay lif|quant`（5.51 节，线性注意力）：块里每个矩阵前面一对 LIF 中继神经元，矩阵只读三值的脉冲（`quant`：同样的三值，没有膜电位）。
`--relay_levels K`（5.52 节，只用于 `quant`）：中继神经元一个词里最多发 K 个脉冲（输出 −K…K 倍的 θ），默认 1。
`--wq ternary`（5.51 节）：块里的矩阵用三值权重（BitNet b1.58 的取整）。
`--mlp lif`（5.50 节）：隐藏单元换成 LIF 神经元（泄漏、阈值、复位每个单元各自可学习，初始化和脉冲网络的神经元群一样）。

## 试用最早的模型

仓库里只有最早的 12.6M 模型（TinyStories，CPU 上训练，docs/experiments.md 第 4 节）带有权重 `runs/main/model.pt`，
其余运行的 `ckpt.pt` 超过 GitHub 的文件大小上限，没有放进仓库。分词器 `data/tok4096.json` 在仓库里，不需要下载语料：

```bash
python3 sample.py --ckpt runs/main/model.pt --prompt "Once upon a time" --num 3
python3 sample.py --ckpt runs/main/model.pt -i
```

从头复现这个模型（CPU，约 3 小时）：

```bash
python3 train.py --out runs/main --steps 14000 --batch 32 --lr 3e-3 --warmup 300 \
    --hard_reset true --sg fast_sigmoid --detach_reset false
```
