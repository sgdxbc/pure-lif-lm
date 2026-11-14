# GPU 集群初始化报告（2026-11-15，集群会话 on xlogin1）

## 1. 集群情况（NUS SoC SLURM）
- 分区：`gpu`（单作业上限 3 小时）、`gpu-long`（上限 3 天）；另有 CPU 分区 `normal`（3 小时）、`long`（3 天）。
- GPU 型号（节点特性里标了 CUDA 算力）：
  | GRES | 节点 | 说明 |
  |---|---|---|
  | `gpu:h100-96` | xgpi[0-10]，每节点 2 卡 | H100 NVL 94GB；gpu-long 上有 xgpi[6-10] |
  | `gpu:h100-47` | xgpi[11-20]，每节点 4 卡 | H100 47GB（MIG 切分或小显存版） |
  | `gpu:a100-80` | xgph[0-9]、xgpj0，每节点 1 卡 | gpu-long 上有 xgph[5-9] |
  | `gpu:a100-40` | xgpg[0-7] 1 卡、xgph[10-15] 2 卡 | |
  | `gpu:h200-141` | xgpk0，4 卡 | 只在 gpu 分区（3 小时） |
  | `gpu:nv` | xgpd（Titan V）、xgpe（Titan RTX）、xgpf | 老卡 |
- 每人同时可用上限（QOS normal 的 MaxTRESPerUser）：总共 8 卡；其中 h100-96 1 张、a100-80 1 张、h100-47 2 张、
  a100-40 2 张、h200 1 张、nv 8 张；CPU 256 核。没有看到 GPU 时长配额，用的是 fair-share（目前 FairShare 0.69）。
  计费权重 h100-96=16、a100-80=12、h100-47=8、a100-40=6、h200=48，用得多 fair-share 会下降、排队变慢。
- 计算节点能上外网（xgpi3 上 curl huggingface.co 返回 200）。
- 没有 environment modules；驱动 580.173.02。用 pip 装 CUDA 版 PyTorch 即可。
- **登录节点的限制（重要）**：
  - 每个进程虚拟内存上限 1GB（`ulimit -v 1048576`）：登录节点上连 `import torch` 都会失败，
    也装不下 2.2GB 的训练文本，所以分词放在 CPU 作业里做（`slurm/prepare.sbatch`，normal 分区；下载仍在登录节点）。
  - 登录节点 `/tmp` 每人配额 10MB：pip 默认在 /tmp 解包会报 Disk quota exceeded，要把 `TMPDIR` 指向家目录。
  - 家目录是 NFS 大容量存储，当前占用约 9GB（venv 约 6GB + 数据 3.4GB），没有遇到配额问题。

## 2. 环境
- venv：`~/venv-lif`（系统 Python 3.12.3），torch 2.11.0+cu128、triton 3.6.0、numpy、tokenizers、ninja、matplotlib。
- 每个作业 `source slurm/env.sh`：激活 venv，把 XDG/pip/torch/HF/Triton/torch_extensions 缓存都指到 `~/.cache`，
  TMPDIR 用计算节点的 /tmp。
- 作业脚本在 `slurm/`，日志在 `slurm/logs/`。

## 3. 数据
- TinyStoriesV2-GPT4 train/valid 下载到 data/，`prepare_data.py` 复用仓库里的 tok4096.json：
  valid 5,577,952 token、train 552,496,754 token，与之前完全相同。
- `md5sum -c checksums.md5`：tok4096.json / valid.bin / train.bin 全部 OK（日志 `slurm/logs/lif-prep-1142625.out`）。

## 4. GPU 检查（作业 1142626，xgpi3，1 × H100 NVL 94GB）
日志：`slurm/logs/lif-gpucheck-1142626.out`。
- `test_lif_gpu.py`：24 种配置（固定泄漏 / 逐步泄漏 × 软/硬重置 × detach × 3 种替代梯度）脉冲全部一致，
  各梯度最大相对误差 1.2e-05；bf16 路径的脉冲与 float32 一致。**全部通过。**
- 内核单独计时（fwd+bwd，32×256×N，bf16）：N=1536/2048/4096 都在 0.66–0.87 ms。
- 训练速度（batch 32 × 256 = 8192 token/步，跑 20 步，取第 20 步的读数）：

  | 规模 | 参数 | LIF 神经元 | tok/s | 峰值显存 |
  |---|---|---|---|---|
  | 12M（无泄漏突触） | 12.63M | 12,288 | 170,422* | 1.5 GB |
  | 12M + leak64 | 13.62M | 12,288 | 499,381 | 1.7 GB |
  | 25M + leak64（10 层 × 2048） | 26.89M | 20,480 | 664,766 | 2.7 GB |
  | 117M + leak128（12 层 × 4096，d_model 1024） | 117.11M | 49,152 | 307,416 | 6.6 GB |

  \* 第一个测试承担了 Triton 内核首次编译的开销，这个数偏低，不代表无泄漏版本更慢。
  对比：CPU 上 12M 约 11,500 tok/s，GPU 快 40 倍以上。batch 32×256 远没有用满 H100（显存只用了几 GB），
  加大 batch 吞吐还会明显上升。按 30 万 tok/s 估算，117M 跑完全部 5.52 亿训练 token 约 30 分钟。

## 5. 遇到的问题
1. 登录节点的 1GB 虚拟内存和 10MB /tmp 限制（见上），已绕开：分词放进 CPU 作业，TMPDIR 指向家目录。
   CLAUDE.md 里“在登录节点下载/分词”的说法要改成“登录节点只下载和 git”。
2. **117M 配置在 20 步内发散**：loss 15.99（随机初始化约 8.3）、梯度范数 177、验证 loss 13.38、发放率升到 0.12，
   深层到 0.27。其他三个规模都正常（loss 约 4.7–5.0）。这次 bench 的学习率调度是 20 步内 warmup 5 步再余弦，
   峰值和 CPU 配置相同；还没查是初始化、学习率还是深层发放率失控的问题。上大模型之前需要先解决
   （比如降学习率、按宽度缩放初始化、或者梯度裁剪），可以在第一个 GPU 实验里顺带排查。
3. 我无法给云端会话发 send_message（服务器要求受信任设备），以 git 为准。

## 6. 下一步建议
- 长训练放 `gpu-long`（h100-96 在 xgpi[6-10]、a100-80 在 xgph[5-9]），或在 `gpu` 分区用 3 小时作业 + `--resume` 接力。
- 每人只有 1 张 h100-96 和 1 张 a100-80；筛选实验可以同时用 2 张 h100-47 和 2 张 a100-40 并行。
