# 纯 LIF 脉冲语言模型（TinyStories 预训练，CPU）

一个**只由 LIF 神经元组成的有状态计算流**，在 TinyStories 上从零预训练，用来做英文故事的文本补全。

## 1. "纯 LIF"具体指什么

| 部件 | 在神经元层面是什么 | 代码 |
|---|---|---|
| 输入编码 | 每个词表项对应一个"输入神经元"。读到 token `x_t`，该神经元在第 t 步发放 1 个脉冲，经突触注入电流（数学上等于 embedding 查表） | `SpikingLM.embed` |
| 隐藏层 | 8 个 LIF 神经元群，每群 1536 个，共 12,288 个 LIF 神经元 | `LIFPopulation` |
| LIF 动力学 | `V[t] = β·V[t-1] + I[t]`；`s[t] = [V[t] ≥ θ]`；硬复位 `V ← V·(1-s)`。β（泄漏）和 θ（阈值）每个神经元各自可学习 | `lif_kernel.cpp` |
| 神经元之间的连接 | 只传 0/1 脉冲，突触是线性加权求和（外加一个常数背景电流，即偏置） | `w_in`, `w_out` |
| 跨层连接 | 第 l 群从输入神经元以及之前所有群接收脉冲。这些跳跃突触经过一条 384 维的"电流总线"做了低秩分解：`bus_l = embed[x_t] + Σ_{k<l} W_out_k·s_k`，`I_l = W_in_l·bus_l + b_l`。总线只是脉冲的线性加权和，**没有状态，也没有非线性**；它起的作用相当于 Transformer 的残差流 | `SpikingLM.forward` |
| 输出解码 | 词表大小的一组"读出神经元"，只积分、不发放（β=0），膜电位 = `W_head·bus_L` 就是 logits，经 softmax 后采样出下一个 token | `SpikingLM.head` |
| 记忆 | **只存在于膜电位里**。生成时每一步只携带 12,288 个膜电位，不保存、也不回看历史 token | `SpikingLM.generate` |

**不包含**：注意力、LayerNorm/RMSNorm/BatchNorm、门控或任何乘法交互、层内循环矩阵。

时间步约定：每个 token 对应一个模拟时间步；在同一时间步内，脉冲逐层向上传播完毕（这是深度 SNN 训练中常用的离散时间约定）。

## 2. 训练方法

- 代理梯度 + 沿时间反向传播（BPTT）：前向用阶跃函数，反向用 fast-sigmoid 代理导数 `1/(1+α|z|)²`，其中 `z=(V-θ)/θ`，α=4。
- LIF 的时间循环写成了 C++/OpenMP 内核（`lif_kernel.cpp`），前向和反向都是手写的，并与纯 PyTorch 参考实现逐项核对过（`test_lif.py`，12 种配置的相对误差都在 1e-6 量级）。
- 矩阵乘法用 bf16（这颗 CPU 带 AMX，能跑到约 3 TFLOPS），膜电位用 fp32 计算和保存。
- 分词：在 TinyStories 上训练的 4096 词表字节级 BPE（平均约 3.95 字符/token）。

## 3. 消融实验（每组 400 步 × 8192 token ≈ 330 万 token，验证 loss）

| 配置 | 验证 loss | 现象 |
|---|---|---|
| 软复位、sigmoid 代理梯度、截断复位梯度、lr 2e-3（base） | 3.141 | 梯度范数常冲到 100 以上；前 4 层几乎不发放（<1%） |
| base + 改用"缩小权重"做初始化 | 3.198 | 梯度范数冲到 800 |
| base + fast-sigmoid | 2.969 | 各层发放率恢复到 2%–16% |
| **base + 硬复位** | **2.819** | 梯度稳定，所有层都在工作 |
| 硬复位 + fast-sigmoid | 2.779 | |
| 硬复位 + 复位梯度不截断 | 2.736 | |
| 硬复位 + lr 4e-3 | 2.703 | |
| 硬复位 + fast-sigmoid + 不截断 + lr 4e-3 | 2.721 | 梯度最稳（0.38），各层发放率约 3% |
| *参照：同规模 Transformer（12.4M，5 层，lr 2e-3）* | *2.707* | |

几点解读：
1. **硬复位是最关键的一项**。发放即清零，等于让神经元"由输入决定何时遗忘"（类似 Mamba 这类模型的选择性），同时把反向传播在发放时刻截断，梯度不会沿长时间链条累积。
2. **阈值法初始化**：慢神经元（β→1）会累积很大的膜电位。与其缩小它们的输入权重，不如把阈值 θ 调高到 `1/sqrt(1-β²)`。两种做法的前向计算完全等价，但后一种让 Adam 对所有权重的相对更新幅度一致，训练初期更平稳。
3. 同样 330 万 token 时，纯 LIF 网络和同规模的 Transformer 基本持平。这只是训练早期的单次对比，Transformer 的超参数也没有专门调。

## 4. 正式训练结果

配置：8 层 × 1536 LIF（共 12,288 个神经元，12.62M 参数），硬复位 + fast-sigmoid 代理梯度 + 复位梯度不截断，
batch 32 × 256，14,000 步 = 1.15 亿 token（约占训练集的 1/5，每条数据最多见一次），学习率峰值 3e-3、余弦衰减，
4 核 CPU 训练 2 小时 45 分钟（约 11,500 token/s）。

| 指标（完整验证集，557 万 token） | 数值 |
|---|---|
| 验证 loss | **1.912 nats/token**（困惑度 6.76） |
| 每字符比特数 | **0.696 bits/char**（与分词方式无关，可横向比较） |
| 平均发放率 | 训练末期约 4%（每个 token 约 500 个神经元发放，共 12,288 个） |
| 流式生成速度 | 约 610 token/s（batch 1，单 CPU；每步只携带膜电位，代价不随上下文变长而增加） |

验证 loss 随训练的变化（每 1000 步）：2.71 → 2.55 → 2.46 → 2.39 → 2.31 → 2.27 → 2.20 → 2.15 → 2.09 → 2.04 → 1.99 → 1.95 → 1.93 → 1.91。
训练结束时曲线还在下降，没有饱和，继续训练应该还能变好。

**样例**（温度 0.8，top-k 50；前缀部分为人工给定）：

> **Once upon a time**, there was a little girl named Mia. She had a pet dog named Max. ... One sunny day, Mia and Max went to the park. ...
> In the end, they were not grumpy anymore. They both had a great time and were very happy.

> **Lily wanted to go to the park, but** she did not know how. "Look, Ben, I found a big box!" Ben said. Lily opened the box and found many colorful balls. ...
> "Help, help, dog!" Lily shouted. "We can't get out!" They pulled and pulled, but the dog did not come.

> **One day, a little bird could not fly. It** was so very high. The bird was sad. ... A girl named Lily wanted to help the bird. She had an idea. ...
> The bird and Lily became good friends. They sang together every day.

完整样例见 `runs/main/final_eval.json`。

**观察到的现象**
1. **快慢神经元自发分化**（`figs/time_constants.png`）：时间常数初始化为 0.5–200 token 之间均匀分布，训练后变成双峰，一群约 1.5–3 token（处理当前词），一群约 100–300 token（保持上下文）。第 1 层分化最明显，少数神经元的时间常数达到上千个 token。
2. **稀疏发放**（`figs/raster.png`）：各层发放率约 4%–6%。故事开头（"Once upon"）会出现一次大范围的同步发放，之后转为稀疏活动。
3. **主要弱点：精确复述上文中的罕见词**。人名前后不一致（"Tom and Tom"、he/she 混用），给一个罕见名字（如 "Bolt"）它很快就记丢了。Transformer 能用注意力直接回看原文，这个网络却必须把身份信息压进膜电位再还原出来，对 1 比特/步的 LIF 群体来说很难。偶尔还会陷入重复（"You are mean. You are rude..."）。

**和 Transformer 对照组的比较**：目前只在 330 万 token 的短实验中做过（两者都约 2.70）。
长时间训练下两者的差距还需要用相同的训练预算再跑一次 Transformer 才能下结论（约 3 小时）。

## 5. 文件

```
lif_kernel.cpp   LIF 前向/反向 C++ 内核
lif.py           autograd 封装 + 纯 PyTorch 参考实现
test_lif.py      内核正确性与速度测试
model.py         纯 LIF 语言模型
model_tf.py      Transformer 对照组（仅用于比较）
prepare_data.py  训练分词器 + 语料转 token
train.py         训练（--arch snn|tf）
sample.py        文本补全（-i 交互模式）
evaluate.py      完整验证集评估 + 批量样例
analyze.py       loss 曲线、脉冲发放图、时间常数分布
figs/            上述三张图
runs/main/       正式训练：log.jsonl（全过程日志）、final_eval.json、model.pt（只含权重，48MB）
                 （含优化器状态的 ckpt.pt 有 145MB，超过 GitHub 单文件上限，没有放进仓库）
runs/abl_*/      消融实验
```

直接试用训练好的模型：
```bash
python3 sample.py --ckpt runs/main/model.pt --prompt "Once upon a time" --num 3
python3 sample.py --ckpt runs/main/model.pt -i
```

分词器 `data/tok4096.json` 已在仓库里，只用 `model.pt` 生成文本时不需要下载语料。

从头复现（需要 g++ 和 OpenMP，LIF 内核会在第一次 import 时自动编译）：
```bash
pip install -r requirements.txt
mkdir -p data
curl -L -o data/TinyStoriesV2-GPT4-train.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl -L -o data/TinyStoriesV2-GPT4-valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
python3 prepare_data.py
python3 train.py --out runs/main --steps 14000 --batch 32 --lr 3e-3 --warmup 300 \
    --hard_reset true --sg fast_sigmoid --detach_reset false
python3 sample.py --ckpt runs/main/ckpt.pt --prompt "Once upon a time" --num 3
```
