"""Figures from the committed memory maps (5.63, runs/memory_maps/*.npz): the read at the answer position, and one
whole map. Run from the repository root: python3 experiments/plot_memory_maps.py (writes runs/memory_maps/*.png)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

CJK = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")   # any font with Chinese glyphs will do
if CJK.exists():
    font_manager.fontManager.addfont(str(CJK))
    plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

repo = Path(__file__).resolve().parents[1]
out = repo / "runs/memory_maps"
summary = json.loads((repo / "runs/memory_maps/summary.json").read_text())
MODELS = [("g13_tf163", "Transformer（老师）"), ("g38_la", "稠密线性注意力（对照）"),
          ("g39_snngla", "脉冲网络 + 线性注意力记忆"), ("g44_all3", "脉冲网络 + 可塑突触记忆")]
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
ACCENT, BAR = "#cf222e", "#8c959f"


def head_of(name):
    layer, head = summary["models"][name]["intervention"]["heads"][0]
    return layer, head


def normalized(matrix):
    mass = np.abs(np.tril(matrix)).sum(-1, keepdims=True)
    return np.where(mass > 0, np.tril(matrix) / np.where(mass > 0, mass, 1), 0.)


def word(tok):
    tok = tok.replace("Ġ", " ").replace("Ċ", "\\n").strip() or "␣"
    return "开头" if tok == "<|endoftext|>" else tok


# Figure 1: the query row (the moment the model must produce the answer).
cases = [(0, "例一：“……with Jenna,’ said ___”，答案 Petr"), (1, "例二：答案 Joey")]
fig, axes = plt.subplots(len(MODELS), len(cases), figsize=(14, 10), sharex="col")
for col, (case, title) in enumerate(cases):
    for row, (name, label) in enumerate(MODELS):
        z = np.load(repo / f"runs/memory_maps/{name}.npz")
        layer, head = head_of(name)
        query, source = z[f"case_{case}_positions"][:2]
        text = z[f"case_{case}_text"]
        share = normalized(z[f"case_{case}_layer_{layer}_head_{head}"])[query, :query + 1]
        ax = axes[row, col]
        pos = np.arange(len(share))
        colors = [ACCENT if p == source else BAR for p in pos]
        ax.bar(pos, share, width=0.9, color=colors, linewidth=0)
        ax.axhline(0, color=GRID, linewidth=0.8)
        top = max(share.max(), 0.05)
        ax.set_ylim(min(share.min(), 0) * 1.15, top * 1.45)
        chosen = [source]
        for p in np.argsort(-np.abs(share)):
            if len(chosen) == 3:
                break
            if all(abs(p - c) > len(share) // 12 for c in chosen):
                chosen.append(p)
        for p in chosen:
            ax.annotate(f"{word(text[p])}\n{share[p]:.0%}", (p, max(share[p], 0)), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_ylabel(f"{label}\n第{layer + 1}层第{head + 1}个头", fontsize=9, color=INK, rotation=0,
                      ha="right", va="center")
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        if row == 0:
            ax.set_title(title, fontsize=11, color=INK, loc="left")
        if row == len(MODELS) - 1:
            ax.set_xlabel("前文中的位置（第几个词）", fontsize=9, color=MUTED)
fig.suptitle("答题那一刻，每个模型“往回看”的分量分布在哪里（每行加起来按绝对值是 100%）",
             fontsize=13, color=INK, x=0.01, ha="left")
fig.text(0.01, 0.005, "红色：答案在前文出现的位置；灰色：其他位置；柱子上标的是正确的词和另外读得最多、彼此不挨着的两个词。每个模型画的是它在开发段落上最常看向正确位置的那个记忆头（两个例子都是没参与选头的段落）。",
         fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 0.96))
fig.savefig(out / "memory_query_rows.png", dpi=150)
plt.close(fig)

# Figure 2: the whole map, as an attention matrix.
case = 0
fig, axes = plt.subplots(1, len(MODELS), figsize=(16, 4.8))
cap = 0.3
for ax, (name, label) in zip(axes, MODELS):
    z = np.load(repo / f"runs/memory_maps/{name}.npz")
    layer, head = head_of(name)
    query, source = z[f"case_{case}_positions"][:2]
    share = normalized(z[f"case_{case}_layer_{layer}_head_{head}"])
    image = ax.imshow(np.clip(share, -cap, cap), cmap="RdBu_r", vmin=-cap, vmax=cap, interpolation="nearest")
    ax.scatter([source], [query], s=60, facecolors="none", edgecolors=INK, linewidths=1.2)
    ax.set_title(f"{label}\n第{layer + 1}层第{head + 1}个头", fontsize=10, color=INK)
    ax.set_xlabel("被读的位置（写入）", fontsize=9, color=MUTED)
    ax.set_ylabel("正在读的位置", fontsize=9, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
cbar = fig.colorbar(image, ax=axes, fraction=0.012, pad=0.01)
cbar.set_label("分量（每行按绝对值归一；超过 30% 的按 30% 画）", fontsize=9, color=MUTED)
cbar.ax.tick_params(labelsize=8, colors=MUTED)
fig.suptitle("例一的整张记忆图：第 i 行第 j 列 = 读到第 i 个词时，第 j 个词的写入占多少。黑圈：答题时刻对正确的词",
             fontsize=12, color=INK, x=0.01, ha="left")
fig.savefig(out / "memory_maps_matrix.png", dpi=150, bbox_inches="tight")
print("ok")
