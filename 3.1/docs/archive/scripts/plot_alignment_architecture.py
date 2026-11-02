"""画出 ViT-B/16 + LLaVA 双路检索模型，和旧的生成式 SFT 图分开放。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
INK = "#19324A"
MUTED = "#5C7183"
FROZEN = "#E8F0F8"
FROZEN_EDGE = "#7295B6"
TRAINABLE = "#E4F3EF"
TRAINABLE_EDGE = "#258875"
DATA = "#F5F3EF"
DATA_EDGE = "#B9AB96"
LOSS = "#FFF3E4"
LOSS_EDGE = "#CE9958"


def rounded_box(fig, x, y, width, height, face, edge, *, radius=.012, linewidth=1.4):
    fig.add_artist(FancyBboxPatch(
        (x, y), width, height, transform=fig.transFigure,
        boxstyle=f"round,pad=0.005,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=linewidth, zorder=2,
    ))


def node(fig, x, y, width, height, title, body, *, face=TRAINABLE,
         edge=TRAINABLE_EDGE, title_size=12, body_size=11):
    rounded_box(fig, x, y, width, height, face, edge)
    fig.text(x + width / 2, y + height * .76, title, ha="center", va="center",
             color=INK, fontsize=title_size, weight="bold", zorder=3)
    fig.text(x + width / 2, y + height * .35, body, ha="center", va="center",
             color=MUTED, fontsize=body_size, linespacing=1.6, zorder=3)


def arrow(fig, start, end, *, color=MUTED, width=1.6):
    fig.add_artist(FancyArrowPatch(
        start, end, transform=fig.transFigure, arrowstyle="-|>",
        mutation_scale=14, linewidth=width, color=color, zorder=1,
        shrinkA=2, shrinkB=3,
    ))


def poly_arrow(fig, points, *, color=MUTED):
    # 折线沿右侧走，不穿过文本分支的模块。
    xs, ys = zip(*points[:-1])
    fig.add_artist(plt.Line2D(xs, ys, transform=fig.transFigure,
                              color=color, linewidth=1.6, zorder=1))
    arrow(fig, points[-2], points[-1], color=color)


def configure_fonts():
    for candidate in (
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
    ):
        if candidate.is_file():
            font_manager.fontManager.addfont(str(candidate))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
            break
    plt.rcParams.update({
        "font.size": 11, "text.color": INK,
        "mathtext.fontset": "dejavusans", "svg.fonttype": "path",
    })


def main():
    configure_fonts()
    fig = plt.figure(figsize=(19, 11.5), facecolor="white")
    fig.text(.04, .95, "ViT-B/16 + LLaVA-1.5：图文双路对齐", fontsize=27,
             weight="bold", color=INK)
    fig.text(.04, .909, "共享 512 维检索空间  ·  配对图文对比学习  ·  道德标签约束",
             fontsize=14, color=MUTED)

    fig.text(.04, .857, "图像分支", fontsize=14, weight="bold")
    fig.text(.04, .641, "文本分支", fontsize=14, weight="bold")
    for x, label, face, edge in (
        (.715, "冻结参数", FROZEN, FROZEN_EDGE),
        (.835, "可训练参数", TRAINABLE, TRAINABLE_EDGE),
    ):
        rounded_box(fig, x, .849, .014, .015, face, edge, radius=.003, linewidth=1)
        fig.text(x + .023, .857, label, fontsize=10.5, va="center", color=MUTED)

    image_y, text_y, height = .69, .475, .135
    node(fig, .04, image_y, .08, height, r"图像 $I_i$", "RGB", face=DATA,
         edge=DATA_EDGE, title_size=13)
    node(fig, .16, image_y, .195, height, "OpenAI CLIP ViT-B/16",
         "普通 CLIP 预训练 · 视觉塔冻结\n224 × 224 → 196 × 768 patch 特征",
         face=FROZEN, edge=FROZEN_EDGE, title_size=12, body_size=10.5)
    node(fig, .395, image_y, .16, height, "新建多模态 Projector",
         "Linear → GELU → Linear\n768 → 4096 → 4096", title_size=12, body_size=11)
    node(fig, .595, image_y, .105, height, "Patch mean",
         "196 个 patch 取均值\n得到 4096 维", face=DATA, edge=DATA_EDGE,
         title_size=12, body_size=10.5)
    node(fig, .735, image_y, .12, height, "Image head", "4096 → 512\n线性投影 · 可训练")
    node(fig, .89, image_y, .065, height, "L2", r"$u_i$" + "\n单位向量",
         face=DATA, edge=DATA_EDGE, title_size=13, body_size=10.5)

    node(fig, .04, text_y, .08, height, r"Caption $T_i$", "图像配对描述",
         face=DATA, edge=DATA_EDGE, title_size=11.5, body_size=10)
    rounded_box(fig, .16, text_y, .395, height, FROZEN, FROZEN_EDGE)
    fig.text(.3575, .582, "Vicuna-7B decoder（LLaVA-1.5 内部语言模型）",
             ha="center", va="center", fontsize=13, weight="bold", zorder=3)
    fig.text(.3575, .547, "基础 LM 权重冻结  ·  纯文本前向  ·  隐藏维度 4096",
             ha="center", va="center", fontsize=11, color=MUTED, zorder=3)
    rounded_box(fig, .18, .482, .355, .04, TRAINABLE, TRAINABLE_EDGE, radius=.006, linewidth=1)
    fig.text(.3575, .502, "训练 LoRA：attention / MLP 线性层  ·  r = 16，α = 32",
             ha="center", va="center", fontsize=10.5, color=TRAINABLE_EDGE, zorder=3)
    node(fig, .595, text_y, .105, height, "Masked mean", "忽略 padding 位置\n得到 4096 维",
         face=DATA, edge=DATA_EDGE, title_size=11.5, body_size=10.5)
    node(fig, .735, text_y, .12, height, "Text head", "4096 → 512\n线性投影 · 可训练")
    node(fig, .89, text_y, .065, height, "L2", r"$v_j$" + "\n单位向量",
         face=DATA, edge=DATA_EDGE, title_size=13, body_size=10.5)

    for y in (image_y + height / 2, text_y + height / 2):
        segments = ((.12, .16), (.555, .595), (.70, .735), (.855, .89))
        for left, right in segments:
            arrow(fig, (left, y), (right, y))
    arrow(fig, (.355, image_y + height / 2), (.395, image_y + height / 2))

    # 两条支路只在相似度矩阵这里汇合，图像特征不送进 Vicuna。
    node(fig, .70, .32, .255, .10, "Batch 图文相似度矩阵 S",
         r"$s_{ij} = u_i^{\mathsf{T}}v_j\,/\,\tau,\quad \tau = 0.07$" + "（固定）",
         face=DATA, edge=DATA_EDGE, title_size=12.5, body_size=12)
    poly_arrow(fig, ((.955, .7575), (.98, .7575), (.98, .37), (.955, .37)))
    arrow(fig, (.9225, .475), (.9225, .42))
    fig.text(.981, .64, r"$u_i$", fontsize=13, ha="center", va="center",
             bbox=dict(facecolor="white", edgecolor="none", pad=2))
    fig.text(.937, .448, r"$v_j$", fontsize=13, ha="center", va="center")

    fig.text(.04, .397, "同一 batch：B 张图像与 B 条配对 caption；每图每轮抽取一条描述。",
             fontsize=12, color=INK)
    fig.text(.04, .364, "11 维道德标签：10 种 virtue / vice + non-moral，只用于损失计算。",
             fontsize=11.5, color=MUTED)
    fig.text(.04, .331,
             "语言侧 LoRA 只训练 caption 特征对齐，不训练答案生成；模型输出图文相似度与排序，不输出五维 JSON。",
             fontsize=11.5, color=MUTED)

    rounded_box(fig, .04, .155, .415, .11, LOSS, LOSS_EDGE)
    fig.text(.2475, .242, "配对图文损失：双向交叉熵", ha="center", va="center",
             fontsize=13, weight="bold", zorder=3)
    fig.text(.2475, .207,
             r"$\mathcal{L}_{\rm CLIP}=\frac{1}{2}\,[\mathrm{CE}(S,\mathrm{diag})+\mathrm{CE}(S^{\mathsf{T}},\mathrm{diag})]$",
             ha="center", va="center", fontsize=14, zorder=3)
    fig.text(.2475, .173, "对角配对为正样本，batch 内其余配对为负样本",
             ha="center", va="center", fontsize=10.5, color=MUTED, zorder=3)

    rounded_box(fig, .505, .155, .45, .11, LOSS, LOSS_EDGE)
    fig.text(.73, .242, "道德约束：标签 Jaccard 与非对角 MSE", ha="center", va="center",
             fontsize=13, weight="bold", zorder=3)
    fig.text(.73, .209,
             r"$g_{ij}=2\,\frac{|M_i\,\cap\,M_j|}{|M_i\,\cup\,M_j|}-1,\qquad"
             r"\mathcal{L}_{\rm moral}=\frac{1}{B(B-1)}\sum_{i\ne j}(s_{ij}-g_{ij})^2$",
             ha="center", va="center", fontsize=12.7, zorder=3)
    fig.text(.73, .173, "使用缩放后的相似度 S；排除对角项；M 为该图像的正标签集合",
             ha="center", va="center", fontsize=10.2, color=MUTED, zorder=3)
    arrow(fig, (.735, .32), (.425, .265), color=LOSS_EDGE)
    arrow(fig, (.86, .32), (.86, .265), color=LOSS_EDGE)

    rounded_box(fig, .19, .064, .665, .045, "#EFF5F3", TRAINABLE_EDGE, radius=.008)
    fig.text(.405, .0865,
             r"$\mathcal{L}=\mathcal{L}_{\rm CLIP}+\lambda\mathcal{L}_{\rm moral},\quad \lambda=0.4$",
             fontsize=16, ha="center", va="center", weight="bold", zorder=3)
    fig.text(.695, .0865, "对照组：λ = 0，其余设置相同", fontsize=11.5,
             ha="center", va="center", color=MUTED, zorder=3)
    arrow(fig, (.2475, .155), (.37, .109), color=TRAINABLE_EDGE)
    arrow(fig, (.73, .155), (.67, .109), color=TRAINABLE_EDGE)

    fig.text(.04, .023,
             "方法范围：基于 LLaVA 的双路检索适配，并非原始 MoralCLIP 严格复现。训练 caption 编码 LoRA、Projector 和两侧投影头；不训练图像问答生成目标。",
             fontsize=10.5, color=MUTED)
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        output = output_dir / f"alignment_architecture.{suffix}"
        fig.savefig(output, dpi=220, facecolor="white")
        print(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
