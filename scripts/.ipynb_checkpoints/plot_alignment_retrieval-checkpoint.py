"""Render a saved, formal alignment retrieval result without loading a model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence


INK = "#20364B"
MUTED = "#617589"
ACCENT = "#098B7E"
BACKGROUND = "#F3F7FA"


def read_result(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("training_run_kind") != "training":
        raise ValueError("Expected retrieve_alignment.py output from a formal training checkpoint")
    count = record.get("candidate_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("candidate_count must be a positive integer")
    results = record.get("results")
    if not isinstance(results, list) or not results or len(results) > count:
        raise ValueError("The result must contain ranked candidate descriptions")
    for rank, row in enumerate(results, 1):
        if not isinstance(row, dict) or row.get("rank") != rank:
            raise ValueError("Results must be ordered with consecutive ranks starting at 1")
        if not isinstance(row.get("caption"), str) or not row["caption"].strip():
            raise ValueError(f"Missing caption at rank {rank}")
        score = row.get("cosine")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"Invalid cosine at rank {rank}")
        if rank > 1 and score > results[rank - 2]["cosine"] + 1e-6:
            raise ValueError("Cosine values must be in descending order")
    return record, results[:5]


def configure_font(font_path: Path | None = None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    candidates = [font_path] if font_path else []
    candidates.extend([
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
    ])
    for candidate in candidates:
        if candidate and candidate.expanduser().is_file():
            candidate = candidate.expanduser()
            font_manager.fontManager.addfont(str(candidate))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return
    raise ValueError("No Chinese font found; pass --font /path/to/a/CJK/font.ttf")


def wrap_caption(caption: str) -> list[str]:
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextToPath

    # 候选描述整句保留，长一点就多占几行，不在图里偷偷截掉。
    font = FontProperties(size=11.2)
    text_path = TextToPath()
    width_limit = 8.85 * 72

    def fits(text: str) -> bool:
        return text_path.get_text_width_height_descent(text, font, ismath=False)[0] <= width_limit

    lines: list[str] = []
    current = ""
    for word in caption.split():
        combined = f"{current} {word}" if current else word
        if fits(combined):
            current = combined
            continue
        if current:
            lines.append(current)
            current = ""
        if fits(word):
            current = word
            continue
        # 极长的单词或网址也能换行，避免伸到画布外面。
        for char in word:
            combined = current + char
            if current and not fits(combined):
                lines.append(current)
                current = char
            else:
                current = combined
    if current:
        lines.append(current)
    return lines


def render_result(record: dict[str, Any], rows: list[dict[str, Any]], image_path: Path,
                  output: Path, *, font_path: Path | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from PIL import Image, ImageOps

    configure_font(font_path)
    caption_lines = [wrap_caption(row["caption"]) for row in rows]
    line_height = .235
    row_heights = [.64 + len(lines) * line_height for lines in caption_lines]
    content_height = max(6.5, sum(row_heights) + .14 * (len(rows) - 1))
    figure_height = content_height + 3.15
    figure_width = 16.4
    fig = plt.figure(figsize=(figure_width, figure_height), facecolor=BACKGROUND)

    def label(x: float, top: float, text: str, **kwargs: Any) -> None:
        fig.text(x / figure_width, 1 - top / figure_height, text,
                 va="top", color=kwargs.pop("color", INK), parse_math=False, **kwargs)

    label(.6, .38, "LLaVA 图文检索：单图测试", fontsize=24, weight="bold")
    label(.6, .94,
          f"候选数量：{record['candidate_count']:,} 条  ·  展示前 {len(rows)} 条  ·  按余弦相似度排序",
          fontsize=12, color=MUTED)
    label(.6, 1.47, "输入图片", fontsize=13, weight="bold")
    label(6.45, 1.47, "检索到的英文描述", fontsize=13, weight="bold")

    top = 1.9
    image_width, image_height = 5.45, min(content_height, 7.0)
    image_ax = fig.add_axes([.6 / figure_width,
                            1 - (top + image_height) / figure_height,
                            image_width / figure_width, image_height / figure_height])
    # 保留原图比例，手机照片的旋转信息也顺手处理下。
    with Image.open(image_path) as image:
        original = ImageOps.exif_transpose(image).convert("RGB")
        image_ax.imshow(original)
    image_ax.set_axis_off()
    image_ax.set_facecolor("white")
    label(.6, top + image_height + .18, "原图完整显示，未按模型输入尺寸裁切。",
          fontsize=10, color=MUTED)

    cursor = top
    for row, lines, height in zip(rows, caption_lines, row_heights):
        fig.add_artist(FancyBboxPatch(
            (6.4 / figure_width, 1 - (cursor + height) / figure_height),
            9.4 / figure_width, height / figure_height,
            transform=fig.transFigure,
            boxstyle="round,pad=0.003,rounding_size=0.008",
            facecolor="white", edgecolor="#E0E8EF", linewidth=.8,
        ))
        label(6.63, cursor + .16, f"TOP {row['rank']}", fontsize=11,
              color=ACCENT, weight="bold")
        label(13.5, cursor + .16, f"cosine  {row['cosine']:.4f}",
              fontsize=11, color=ACCENT, weight="bold")
        label(6.63, cursor + .51, "\n".join(lines), fontsize=11.2, linespacing=1.51)
        cursor += height + .14

    footer = figure_height - .63
    label(.6, footer, "cosine 是余弦相似度，不是概率；这里只对候选描述排序，不生成自由回答。",
          fontsize=11, color=MUTED)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output, dpi=180, facecolor=BACKGROUND)
    finally:
        plt.close(fig)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True,
                        help="Formal JSON written by retrieve_alignment.py --output")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--image", type=Path,
                        help="Local image override, useful when JSON contains a server path")
    parser.add_argument("--font", type=Path, help="Optional Chinese font path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        result_path = args.result.expanduser().resolve()
        record, rows = read_result(result_path)
        image_path = args.image or Path(record["image"])
        image_path = image_path.expanduser()
        if not image_path.is_absolute() and not args.image:
            image_path = result_path.parent / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise ValueError(f"Input image missing: {image_path}; use --image to set its local path")
        output = args.output.expanduser().resolve()
        if output.suffix.lower() != ".png":
            raise ValueError("--output must end in .png")
        render_result(record, rows, image_path, output, font_path=args.font)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Cannot render retrieval result: {error}\n")
    except ModuleNotFoundError as error:
        parser.exit(2, f"Missing dependency: {error.name}; install matplotlib and Pillow in the plotting environment\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
