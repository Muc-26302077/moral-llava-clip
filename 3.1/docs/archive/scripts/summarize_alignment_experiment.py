"""Summarize completed alignment ablations; never substitute smoke or missing results.

Only the figure needs matplotlib. The report and comparison JSON use the
standard library. Run from the project root after copying both complete run
records and their independent test evaluations back from the model host.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
MATCHED_CONFIG_KEYS = (
    "base_model", "revision", "vision_model", "vision_revision", "projection_dim",
    "freeze_projector", "lora_r", "lora_alpha", "lora_dropout", "epochs", "batch_size",
    "gradient_accumulation", "learning_rate", "weight_decay", "warmup_ratio",
    "max_grad_norm", "moral_similarity", "temperature", "train_temperature",
    "text_max_length", "seed", "precision", "no_gradient_checkpointing", "selection_metric",
    "vision_initialization_source",
)
METRIC_FIELDS = {
    "paired_r_at_1": ("same_image_pairing", "r_at_1"),
    "paired_r_at_5": ("same_image_pairing", "r_at_5"),
    "paired_r_at_10": ("same_image_pairing", "r_at_10"),
    "paired_map_i2t": ("same_image_pairing", "map_i2t"),
    "moral_hit_at_1": ("label_relevance", "hit_at_1"),
    "moral_hit_at_5": ("label_relevance", "hit_at_5"),
    "moral_hit_at_10": ("label_relevance", "hit_at_10"),
    "moral_map_i2t": ("label_relevance", "map_i2t"),
    "cross_image_moral_hit_at_1": ("cross_image_moral_relevance", "hit_at_1"),
    "cross_image_moral_hit_at_5": ("cross_image_moral_relevance", "hit_at_5"),
    "cross_image_moral_hit_at_10": ("cross_image_moral_relevance", "hit_at_10"),
    "cross_image_moral_map_i2t": ("cross_image_moral_relevance", "map_i2t"),
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fraction(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid [0,1] metric {label}: {value!r}")
    return float(value)


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    values = {}
    for name, (section, field) in METRIC_FIELDS.items():
        # 旧别名只在读入时兼容，报告里仍把共享标签的命中率写成 Hit。
        source = metrics[section]
        raw = source.get(field, source.get(field.replace("hit_at_", "r_at_")))
        values[name] = fraction(raw, f"{section}.{field}")
    return values


def validate_vision_source(config: dict[str, Any], *, label: str) -> None:
    vision_model = config.get("vision_model")
    if not vision_model:
        return
    reference = str(vision_model)
    path = Path(reference).expanduser()
    is_local = path.is_absolute() or reference.startswith((".", "~")) or path.is_dir()
    source = config.get("vision_initialization_source")
    if source is None and not is_local:
        return
    # 本地目录名相同并不能说明权重相同，实际来源得看 sidecar 里的指纹。
    if not isinstance(source, dict) or source.get("status") != "PASS":
        raise ValueError(f"{label}: local vision initialization needs a PASS source record")
    for field in ("repository", "revision"):
        if not isinstance(source.get(field), str) or not source[field].strip():
            raise ValueError(f"{label}: vision initialization source is missing {field}")
    files = source.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"{label}: vision initialization source is missing file hashes")
    if not any(isinstance(name, str) and name.endswith((".bin", ".safetensors")) for name in files):
        raise ValueError(f"{label}: vision initialization source has no weight-file hash")
    for name, record in files.items():
        digest = record.get("sha256") if isinstance(record, dict) else None
        if (not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdefABCDEF" for character in digest)):
            raise ValueError(f"{label}: invalid SHA-256 for vision initialization file {name!r}")


def load_run(run_path: Path, test_path: Path, *, key: str) -> dict[str, Any]:
    run_path, test_path = run_path.expanduser().resolve(), test_path.expanduser().resolve()
    config = read_json(run_path / "run_config.json")
    validate_vision_source(config, label=key)
    summary = read_json(run_path / "summary.json")
    best = read_json(run_path / "best_checkpoint.json")
    test = read_json(test_path)
    if config.get("run_kind") != "training" or summary.get("run_kind") != "training":
        raise ValueError(f"{key}: smoke/partial records cannot be reported as formal results")
    if config.get("smoke_test") or config.get("max_steps", 0):
        raise ValueError(f"{key}: a step-limited or smoke run is not a complete experiment")
    if summary.get("status") != "PASS" or test.get("status") != "PASS":
        raise ValueError(f"{key}: training and independent test must both finish with PASS")
    # summary.test_evaluated=False 是训练端没使用测试集选权重，并不是独立测试失败。
    if test.get("original_split") != "test" or test.get("metric_direction") != "image_to_text":
        raise ValueError(f"{key}: expected an independent image-to-text test evaluation")
    if test.get("model_type") != "LLaVA learned image-text alignment checkpoint":
        raise ValueError(f"{key}: the test file must evaluate the learned LLaVA alignment checkpoint")
    if not test.get("checkpoint_sha256"):
        raise ValueError(f"{key}: missing evaluated checkpoint hashes")
    if test.get("manifest_sha256") != config.get("manifest_sha256"):
        raise ValueError(f"{key}: training and test manifest fingerprints disagree")
    if config.get("test_used_for_selection") is not False:
        raise ValueError(f"{key}: training must explicitly exclude test-based checkpoint selection")
    if any(config.get(f"used_{split}_images") != config["original_split_sizes"][split]
           for split in ("train", "validation")):
        raise ValueError(f"{key}: formal training/validation does not cover the complete original splits")
    if any(config.get("text_token_lengths", {}).get(split, {}).get("truncated") != 0
           for split in ("train", "validation")):
        raise ValueError(f"{key}: training/validation text was truncated or truncation is undocumented")
    query_ids, candidate_ids = test["query_ids"], test["candidate_ids"]
    candidate_image_ids = test["candidate_image_ids"]
    if (
        len(query_ids) != len(set(query_ids)) or len(candidate_ids) != len(set(candidate_ids))
        or len(query_ids) != test["num_queries"] or len(candidate_ids) != test["num_candidates"]
        or len(candidate_image_ids) != len(candidate_ids)
        or test["metrics"]["num_queries"] != len(query_ids)
        or test["metrics"]["num_candidates"] != len(candidate_ids)
        or set(candidate_image_ids) != set(query_ids)
    ):
        raise ValueError(f"{key}: invalid query/candidate IDs or counts")
    if len(query_ids) != config["original_split_sizes"]["test"]:
        raise ValueError(f"{key}: independent test does not cover the complete original test split")
    if test.get("text_token_lengths", {}).get("truncated") != 0:
        raise ValueError(f"{key}: the test candidate text was truncated or truncation is undocumented")

    validation_paths = sorted(run_path.glob("validation_epoch_*.json"))
    validation = [read_json(path) for path in validation_paths]
    epochs = int(config["epochs"])
    if [row.get("epoch") for row in validation] != list(range(1, epochs + 1)):
        raise ValueError(f"{key}: incomplete or duplicated epoch validation records")
    for row in validation:
        if row.get("split") != "validation" or row.get("run_kind") != "training":
            raise ValueError(f"{key}: checkpoint selection must use formal validation records")
        if row.get("selection_metric") != config["selection_metric"]:
            raise ValueError(f"{key}: selection metric changes across epochs")
        if row["metrics"]["num_queries"] != config["used_validation_images"]:
            raise ValueError(f"{key}: validation image coverage changes across epochs")
        chosen_metric = (row["metrics"]["label_relevance"]["map_i2t"]
                         if config["selection_metric"] == "moral_map"
                         else row["metrics"]["same_image_pairing"]["r_at_1"])
        if not math.isclose(chosen_metric, row["selection_value"], rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"{key}: validation selection value does not match its metrics")
    selected = max(validation, key=lambda row: row["selection_value"])
    selected_name = f"epoch-{selected['epoch']:03d}"
    expected_checkpoint_names = [Path(str(row["checkpoint"])).name for row in (best, test)]
    expected_checkpoint_names.append(Path(str(summary["best_checkpoint"])).name)
    if any(name != selected_name for name in expected_checkpoint_names):
        raise ValueError(f"{key}: independent test does not use the validation-selected checkpoint {selected_name}")
    if not math.isclose(best["selection_value"], selected["selection_value"], rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"{key}: best-checkpoint pointer disagrees with validation records")
    saved = test["checkpoint_metadata"]
    saved_metadata = saved.get("metadata", {})
    saved_config = saved_metadata.get("run_config", {})
    validate_vision_source(saved_config, label=f"{key} evaluated checkpoint")
    if saved_metadata.get("validation", {}).get("epoch") != selected["epoch"]:
        raise ValueError(f"{key}: evaluated checkpoint metadata has a different epoch")
    for name in (*MATCHED_CONFIG_KEYS, "moral_weight", "manifest_sha256"):
        if saved_config.get(name) != config.get(name):
            raise ValueError(f"{key}: evaluated checkpoint does not match run configuration: {name}")

    log_path = run_path / "train_log.jsonl"
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not events or [row["optimizer_step"] for row in events] != list(range(1, summary["optimizer_steps"] + 1)):
        raise ValueError(f"{key}: missing or duplicated optimizer-step logs")
    if validation[-1]["optimizer_step"] != summary["optimizer_steps"]:
        raise ValueError(f"{key}: training summary and final validation step disagree")
    expected_steps = epochs * math.ceil(
        (config["used_train_images"] // config["batch_size"]) / config["gradient_accumulation"]
    )
    if summary["optimizer_steps"] != expected_steps:
        raise ValueError(f"{key}: run did not complete the configured number of optimizer steps")
    history = []
    for row in validation:
        flat = flatten_metrics(row["metrics"])
        losses = row["mean_train_losses"]
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in losses.values()):
            raise ValueError(f"{key}: nonfinite epoch losses")
        history.append({"epoch": row["epoch"], "optimizer_step": row["optimizer_step"],
                        "selection_value": row["selection_value"], "metrics": flat,
                        "mean_train_losses": losses})
    start = datetime.fromisoformat(config["started_at"])
    finish = datetime.fromisoformat(summary["finished_at"])
    sources = [run_path / "run_config.json", run_path / "summary.json", run_path / "best_checkpoint.json",
               log_path, *validation_paths, test_path]
    return {
        "key": key, "run_directory": str(run_path), "test_file": str(test_path),
        "config": config, "training_summary": summary,
        "best_epoch": selected["epoch"], "best_checkpoint": test["checkpoint"],
        "best_validation_value": selected["selection_value"],
        "selection_metric": config["selection_metric"],
        "moral_weight": config["moral_weight"],
        "training_wall_seconds": (finish - start).total_seconds(),
        "test_metrics": flatten_metrics(test["metrics"]),
        "cross_image_query_coverage": {
            name: test["metrics"]["cross_image_moral_relevance"][name]
            for name in ("num_queries", "num_queries_with_relevant_candidates", "num_queries_without_relevant_candidates")
        },
        "validation_history": history,
        "source_sha256": {str(path): file_sha256(path) for path in sources},
        "evaluated_checkpoint_sha256": test["checkpoint_sha256"],
        "test_protocol": {
            name: test[name] for name in (
                "original_split", "metric_direction", "num_queries", "num_candidates", "query_ids",
                "candidate_ids", "candidate_image_ids", "caption_count_distribution", "manifest_sha256",
                "combined_manifest_sha256", "embedding_dimension", "precision", "text_token_lengths",
            )
        },
    }


def build_comparison(clip: dict[str, Any], moral: dict[str, Any]) -> dict[str, Any]:
    for run in (clip, moral):
        validate_vision_source(run["config"], label=run["key"])
    differing_config = {
        key: {"clip": clip["config"].get(key), "moral": moral["config"].get(key)}
        for key in MATCHED_CONFIG_KEYS if clip["config"].get(key) != moral["config"].get(key)
    }
    if differing_config:
        raise ValueError(f"The lambda ablation has other configuration differences: {differing_config}")
    if clip["moral_weight"] != 0 or moral["moral_weight"] <= 0:
        raise ValueError("Expected lambda=0 baseline and a positive-lambda moral run")
    if clip["test_protocol"] != moral["test_protocol"]:
        raise ValueError("The two test evaluations use different IDs, candidate pools, manifests or numeric precision")
    if clip["config"]["manifest_sha256"] != moral["config"]["manifest_sha256"]:
        raise ValueError("The two runs use different training/validation/test manifests")
    return {
        "status": "PASS", "created_at": datetime.now(timezone.utc).isoformat(),
        "comparison": "fixed-initialization LLaVA image-text alignment lambda ablation",
        "matched_configuration_keys": list(MATCHED_CONFIG_KEYS),
        "varied_configuration": {"moral_weight": {"clip": clip["moral_weight"], "moral": moral["moral_weight"]}},
        "runs": {"clip": clip, "moral": moral},
        "delta_moral_minus_clip_percentage_points": {
            name: 100 * (moral["test_metrics"][name] - clip["test_metrics"][name])
            for name in METRIC_FIELDS
        },
        "metric_definitions": {
            "paired_r_at_k": "fraction of queries with at least one own-image caption in the first k results",
            "moral_hit_at_k": "fraction of queries with at least one caption sharing a moral label in the first k results",
            "moral_map_i2t": "mean AP with shared-moral-label relevance, including captions from the query image",
            "cross_image_moral_map_i2t": "mean AP after removing all own-image captions; no-positive queries contribute zero",
        },
        "limitations": [
            "One seed and one original split; no multi-seed confidence interval or significance test.",
            "Retrieval metrics do not measure VQA or five-foundation generation performance.",
            "Project shared-label R@K (formerly Hit@K) is query hit rate, not the fraction of all relevant captions recalled.",
            "Validation moral_map includes own-image captions; cross-image moral MAP is a separate diagnostic.",
            "This is a LLaVA alignment adaptation, not a reproduction of the full MoralCLIP paper experiment.",
        ],
    }


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def delta_text(value: float) -> str:
    return f"{value:+.2f} 个百分点"


def render_report(comparison: dict[str, Any], figure_name: str) -> str:
    clip, moral = (comparison["runs"][name] for name in ("clip", "moral"))
    protocol, config = clip["test_protocol"], clip["config"]
    delta = comparison["delta_moral_minus_clip_percentage_points"]
    selection = "共享道德标签 MAP" if config["selection_metric"] == "moral_map" else "同图配对 R@1"
    train_batches = config["used_train_images"] // config["batch_size"]
    skipped_per_epoch = config["used_train_images"] % config["batch_size"]
    vision_source = config.get("vision_initialization_source") or {}
    vision_revision = vision_source.get("revision") or config.get("vision_revision") or "未记录"
    vision_repository = vision_source.get("repository")
    vision_origin = f"（`{vision_repository}`）" if vision_repository else ""
    lines = [
        "# LLaVA 图文对齐与道德约束实验", "",
        f"两组使用相同初始化、数据划分和训练配置，仅改变道德约束权重 λ（{clip['moral_weight']:g} → {moral['moral_weight']:g}）。"
        f"测试集包含 {protocol['num_queries']} 张图片、{protocol['num_candidates']} 条候选描述；"
        f"所有测试结果均来自验证集按{selection}选择的 checkpoint。", "",
        f"加入道德约束后，同图配对 R@1 为 {percent(moral['test_metrics']['paired_r_at_1'])}"
        f"（{delta_text(delta['paired_r_at_1'])}），共享道德标签 MAP 为 {percent(moral['test_metrics']['moral_map_i2t'])}"
        f"（{delta_text(delta['moral_map_i2t'])}），排除本图描述后的跨图道德 MAP 为 "
        f"{percent(moral['test_metrics']['cross_image_moral_map_i2t'])}（{delta_text(delta['cross_image_moral_map_i2t'])}）。", "",
        "## 测试结果", "",
        "| 方案 | λ | 最佳 epoch | 配对 R@1 | 配对 R@5 | 配对 R@10 | 道德 MAP | 跨图道德 MAP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run, name in ((clip, "CLIP式对比基线"), (moral, "MoralCLIP式约束迁移到LLaVA")):
        scores = run["test_metrics"]
        values = [percent(scores[key]) for key in ("paired_r_at_1", "paired_r_at_5", "paired_r_at_10", "moral_map_i2t", "cross_image_moral_map_i2t")]
        lines.append(f"| {name} | {run['moral_weight']:g} | {run['best_epoch']} | " + " | ".join(values) + " |")
    lines.extend(["", "| 方案 | 道德 R@1 | 道德 R@5 | 道德 R@10 | 跨图 R@1 | 跨图 R@5 | 跨图 R@10 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for run, name in ((clip, "CLIP式对比基线"), (moral, "MoralCLIP式约束迁移到LLaVA")):
        values = [percent(run["test_metrics"][key]) for key in (
            "moral_hit_at_1", "moral_hit_at_5", "moral_hit_at_10",
            "cross_image_moral_hit_at_1", "cross_image_moral_hit_at_5", "cross_image_moral_hit_at_10")]
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    lines.extend(["", f"![实验结果与训练曲线]({figure_name})", "", "## 模型与训练设置", "",
        f"- 基模：`{config['base_model']}`，revision `{config['revision']}`。",
        f"- 视觉初始化：`{config.get('vision_model') or 'LLaVA 原始视觉塔'}`{vision_origin}，revision `{vision_revision}`。",
        f"- 原始图片划分：训练 {config['original_split_sizes']['train']}、验证 {config['original_split_sizes']['validation']}、测试 {config['original_split_sizes']['test']}。",
        f"- 训练 {config['epochs']} epochs，实际对比 batch {config['batch_size']}，梯度累积 {config['gradient_accumulation']}；学习率 {config['learning_rate']:g}，seed {config['seed']}，{config['precision'].upper()}。",
        f"- LoRA rank {config['lora_r']}、alpha {config['lora_alpha']}、dropout {config['lora_dropout']:g}；共享向量维度 {protocol['embedding_dimension']}。",
        f"- 目标为双向配对 CLIP 交叉熵，加 λ 倍非对角道德相似度 MSE；道德项模式为 `{config['moral_similarity']}`。",
        f"- caption 按 epoch 编号确定性轮换，每张图每轮最多选择一条；训练使用 shuffle 与 drop_last，每轮实际 {train_batches} 批、{train_batches * config['batch_size']} 张图，略过尾批 {skipped_per_epoch} 张。计划轮换不代表每图的所有描述都已实际参与训练。",
        "- 验证与测试使用对应划分的全部 caption，不把同图描述分到不同划分。",
        "- 图像经过冻结视觉塔、视觉投影器、patch 均值池化和图像头；caption 经过 Vicuna + LoRA、mask 均值池化和文本头，使用归一化向量余弦相似度检索。",
        "", "## 指标口径", "",
        "- 配对 R@K：前 K 条结果中至少有一条属于查询图片的 caption，按图片查询数平均。",
        "- 道德相关性：候选 caption 对应图片与查询图片共享至少一个原始道德标签；11 类中包含 non-moral。道德 R@K（原 Hit@K）是查询命中率，不能解释为找回了多少比例的全部相关描述。",
        "- 道德 MAP：在全部测试 caption 中按上述标签相关性计算每个查询的平均精度，再按查询平均；它包含查询图片自身的 caption。",
        "- 跨图道德 MAP / Hit：先删除查询图片自身的全部 caption，再排序；分母仍为全部查询，没有跨图正例的查询计零。",
    ])
    for run, name in ((clip, "基线"), (moral, "道德约束")):
        coverage = run["cross_image_query_coverage"]
        lines.append(f"- {name}的跨图正例覆盖：{coverage['num_queries_with_relevant_candidates']}/{coverage['num_queries']}，无跨图正例 {coverage['num_queries_without_relevant_candidates']} 张。")
    lines.extend(["", "## checkpoint 与记录", "", "| 方案 | 最佳 epoch | 验证选择值 | 完成优化器步数 | 训练墙钟时间 |", "| --- | ---: | ---: | ---: | ---: |"])
    for run, name in ((clip, "基线"), (moral, "道德约束")):
        lines.append(f"| {name} | {run['best_epoch']} | {percent(run['best_validation_value'])} | {run['training_summary']['optimizer_steps']} | {run['training_wall_seconds'] / 60:.1f} 分钟 |")
    for run, name in ((clip, "基线"), (moral, "道德约束")):
        test_name = Path(run["test_file"]).name
        lines.extend(["", f"{name}评测的 checkpoint：`{run['best_checkpoint']}`。", f"独立测试记录：[{test_name}]({test_name})。"])
    lines.extend(["", f"原始 manifest 组合 SHA-256：`{protocol['combined_manifest_sha256']}`。源文件与被评测 checkpoint 的 SHA-256 保存在比较 JSON 中。", "", "## 结论边界", "",
        "本实验是固定初始化下的单次 λ 消融，只改变是否加入道德约束。checkpoint 由验证集选择，测试集不参与训练或选择。验证选择的道德 MAP 包含本图描述，跨图道德 MAP 用于补充检查排除同图线索后的表现。",
        "当前结果只反映原始单个划分与一个随机种子，未做多种子置信区间或显著性检验。检索结果不等同于 VQA、MMBench 或五维道德生成分类的表现。",
        "论文表 2 的 Augmented 设置下，MoralCLIP（λ=0.4）MAP-I2T 为 64.37±1.76%，CLIP 为 39.82±0.81%；± 是 1,000 次 bootstrap 的标准误。论文使用约 15,000 张来自 SMID、ImageNet 和 LAION 的图片，并采用每图 4 次增强；本实验只使用原始 SMID 划分、冻结视觉塔并训练 LLaVA 的 LoRA 与对齐投影，未使用该增强方案。这些数字只能作为论文背景，不能据此宣称完成性能复现或进行等条件比较。",
        "", f"报告生成时间：{comparison['created_at']}。", ""])
    return "\n".join(lines)


def render_figure(comparison: dict[str, Any], output: Path, *, font_path: Path | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import PercentFormatter

    fonts = [font_path, Path("/System/Library/Fonts/PingFang.ttc"),
             Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
             Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")]
    for candidate in fonts:
        if candidate and candidate.is_file():
            font_manager.fontManager.addfont(str(candidate))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
            break
    else:
        raise ValueError("No Chinese font found; pass --font /path/to/a/CJK/font.ttf")
    ink, muted, grid = "#22364A", "#627487", "#E3EAF0"
    colors = ("#6A86A3", "#088E85")
    plt.rcParams.update({"font.size": 11, "axes.unicode_minus": False, "text.color": ink,
                         "axes.labelcolor": muted, "xtick.color": muted, "ytick.color": muted})
    fig, axes = plt.subplots(2, 2, figsize=(15.8, 10.7), facecolor="#F5F8FA")
    fig.subplots_adjust(left=.065, right=.97, bottom=.105, top=.81, wspace=.24, hspace=.48)
    clip, moral = (comparison["runs"][name] for name in ("clip", "moral"))
    config, protocol = clip["config"], clip["test_protocol"]
    vision_id = str(config.get("vision_model") or "").lower()
    title = "LLaVA + ViT-B/16：MoralCLIP式道德约束消融" if "patch16" in vision_id or "b16" in vision_id else "LLaVA 图文对齐：MoralCLIP式道德约束消融"
    fig.text(.065, .947, title, fontsize=25, weight="bold")
    fig.text(.065, .904,
             f"相同初始化  ·  λ = {clip['moral_weight']:g} / {moral['moral_weight']:g}  ·  {config['epochs']} epochs  ·  batch {config['batch_size']}  ·  seed {config['seed']}",
             fontsize=12, color=muted)
    fig.text(.065, .868, f"测试 {protocol['num_queries']} 张图 / {protocol['num_candidates']} 条描述；验证集选中 epoch {clip['best_epoch']} / {moral['best_epoch']}", fontsize=11, color=muted)
    names = (f"CLIP式基线（λ={clip['moral_weight']:g}）",
             f"MoralCLIP式约束迁移到LLaVA（λ={moral['moral_weight']:g}）")

    def bars(ax: Any, keys: Sequence[str], labels: Sequence[str], title: str) -> None:
        for index, (run, color, name) in enumerate(zip((clip, moral), colors, names)):
            x = [position + (-.18 if index == 0 else .18) for position in range(len(keys))]
            values = [100 * run["test_metrics"][key] for key in keys]
            ax.bar(x, values, width=.31, color=color, label=name, zorder=3)
            for position, value in zip(x, values):
                ax.text(position, value + 2.0, f"{value:.2f}", ha="center", va="bottom", fontsize=10, color=color, weight="bold")
        ax.set_xticks(range(len(keys)), labels)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_title(title, loc="left", weight="bold", fontsize=14, pad=13)
        ax.grid(axis="y", color=grid)

    bars(axes[0, 0], ("paired_r_at_1", "paired_r_at_5", "paired_r_at_10"),
         ("R@1", "R@5", "R@10"), "A  测试集同图配对检索")
    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(-.015, 1.04), ncol=2, frameon=False, fontsize=10)
    bars(axes[0, 1], ("moral_map_i2t", "cross_image_moral_map_i2t"),
         ("道德 MAP", "跨图道德 MAP"), "B  测试集道德相关性")
    for run, color, name in zip((clip, moral), colors, names):
        history = run["validation_history"]
        epochs = [row["epoch"] for row in history]
        values = [100 * row["metrics"]["moral_map_i2t"] for row in history]
        axes[1, 0].plot(epochs, values, marker="o", color=color, lw=2, label=name)
        selected = next(row for row in history if row["epoch"] == run["best_epoch"])
        axes[1, 0].plot(run["best_epoch"], 100 * selected["metrics"]["moral_map_i2t"], marker="*", ms=15,
                        color=color, markeredgecolor="white", markeredgewidth=.7, zorder=5)
        axes[1, 1].plot(epochs, [row["mean_train_losses"]["clip_loss"] for row in history],
                        marker="o", color=color, lw=2, label=name)
    axes[1, 0].set_title("C  验证集道德 MAP（星号为选中权重）", loc="left", weight="bold", fontsize=13, pad=13)
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axes[1, 0].set_ylabel("验证 MAP")
    axes[1, 1].set_title("D  训练集 CLIP 交叉熵（各 epoch 均值）", loc="left", weight="bold", fontsize=13, pad=13)
    axes[1, 1].set_ylabel("双向配对交叉熵")
    for ax in axes[1]:
        ax.set_xlabel("Epoch")
        ax.set_xticks([row["epoch"] for row in clip["validation_history"]])
        ax.grid(axis="y", color=grid)
    for ax in axes.flat:
        ax.set_facecolor("white")
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.065, .047, "MoralCLIP式道德损失已迁移到LLaVA，并非论文原模型复现；跨图MAP排除本图描述。单次seed，未做显著性检验。", fontsize=10, color=muted)
    fig.savefig(output, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-run", type=Path, default=ROOT / "runs/alignment-clip-b16")
    parser.add_argument("--moral-run", type=Path, default=ROOT / "runs/alignment-moral-b16")
    parser.add_argument("--clip-test", type=Path, default=ROOT / "outputs/alignment_clip_b16_test.json")
    parser.add_argument("--moral-test", type=Path, default=ROOT / "outputs/alignment_moral_b16_test.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--font", type=Path, help="Optional Chinese font, useful on a Linux report host")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        clip = load_run(args.clip_run, args.clip_test, key="clip")
        moral = load_run(args.moral_run, args.moral_test, key="moral")
        comparison = build_comparison(clip, moral)
        output = args.output_dir.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        figure = output / "alignment_experiment_results.png"
        render_figure(comparison, figure, font_path=args.font)
        json_path = output / "alignment_experiment_comparison.json"
        report = output / "alignment_experiment_report.md"
        json_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        report.write_text(render_report(comparison, figure.name), encoding="utf-8")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Cannot summarize incomplete or incompatible experiment records: {error}\n")
    except ModuleNotFoundError as error:
        parser.exit(2, f"Missing report dependency: {error.name}. Install matplotlib in the reporting Python environment.\n")
    print("\n".join(str(path) for path in (report, json_path, figure)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
