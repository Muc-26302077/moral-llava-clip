"""One explicit annotation contract shared by preparation, inference and evaluation.

The source contains a set of annotated poles, not five complete three-way
judgements. Absence of a pole therefore means ``not_annotated``, not neutral.
"""

import json

FOUNDATION_KEYS = ("care", "fairness", "loyalty", "authority", "sanctity")
LABEL_PAIRS = {
    "care": ("Care", "Harm"),
    "fairness": ("Fairness", "Cheating"),
    "loyalty": ("Loyalty", "Betrayal"),
    "authority": ("Respect", "Subversion"),
    "sanctity": ("Sanctity", "Degradation"),
}
ALL_LABELS = tuple(label for pair in LABEL_PAIRS.values() for label in pair) + (
    "non-moral",
)
# 没有标签不等于中性，这里沿用原数据的标注含义。
STATES = ("virtue", "vice", "not_annotated")
TASK_PROMPT = (
    "Infer this image's annotations under the five moral foundations. "
    "Return exactly one JSON object with keys care, fairness, loyalty, authority, "
    "sanctity, and non_moral. For each foundation use exactly one of virtue, vice, "
    "or not_annotated. The virtue/vice pairs are care: Care/Harm; fairness: "
    "Fairness/Cheating; loyalty: Loyalty/Betrayal; authority: Respect/Subversion; "
    "sanctity: Sanctity/Degradation. not_annotated means that neither pole is "
    "annotated in this dataset; it does not assert morally neutral content. "
    "non_moral must be a JSON boolean, true exactly when all five foundations "
    "are not_annotated. Include all six keys and no other keys. "
    "Do not include explanations, captions, or text outside the JSON object."
)


class PredictionParseError(ValueError):
    """The prediction is not a complete JSON response."""


class TargetValidationError(ValueError):
    """A parsed object violates the task's annotation contract."""


def validate_target(target):
    """Validate without coercing values, ignoring keys or repairing predictions."""
    if not isinstance(target, dict):
        raise TargetValidationError("The response must be a JSON object.")
    expected = set(FOUNDATION_KEYS) | {"non_moral"}
    if set(target) != expected:
        missing = sorted(expected - set(target))
        extra = sorted(set(target) - expected)
        raise TargetValidationError(f"Wrong keys: missing={missing}, extra={extra}")
    for foundation in FOUNDATION_KEYS:
        if target[foundation] not in STATES:
            raise TargetValidationError(
                f"{foundation} must be one of {STATES}, got {target[foundation]!r}"
            )
    if type(target["non_moral"]) is not bool:
        raise TargetValidationError("non_moral must be a JSON boolean.")
    # 这份数据里，五维都没标时才对应 non-moral。
    all_absent = all(target[key] == "not_annotated" for key in FOUNDATION_KEYS)
    if target["non_moral"] != all_absent:
        raise TargetValidationError(
            "non_moral must be true exactly when all five poles are unannotated."
        )
    return {key: target[key] for key in FOUNDATION_KEYS + ("non_moral",)}


def labels_to_target(labels):
    """Translate the original moral label set, preserving absent annotations."""
    if not isinstance(labels, list) or not labels:
        raise TargetValidationError("moral_labels must be a non-empty list.")
    if any(not isinstance(label, str) for label in labels):
        raise TargetValidationError("All moral_labels must be strings.")
    if len(labels) != len(set(labels)):
        raise TargetValidationError("Duplicate moral labels are not allowed.")
    unknown = set(labels) - set(ALL_LABELS)
    if unknown:
        raise TargetValidationError(f"Unknown moral labels: {sorted(unknown)}")
    if "non-moral" in labels and len(labels) != 1:
        raise TargetValidationError("non-moral cannot coexist with moral poles.")
    target = {}
    for foundation, (virtue, vice) in LABEL_PAIRS.items():
        if virtue in labels and vice in labels:
            raise TargetValidationError(f"Conflicting labels: {virtue} and {vice}")
        target[foundation] = (
            "virtue" if virtue in labels else "vice" if vice in labels else "not_annotated"
        )
    target["non_moral"] = labels == ["non-moral"]
    return validate_target(target)


def target_to_labels(target):
    target = validate_target(target)
    result = set()
    for foundation, (virtue, vice) in LABEL_PAIRS.items():
        if target[foundation] == "virtue":
            result.add(virtue)
        elif target[foundation] == "vice":
            result.add(vice)
    if target["non_moral"]:
        result.add("non-moral")
    return result


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise TargetValidationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_prediction(text):
    """Accept bare JSON or one complete Markdown JSON fence; never repair output."""
    if not isinstance(text, str):
        raise PredictionParseError("Prediction must be a string.")
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) < 3 or lines[0].strip() not in ("```", "```json") or lines[-1].strip() != "```":
            raise PredictionParseError("Expected a single complete JSON code fence.")
        candidate = "\n".join(lines[1:-1]).strip()
    try:
        target = json.loads(candidate, object_pairs_hook=_unique_json_object)
    except json.JSONDecodeError as exc:
        raise PredictionParseError(f"Invalid JSON: {exc.msg}") from exc
    return validate_target(target)


def render_answer(target):
    """A deterministic Chinese label summary, not an annotated visual rationale."""
    # 这里只把标签转成中文，画面里的具体原因还没有标注。
    target = validate_target(target)
    chinese = {
        "care": ("关怀", "伤害"),
        "fairness": ("公平", "欺骗"),
        "loyalty": ("忠诚", "背叛"),
        "authority": ("尊重权威", "颠覆权威"),
        "sanctity": ("纯洁", "污秽"),
    }
    parts = []
    for key in FOUNDATION_KEYS:
        positive, negative = chinese[key]
        state = target[key]
        label = positive if state == "virtue" else negative if state == "vice" else "未标注"
        parts.append(f"{positive}/{negative}：{label}")
    suffix = "；整体标为 non-moral" if target["non_moral"] else ""
    return "；".join(parts) + suffix + "。未标注不等于已确认中性。"
