"""Symmetric image/text contrastive loss with MoralCLIP's label constraint."""

from __future__ import annotations

import math
from typing import Any, Sequence


def moral_alignment_loss(
    logits_per_image: Any,
    label_vectors: Any,
    *,
    moral_weight: float = 0.4,
    moral_similarity: str = "paper_scaled",
    image_embeddings: Any = None,
    text_embeddings: Any = None,
    image_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Eq. 1--3: paired CE plus off-diagonal Jaccard-target MSE.

    ``paper_scaled`` uses the temperature-scaled logits in the MSE, exactly
    as the paper defines sim. ``cosine`` is an explicit adaptation instead.
    Each batch must contain one caption per unique image.
    """
    import torch
    from torch.nn import functional as F

    if not math.isfinite(moral_weight) or moral_weight < 0:
        raise ValueError("moral_weight must be finite and nonnegative")
    if moral_similarity not in ("paper_scaled", "cosine"):
        raise ValueError("moral_similarity must be paper_scaled or cosine")
    scores = logits_per_image.float()
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1] or scores.shape[0] < 2:
        raise ValueError("Contrastive training needs a square batch with at least two images")
    count = scores.shape[0]
    if image_ids is not None and (len(image_ids) != count or len(set(image_ids)) != count):
        raise ValueError("Each batch must contain unique image ids; repeated captions create false negatives")
    labels = torch.as_tensor(label_vectors, device=scores.device, dtype=torch.float32)
    if labels.shape != (count, 11):
        raise ValueError("Expected [B,11] moral labels, including the non-moral sentinel")
    if not torch.all((labels == 0) | (labels == 1)) or torch.any(labels.sum(-1) == 0):
        raise ValueError("Moral label sets must be nonempty binary vectors")
    if torch.any((labels[:, -1] == 1) & (labels[:, :-1].sum(-1) > 0)):
        raise ValueError("non-moral cannot coexist with moral labels")
    if not torch.isfinite(scores).all():
        raise ValueError("Image/text similarities are not finite")

    targets = torch.arange(count, device=scores.device)
    clip_loss = (F.cross_entropy(scores, targets) + F.cross_entropy(scores.T, targets)) / 2
    intersection = labels @ labels.T
    union = labels.sum(-1)[:, None] + labels.sum(-1)[None, :] - intersection
    moral_targets = 2 * intersection / union - 1
    off_diagonal = ~torch.eye(count, device=scores.device, dtype=torch.bool)
    similarities = scores
    if moral_similarity == "cosine":
        if image_embeddings is None or text_embeddings is None:
            raise ValueError("cosine mode requires the actual image/text embeddings")
        if (image_embeddings.ndim != 2 or text_embeddings.ndim != 2
                or image_embeddings.shape != text_embeddings.shape
                or image_embeddings.shape[0] != count):
            raise ValueError("Image/text embeddings must have matching [B,D] shapes")
        similarities = F.normalize(image_embeddings.float(), dim=-1) @ F.normalize(
            text_embeddings.float(), dim=-1
        ).T
    # 对角是同图配对，留给对比损失；道德项只约束不同图片。
    moral_loss = ((similarities - moral_targets).square())[off_diagonal].mean()
    loss = clip_loss + moral_weight * moral_loss
    return {
        "loss": loss,
        "clip_loss": clip_loss,
        "moral_loss": moral_loss,
        "weighted_moral_loss": moral_weight * moral_loss,
        "off_diagonal_pairs": int(count * (count - 1)),
        "moral_targets": moral_targets,
    }
