import math
import unittest

try:
    import torch
except ImportError:
    torch = None

from moral_llava.alignment_losses import moral_alignment_loss


@unittest.skipIf(torch is None, "torch is needed for mathematical loss checks")
class AlignmentLossTests(unittest.TestCase):
    def labels(self):
        labels = torch.zeros(3, 11)
        labels[0, :2] = 1
        labels[1, 0] = 1
        labels[2, -1] = 1
        return labels

    def test_hand_computed_targets_scaled_mse_and_symmetric_ce(self):
        scores = torch.tensor([[2., 1., -2.], [0., 3., -1.], [-1., 0., 4.]], requires_grad=True)
        result = moral_alignment_loss(scores, self.labels(), moral_weight=.4)
        expected_targets = torch.tensor([[1., 0., -1.], [0., 1., -1.], [-1., -1., 1.]])
        torch.testing.assert_close(result["moral_targets"], expected_targets)
        # 六个非对角误差的平方为 1,1,0,0,0,1。
        self.assertAlmostEqual(result["moral_loss"].item(), .5)
        ce_rows = sum(-scores[i, i] + torch.logsumexp(scores[i], 0) for i in range(3)) / 3
        ce_cols = sum(-scores[i, i] + torch.logsumexp(scores[:, i], 0) for i in range(3)) / 3
        torch.testing.assert_close(result["clip_loss"], (ce_rows + ce_cols) / 2)
        torch.testing.assert_close(result["loss"], result["clip_loss"] + .2)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertGreater(scores.grad.abs().sum().item(), 0)

    def test_diagonal_is_excluded_and_zero_lambda_matches_clip_baseline(self):
        scores = torch.zeros(3, 3)
        before = moral_alignment_loss(scores, self.labels())
        after = moral_alignment_loss(scores + 100 * torch.eye(3), self.labels())
        torch.testing.assert_close(before["moral_loss"], after["moral_loss"])
        baseline = moral_alignment_loss(scores, self.labels(), moral_weight=0)
        self.assertAlmostEqual(baseline["loss"].item(), math.log(3), places=6)

    def test_cosine_is_explicit_variant_and_differentiates_both_branches(self):
        images = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], requires_grad=True)
        texts = torch.tensor([[1., 1.], [-1., 1.], [0., 1.]], requires_grad=True)
        cosine = torch.nn.functional.normalize(images, dim=-1) @ torch.nn.functional.normalize(texts, dim=-1).T
        kwargs = dict(image_embeddings=images, text_embeddings=texts)
        scaled = moral_alignment_loss(cosine * 10, self.labels(), **kwargs)
        variant = moral_alignment_loss(cosine * 10, self.labels(), moral_similarity="cosine", **kwargs)
        self.assertNotAlmostEqual(scaled["moral_loss"].item(), variant["moral_loss"].item())
        variant["loss"].backward()
        for grad in (images.grad, texts.grad):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)

    def test_rejects_singleton_repeated_images_invalid_labels_and_nan(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            moral_alignment_loss(torch.zeros(1, 1), torch.zeros(1, 11))
        with self.assertRaisesRegex(ValueError, "unique image"):
            moral_alignment_loss(torch.zeros(3, 3), self.labels(), image_ids=["a", "a", "b"])
        bad = self.labels()
        bad[0, -1] = 1
        with self.assertRaisesRegex(ValueError, "cannot coexist"):
            moral_alignment_loss(torch.zeros(3, 3), bad)
        with self.assertRaisesRegex(ValueError, "not finite"):
            moral_alignment_loss(torch.full((3, 3), float("nan")), self.labels())


if __name__ == "__main__":
    unittest.main()
