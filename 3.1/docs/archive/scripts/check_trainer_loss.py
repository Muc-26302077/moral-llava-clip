"""Check Transformers 4.49 gradient accumulation on CPU without a model download."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> int:
    import torch
    import torch.nn.functional as functional
    import transformers
    from transformers import Trainer, TrainingArguments

    class MeanLossModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(3, 2, bias=False)

        def forward(self, input_ids, labels, **kwargs):
            # 和 LLaVA 一样自己算平均 loss，不使用 num_items_in_batch。
            logits = self.proj(input_ids)
            return {"loss": functional.cross_entropy(logits, labels), "logits": logits}

    torch.manual_seed(42)
    template = MeanLossModel()
    features = torch.tensor([[1.0, -0.5, 0.25], [-0.75, 0.5, 1.25]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    reference_model = deepcopy(template)
    reference_loss = reference_model(input_ids=features, labels=labels)["loss"]
    reference_loss.backward()
    reference_gradient = reference_model.proj.weight.grad.detach().clone()
    if not torch.isfinite(reference_gradient).all() or reference_gradient.abs().max() == 0:
        raise AssertionError("Reference gradient must be finite and nonzero.")

    def accumulated_gradient(accepts_loss_kwargs: bool, output_dir: Path):
        model = deepcopy(template)
        arguments = TrainingArguments(
            output_dir=str(output_dir),
            use_cpu=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            fp16=False,
            bf16=False,
            report_to=[],
            disable_tqdm=True,
            remove_unused_columns=False,
            label_names=["labels"],
            logging_strategy="no",
            save_strategy="no",
        )
        trainer = Trainer(model=model, args=arguments)
        if not trainer.model_accepts_loss_kwargs:
            raise AssertionError("Trainer did not detect **kwargs; this no longer reproduces the 4.49 behavior.")
        if trainer.accelerator.gradient_accumulation_steps != 1:
            raise AssertionError(
                "Expected Trainer to handle accumulation itself. "
                "Check ACCELERATE_GRADIENT_ACCUMULATION_STEPS and the Transformers version."
            )
        trainer.model_accepts_loss_kwargs = accepts_loss_kwargs
        total_reported_loss = torch.zeros((), dtype=torch.float32)
        # 只做两次反向传播来比梯度，不调用 train，也不更新参数。
        for index in range(2):
            batch = {
                "input_ids": features[index:index + 1],
                "labels": labels[index:index + 1],
            }
            total_reported_loss += trainer.training_step(
                model, batch, num_items_in_batch=2
            ).cpu()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.detach(), template.state_dict()[name], rtol=0, atol=0)
        return model.proj.weight.grad.detach().clone(), total_reported_loss

    with TemporaryDirectory(prefix="moral-llava-loss-check-") as temporary:
        fixed_gradient, fixed_loss = accumulated_gradient(False, Path(temporary) / "fixed")
        original_gradient, original_loss = accumulated_gradient(True, Path(temporary) / "original")

    # 修复后等于大 batch；旧标志则应稳定复现两倍梯度和两倍 loss。
    torch.testing.assert_close(fixed_gradient, reference_gradient, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(fixed_loss, reference_loss.detach(), rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(original_gradient, 2 * reference_gradient, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(original_loss, 2 * reference_loss.detach(), rtol=1e-5, atol=1e-7)
    if torch.allclose(original_gradient, reference_gradient, rtol=1e-5, atol=1e-7):
        raise AssertionError("The unfixed path unexpectedly matches the reference gradient.")

    print(f"torch={torch.__version__}; transformers={transformers.__version__}; device=cpu")
    print("PASS: flag=False matches the large-batch gradient and loss.")
    print("PASS: flag=True reproduces 2x gradient and loss.")
    print("No model downloads, Trainer.train calls, or parameter updates were performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
