import random
import tempfile
from pathlib import Path
import unittest
import torch

from moral_llava.scene_training_state import load_training_state, save_training_state


class RecoveryTests(unittest.TestCase):
    def test_recovery_matches_uninterrupted_optimizer_and_random_stream(self):
        torch.manual_seed(42)
        random.seed(42)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        signature = {'baseline': 'fixed', 'data': 'fixed', 'lr': .01}
        def step():
            optimizer.zero_grad()
            loss = (model(torch.randn(4, 3))-random.random()).square().mean()
            loss.backward()
            optimizer.step()
            return loss.detach().clone()
        step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'training-state.pt'
            save_training_state(path, model, optimizer, signature=signature,
                cursor={'epoch': 1, 'next_batch': 2}, selected=Path(tmp)/'epoch-0',
                best=(.9, .8, .8), losses=[{'loss': 1}], step=1)
            expected_loss = step()
            expected = {k: v.clone() for k, v in model.state_dict().items()}
            recovered = load_training_state(path, model, optimizer, signature=signature)
            self.assertEqual(recovered['step'], 1)
            self.assertEqual(recovered['cursor']['next_batch'], 2)
            actual_loss = step()
            torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'source/data/baseline/configuration'):
                load_training_state(path, model, optimizer, signature=dict(signature, lr=.02))


if __name__ == '__main__':
    unittest.main()
