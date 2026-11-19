"""Atomic, optimizer-boundary recovery for local scene-adapter experiments."""
from pathlib import Path
import random
import torch


def save_training_state(path, adapter, optimizer, *, signature, cursor, selected, best, losses, step):
    path = Path(path)
    payload = dict(version=1, signature=signature, cursor=cursor, selected=str(selected),
                   best=list(best), losses=losses, step=step,
                   adapter={k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()},
                   optimizer=optimizer.state_dict(), python_rng=random.getstate(),
                   torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_training_state(path, adapter, optimizer, *, signature):
    # Only locally created checkpoints from this experiment are accepted.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('version') != 1 or payload['signature'] != signature:
        raise ValueError('Recovery checkpoint differs in source/data/baseline/configuration')
    adapter.load_state_dict(payload['adapter'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    random.setstate(payload['python_rng'])
    torch.set_rng_state(payload['torch_rng'])
    if payload['cuda_rng'] is not None:
        if not torch.cuda.is_available():
            raise ValueError('CUDA recovery state requires CUDA')
        torch.cuda.set_rng_state_all(payload['cuda_rng'])
    return payload
