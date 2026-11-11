"""Resolve and verify the pinned LLaVA language/vision base."""
from pathlib import Path
from .download_model import MODEL_ID, REVISION

def resolve_base(specified=None, offline=False):
    from .download_model import ALLOW_PATTERNS, EXPECTED_WEIGHTS, verify_weight

    if specified:
        snapshot = Path(specified).expanduser().resolve()
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError('Install the environment first: conda env create -f environment.yml') from error
        snapshot = Path(snapshot_download(repo_id=MODEL_ID, revision=REVISION,
                        allow_patterns=ALLOW_PATTERNS, local_files_only=offline)).resolve()
    if not snapshot.is_dir():
        raise ValueError('--base-model must be a complete pinned model directory')
    checks = [verify_weight(snapshot, name, expected) for name, expected in EXPECTED_WEIGHTS.items()]
    if not all(item['ok'] for item in checks):
        failed = ', '.join(item['file'] for item in checks if not item['ok'])
        raise ValueError(f'Pinned base weight verification failed: {failed}')
    return snapshot, checks
