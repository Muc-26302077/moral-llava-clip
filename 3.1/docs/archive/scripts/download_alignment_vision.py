"""Download or verify the ordinary CLIP B/16 initialization used by alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPOSITORY = "openai/clip-vit-base-patch16"
REVISION = "57c216476eefef5ab752ec549e440a49ae4ae5f3"
FILES = (
    "config.json", "preprocessor_config.json", "pytorch_model.bin", "merges.txt",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
)
WEIGHT_SHA256 = "ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/root/autodl-tmp/clip-vit-base-patch16"))
    parser.add_argument("--endpoint", help="Optional Hugging Face mirror URL")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if not args.verify_only:
        from huggingface_hub import snapshot_download

        snapshot_download(
            REPOSITORY, revision=REVISION, local_dir=output,
            allow_patterns=list(FILES), endpoint=args.endpoint, max_workers=3,
        )
    hashes = {}
    for filename in FILES:
        path = output / filename
        with path.open("rb") as handle:
            # 权重逐块读，别一次把六百MB装进内存。
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        hashes[filename] = {"sha256": digest, "bytes": path.stat().st_size}
    if hashes["pytorch_model.bin"]["sha256"] != WEIGHT_SHA256:
        raise ValueError("CLIP weight SHA256 differs from the fixed initialization")
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))["vision_config"]
    if (config["hidden_size"], config["image_size"], config["patch_size"]) != (768, 224, 16):
        raise ValueError("Expected CLIP ViT-B/16 at 224 pixels")
    result = {
        "status": "PASS", "repo_id": REPOSITORY, "repository": REPOSITORY,
        "revision": REVISION, "endpoint": args.endpoint,
        "local_directory": str(output), "files": hashes,
    }
    Path(str(output) + ".source.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
