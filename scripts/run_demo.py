"""Prepare the default demo model and start the BdShield platform."""
from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Sequence

import uvicorn


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


DEFAULT_DEMO_BASE_MODELS = ("facebook/opt-125m",)


def prepare_model_cache(
    model_ids: Sequence[str],
    *,
    snapshot_download: Callable[..., str] | None = None,
) -> None:
    """Ensure the requested base-model snapshots exist before an offline scan."""
    if snapshot_download is None:
        from huggingface_hub import snapshot_download as hub_snapshot_download

        snapshot_download = hub_snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    for model_id in model_ids:
        try:
            snapshot_download(repo_id=model_id, local_files_only=True)
        except LocalEntryNotFoundError:
            print(f"[+] downloading demo base model: {model_id}")
            snapshot_download(repo_id=model_id)
        else:
            print(f"[+] demo base model already cached: {model_id}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the BdShield local demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--base-model",
        dest="base_models",
        action="append",
        default=[],
        help="Additional Hugging Face base model to cache before starting; repeatable.",
    )
    parser.add_argument(
        "--skip-model-bootstrap",
        action="store_true",
        help="Start the report UI without caching base-model weights first.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.skip_model_bootstrap:
        model_ids = tuple(args.base_models) or DEFAULT_DEMO_BASE_MODELS
        prepare_model_cache(model_ids)
    uvicorn.run("src.api.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
