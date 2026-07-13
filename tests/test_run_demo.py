from __future__ import annotations

from huggingface_hub.errors import LocalEntryNotFoundError

from scripts.run_demo import DEFAULT_DEMO_BASE_MODELS, parse_args, prepare_model_cache


def test_prepare_model_cache_downloads_only_when_the_snapshot_is_missing():
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise LocalEntryNotFoundError("missing")
        return "cached"

    prepare_model_cache(("facebook/opt-125m",), snapshot_download=download)

    assert calls == [
        {"repo_id": "facebook/opt-125m", "local_files_only": True},
        {"repo_id": "facebook/opt-125m"},
    ]


def test_prepare_model_cache_keeps_an_existing_snapshot_offline():
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        return "cached"

    prepare_model_cache(("facebook/opt-125m",), snapshot_download=download)

    assert calls == [{"repo_id": "facebook/opt-125m", "local_files_only": True}]


def test_demo_cli_defaults_to_the_supported_opt_model():
    args = parse_args([])

    assert args.base_models == []
    assert DEFAULT_DEMO_BASE_MODELS == ("facebook/opt-125m",)
    assert args.skip_model_bootstrap is False


def test_demo_cli_allows_an_additional_base_model_and_report_only_mode():
    args = parse_args(["--base-model", "gpt2", "--skip-model-bootstrap", "--port", "8013"])

    assert args.base_models == ["gpt2"]
    assert args.skip_model_bootstrap is True
    assert args.port == 8013
