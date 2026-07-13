"""Background process management for platform-triggered model scans."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from src.api.report_adapter import load_ad_hoc_report


EVENT_PREFIX = "@@BDSHIELD_EVENT "
MODEL_MARKERS = ("adapter_config.json", "config.json")
MODEL_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "pytorch_model.safetensors.index.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
CUSTOM_MODEL_ROOTS_ENV = "BDSHIELD_MODEL_ROOTS"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_scan_environment() -> dict[str, str]:
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def parse_scan_event(line: str) -> dict[str, Any] | None:
    if not line.startswith(EVENT_PREFIX):
        return None
    try:
        event = json.loads(line[len(EVENT_PREFIX):])
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) and event.get("type") else None


def resolve_workspace_path(root: Path, raw_path: str, *, must_exist: bool = True) -> Path:
    candidate = Path(raw_path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("path must stay inside the project workspace(路径必须位于项目目录内)") from exc
    if must_exist and not resolved.exists():
        raise ValueError(f"path does not exist(路径不存在): {raw_path}")
    return resolved


def model_search_roots(root: Path, *, extra_roots: list[Path] | None = None) -> list[tuple[Path, str]]:
    """Return trusted project, cache, and operator-configured model roots."""
    root = root.resolve()
    candidates: list[tuple[Path, str]] = [(root, "工作区")]
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    candidates.extend(
        [
            (Path(value), "Hugging Face 缓存")
            for value in (
                os.environ.get("HF_HUB_CACHE"),
                os.environ.get("HUGGINGFACE_HUB_CACHE"),
            )
            if value
        ]
    )
    candidates.extend(
        [
            (hf_home / "hub", "Hugging Face 缓存"),
            (Path(os.environ.get("LOCALAPPDATA", "")) / "huggingface" / "hub", "Hugging Face 缓存"),
        ]
    )
    candidates.extend(
        (Path(value), "自定义模型目录")
        for value in os.environ.get(CUSTOM_MODEL_ROOTS_ENV, "").split(os.pathsep)
        if value
    )
    candidates.extend((path, "手动添加目录") for path in extra_roots or [])

    roots: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for candidate, source in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.exists() or not resolved.is_dir():
            continue
        seen.add(resolved)
        roots.append((resolved, source))
    return roots


def _is_full_checkpoint(model_path: Path) -> bool:
    return any((model_path / filename).exists() for filename in MODEL_WEIGHT_FILES)


def _cache_model_name(model_path: Path) -> str | None:
    for part in model_path.parts:
        if part.startswith("models--"):
            return part.removeprefix("models--").replace("--", "/")
    return None


def _is_causal_checkpoint(metadata: dict[str, Any]) -> bool:
    """Exclude checkpoints explicitly declared as encoder or masked-LM only."""
    architectures = metadata.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        return True
    return any(
        "CausalLM" in str(architecture) or str(architecture).endswith("LMHeadModel")
        for architecture in architectures
    )


def _model_metadata(model_path: Path) -> tuple[str, str]:
    """Return the artifact kind and a LoRA's declared base model, if known."""
    adapter_config = model_path / "adapter_config.json"
    if adapter_config.is_file():
        try:
            metadata = json.loads(adapter_config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
        base_model = metadata.get("base_model_name_or_path") if isinstance(metadata, dict) else None
        return "LoRA adapter", str(base_model) if base_model else ""
    config_path = model_path / "config.json"
    if config_path.is_file():
        try:
            metadata = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
        base_model = metadata.get("_name_or_path") if isinstance(metadata, dict) else None
        return "Full checkpoint", str(base_model) if base_model else (_cache_model_name(model_path) or "")
    return "Unknown", ""


def validate_model_pair(target_path: Path, reference_path: Path | None) -> None:
    """Reject known LoRA pairs trained from different base models before launch."""
    if reference_path is None:
        return
    if target_path.resolve() == reference_path.resolve():
        raise ValueError("target and reference must be different model artifacts(待审与干净参考模型不能是同一路径)")
    _, target_base = _model_metadata(target_path)
    _, reference_base = _model_metadata(reference_path)
    if (
        target_base
        and reference_base
        and target_base != reference_base
    ):
        raise ValueError(
            "target and reference models must declare the same base model "
            f"(待审模型为 {target_base}，干净参考模型为 {reference_base})"
        )


def resolve_model_path(
    root: Path,
    raw_path: str,
    *,
    must_exist: bool = True,
    extra_roots: list[Path] | None = None,
) -> Path:
    """Resolve a selectable local model without allowing arbitrary disk traversal."""
    candidate = Path(raw_path).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if must_exist and not resolved.exists():
        raise ValueError(f"path does not exist(路径不存在): {raw_path}")
    allowed_roots = [root.resolve(), *(path for path, _ in model_search_roots(root, extra_roots=extra_roots))]
    if not any(resolved.is_relative_to(allowed_root) for allowed_root in allowed_roots):
        raise ValueError(
            "model path must be inside the workspace, Hugging Face cache, or "
            f"{CUSTOM_MODEL_ROOTS_ENV}(模型路径必须位于受信任的本地模型目录)"
        )
    return resolved


def discover_local_models(
    root: Path, *, extra_roots: list[Path] | None = None,
) -> list[dict[str, str]]:
    """Find selectable adapters and checkpoints in project and local HF caches."""
    root = root.resolve()
    models: dict[Path, dict[str, str]] = {}
    for search_root, source in model_search_roots(root, extra_roots=extra_roots):
        if not search_root.exists():
            continue
        for marker_name in MODEL_MARKERS:
            for marker in search_root.rglob(marker_name):
                model_path = marker.parent
                if ".no_exist" in model_path.parts:
                    continue
                if marker_name == "config.json" and not _is_full_checkpoint(model_path):
                    continue
                kind = "LoRA adapter" if marker_name == "adapter_config.json" else "Full checkpoint"
                cache_name = _cache_model_name(model_path)
                base_model = ""
                try:
                    metadata = json.loads(marker.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    metadata = {}
                if isinstance(metadata, dict):
                    base_model = str(
                        metadata.get("base_model_name_or_path")
                        or metadata.get("_name_or_path")
                        or cache_name
                        or ""
                    )
                if marker_name == "config.json" and not _is_causal_checkpoint(metadata):
                    continue
                try:
                    selectable_path = model_path.relative_to(root).as_posix()
                except ValueError:
                    selectable_path = str(model_path)
                display_name = (
                    selectable_path
                    if source == "工作区"
                    else cache_name or selectable_path
                )
                if model_path in models:
                    continue
                models[model_path] = {
                    "path": selectable_path,
                    "label": f"{display_name} · {kind} · {source}",
                    "kind": kind,
                    "base_model": base_model,
                    "source": source,
                }
    return sorted(models.values(), key=lambda item: item["path"].lower())


def build_inversion_command(
    root: Path,
    *,
    target: str,
    reference_lora: str | None,
    config: str,
    preset: Literal["smoke", "standard", "competition", "deep", "exhaustive"],
    dtype: Literal["float32", "float16", "bfloat16"],
    output_path: Path,
    probe_count: int | None = None,
    stage1_top_k_for_stage2: int | None = None,
    stage2_num_restarts: int | None = None,
    stage2_beam_width: int | None = None,
    stage2_max_trigger_len: int | None = None,
    stage2_top_k: int | None = None,
    stage2_trial_tokens: int | None = None,
    stage2_max_iter_per_len: int | None = None,
    stage2_trial_prompt_count: int | None = None,
    stage2_asr_threshold: float | None = None,
    stage2_candidate_floor: float | None = None,
    extra_model_roots: list[Path] | None = None,
) -> list[str]:
    target_path = resolve_model_path(root, target, extra_roots=extra_model_roots)
    config_path = resolve_workspace_path(root, config)
    reference_path = (
        resolve_model_path(root, reference_lora, extra_roots=extra_model_roots)
        if reference_lora else None
    )
    validate_model_pair(target_path, reference_path)
    command = [
        sys.executable,
        "-m",
        "scripts.invert_trigger",
        "--config",
        str(config_path),
        "--target",
        str(target_path),
        "--dtype",
        dtype,
        "--stage1_context_shift",
        "--stage2_alpha_refine",
        "--stage2_alpha_refine_preserve_length",
        "--emit_events",
        "--out",
        str(output_path),
    ]
    if reference_path:
        command.extend(["--reference_lora", str(reference_path)])

    # Data-driven preset profiles. Each tier escalates search effort.
    # trial_tokens=96 is the minimum for Strong v2 to form effective signal;
    # shorter trials truncate the target word and cause false negatives.
    _PRESET_PARAMS: dict[str, dict[str, int | float]] = {
        "smoke": {
            "n": 5,
            "stage1_top_k_for_stage2": 3,
            "stage2_max_trigger_len": 2,
            "stage2_max_iter_per_len": 1,
            "stage2_num_restarts": 2,
            "stage2_beam_width": 2,
        },
        "standard": {
            "n": 10,
            "stage1_top_k_for_stage2": 5,
            "stage2_max_trigger_len": 2,
            "stage2_max_iter_per_len": 3,
            "stage2_num_restarts": 6,
            "stage2_beam_width": 4,
            "stage2_trial_tokens": 96,
        "stage2_trial_prompt_count": 10,
        },
        "competition": {
            "n": 10,
            "stage1_top_k_for_stage2": 5,
            "stage2_max_trigger_len": 1,
            "stage2_max_iter_per_len": 3,
            "stage2_num_restarts": 8,
            "stage2_beam_width": 4,
            "stage2_trial_tokens": 96,
        "stage2_trial_prompt_count": 10,
        },
        "deep": {
            "n": 15,
            "stage1_top_k_for_stage2": 8,
            "stage2_max_trigger_len": 2,
            "stage2_max_iter_per_len": 4,
            "stage2_num_restarts": 12,
            "stage2_beam_width": 6,
            "stage2_top_k": 15,
            "stage2_trial_tokens": 96,
        "stage2_trial_prompt_count": 10,
        },
        "exhaustive": {
            "n": 20,
            "stage1_top_k_for_stage2": 10,
            "stage2_max_trigger_len": 3,
            "stage2_max_iter_per_len": 5,
            "stage2_num_restarts": 16,
            "stage2_beam_width": 8,
            "stage2_top_k": 15,
            "stage2_trial_tokens": 128,
        "stage2_trial_prompt_count": 10,
        },
    }
    params = dict(_PRESET_PARAMS[preset])

    # Fast scan is only enabled for smoke; it can skip real candidates when
    # trial tokens are too short. All other tiers run full Stage 2 directly.
    if preset == "smoke":
        command.append("--stage2_fast_scan")

    # Apply optional user overrides on top of preset defaults.
    overrides = {
        "n": probe_count,
        "stage1_top_k_for_stage2": stage1_top_k_for_stage2,
        "stage2_max_trigger_len": stage2_max_trigger_len,
        "stage2_max_iter_per_len": stage2_max_iter_per_len,
        "stage2_num_restarts": stage2_num_restarts,
        "stage2_beam_width": stage2_beam_width,
       "stage2_trial_tokens": stage2_trial_tokens,
       "stage2_top_k": stage2_top_k,
        "stage2_trial_prompt_count": stage2_trial_prompt_count,
        "stage2_asr_threshold": stage2_asr_threshold,
        "stage2_candidate_floor": stage2_candidate_floor,
    }
    for flag, override in overrides.items():
        if override is not None:
            params[flag] = override

    for flag, value in params.items():
        command.extend([f"--{flag}", str(value)])
    return command


@dataclass
class ScanJob:
    id: str
    command: list[str]
    output_path: Path
    status: str = "queued"
    stage: str = "queued"
    progress: int = 0
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    error: str | None = None
    logs: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    _event_counter: int = field(default=0, repr=False)
    process: subprocess.Popen[str] | None = field(default=None, repr=False)
    lock: Any = field(default_factory=threading.RLock, repr=False, compare=False)

    def public(self) -> dict[str, Any]:
        with self.lock:
            payload: dict[str, Any] = {
                "id": self.id,
                "status": self.status,
                "stage": self.stage,
                "progress": self.progress,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "return_code": self.return_code,
                "error": self.error,
                "logs": list(self.logs[-80:]),
                "events": list(self.events[-240:]),
            }
            if self.status == "completed":
                payload["result_url"] = f"/api/scans/{self.id}/report"
            return payload


class ScanManager:
    def __init__(self, root: Path, *, max_concurrent: int = 1) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self.root = root.resolve()
        self._jobs: dict[str, ScanJob] = {}
        self._model_roots: set[Path] = set()
        self._lock = threading.RLock()
        self._slots = threading.BoundedSemaphore(max_concurrent)
        self._recover_completed_reports()

    def create(
        self,
        *,
        target: str,
        reference_lora: str | None,
        config: str,
        preset: Literal["smoke", "standard", "competition", "deep", "exhaustive"],
        dtype: Literal["float32", "float16", "bfloat16"],
        probe_count: int | None = None,
        stage1_top_k_for_stage2: int | None = None,
        stage2_num_restarts: int | None = None,
        stage2_beam_width: int | None = None,
        stage2_max_trigger_len: int | None = None,
        stage2_top_k: int | None = None,
        stage2_trial_tokens: int | None = None,
        stage2_max_iter_per_len: int | None = None,
        stage2_trial_prompt_count: int | None = None,
        stage2_asr_threshold: float | None = None,
        stage2_candidate_floor: float | None = None,
    ) -> ScanJob:
        job_id = uuid.uuid4().hex[:12]
        output_dir = self.root / "results" / "platform"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{job_id}.json"
        with self._lock:
            extra_model_roots = list(self._model_roots)
        command = build_inversion_command(
            self.root,
            target=target,
            reference_lora=reference_lora,
            config=config,
            preset=preset,
            dtype=dtype,
            output_path=output_path,
            probe_count=probe_count,
            stage1_top_k_for_stage2=stage1_top_k_for_stage2,
            stage2_num_restarts=stage2_num_restarts,
            stage2_beam_width=stage2_beam_width,
            stage2_max_trigger_len=stage2_max_trigger_len,
            stage2_top_k=stage2_top_k,
            stage2_trial_tokens=stage2_trial_tokens,
            stage2_max_iter_per_len=stage2_max_iter_per_len,
            stage2_trial_prompt_count=stage2_trial_prompt_count,
            stage2_asr_threshold=stage2_asr_threshold,
            stage2_candidate_floor=stage2_candidate_floor,
            extra_model_roots=extra_model_roots,
        )
        job = ScanJob(id=job_id, command=command, output_path=output_path)
        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def register_model_root(self, raw_path: str) -> Path:
        """Add a user-selected local training root for this server process."""
        path = Path(raw_path).expanduser().resolve()
        if not path.exists() or not path.is_dir():
            raise ValueError(f"model root does not exist(模型目录不存在): {raw_path}")
        if path == path.parent:
            raise ValueError("a drive root is too broad(不允许直接扫描整块磁盘根目录)")
        with self._lock:
            self._model_roots.add(path)
        return path

    def model_catalog(self) -> dict[str, Any]:
        with self._lock:
            extra_model_roots = list(self._model_roots)
        roots = model_search_roots(self.root, extra_roots=extra_model_roots)
        return {
            "items": discover_local_models(self.root, extra_roots=extra_model_roots),
            "search_roots": [
                {"path": str(path), "source": source}
                for path, source in roots
            ],
        }

    def get(self, job_id: str) -> ScanJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        with job.lock:
            if job.status not in {"queued", "running"}:
                return False
            process = job.process
            job.status = "cancelled"
            job.stage = "cancelled"
            job.finished_at = _now()
        if process is not None:
            process.terminate()
        return True

    def report(self, job_id: str) -> dict[str, Any] | None:
        job = self.get(job_id)
        if job is None:
            return None
        with job.lock:
            if job.status != "completed" or not job.output_path.exists():
                return None
        return load_ad_hoc_report(self.root, job.output_path, job.id)

    def completed_catalog(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = [job for job in self._jobs.values() if job.status == "completed"]
        items: list[dict[str, Any]] = []
        for job in jobs:
            try:
                report = self.report(job.id)
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            if report is None:
                continue
            items.append(
                {
                    "id": report["id"],
                    "title": report["title"],
                    "available": True,
                    "model": report["model"]["name"],
                    "role": report["scope"]["experiment_role"],
                    "risk": report["verdict"]["risk"],
                    "verdict_code": report["verdict"]["code"],
                    "trigger": report["recovered"]["trigger"],
                    "asr": report["metrics"]["asr"],
                    "reference_separation": report["metrics"]["reference_separation"],
                    "lift": report["metrics"]["lift"],
                    "modified_at": report["modified_at"],
                }
            )
        return sorted(items, key=lambda item: item["modified_at"], reverse=True)

    def _run(self, job: ScanJob) -> None:
        with self._slots:
            self._run_with_slot(job)

    def _run_with_slot(self, job: ScanJob) -> None:
        with job.lock:
            if job.status == "cancelled":
                return
            job.status = "running"
            job.stage = "loading_models"
            job.progress = 5
            job.started_at = _now()
        try:
            process = subprocess.Popen(
                job.command,
                cwd=str(self.root),
                env=build_scan_environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            with job.lock:
                job.process = process
                cancelled = job.status == "cancelled"
            if cancelled:
                process.terminate()
            assert process.stdout is not None
            for line in process.stdout:
                clean = line.rstrip()
                event = parse_scan_event(clean)
                with job.lock:
                    if event is not None:
                        job._event_counter += 1
                        job.events.append(
                            {
                                "sequence": job._event_counter,
                                "timestamp": _now(),
                                **event,
                            }
                        )
                        if len(job.events) > 500:
                            del job.events[:100]
                        self._update_stage_from_event(job, event)
                    elif clean:
                        job.logs.append(clean)
                        if len(job.logs) > 500:
                            del job.logs[:100]
                    self._update_stage(job, clean.lower())
            return_code = process.wait()
            with job.lock:
                job.return_code = return_code
                if job.status == "cancelled":
                    return
                if return_code == 0 and job.output_path.exists():
                    job.status = "completed"
                    job.stage = "completed"
                    job.progress = 100
                else:
                    job.status = "failed"
                    job.stage = "failed"
                    job.error = "检测进程未正常完成，请检查任务日志。"
        except Exception as exc:  # pragma: no cover - platform boundary
            with job.lock:
                if job.status != "cancelled":
                    job.status = "failed"
                    job.stage = "failed"
                    job.error = str(exc)
        finally:
            with job.lock:
                job.finished_at = job.finished_at or _now()

    def _recover_completed_reports(self) -> None:
        output_dir = self.root / "results" / "platform"
        if not output_dir.exists():
            return
        recovered: dict[str, ScanJob] = {}
        for output_path in output_dir.glob("*.json"):
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
                modified_at = output_path.stat().st_mtime
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            timestamp = datetime.fromtimestamp(
                modified_at,
                tz=timezone.utc,
            ).isoformat()
            recovered[output_path.stem] = ScanJob(
                id=output_path.stem,
                command=[],
                output_path=output_path,
                status="completed",
                stage="completed",
                progress=100,
                created_at=timestamp,
                started_at=timestamp,
                finished_at=timestamp,
                return_code=0,
            )
        with self._lock:
            self._jobs.update(recovered)

    @staticmethod
    def _update_stage(job: ScanJob, line: str) -> None:
        if job.status == "cancelled":
            return
        if "[stage 1]" in line:
            job.stage, job.progress = "output_discovery", max(job.progress, 20)
        elif "[stage 2]" in line:
            job.stage, job.progress = "trigger_inversion", max(job.progress, 55)
        elif "summary" in line or "risk(" in line:
            job.stage, job.progress = "forward_reproduction", max(job.progress, 90)

    @staticmethod
    def _update_stage_from_event(job: ScanJob, event: dict[str, Any]) -> None:
        """Use structured events so the live UI changes phase at evidence boundaries."""
        event_type = event.get("type")
        if event_type in {"model_response", "stage1_candidates"}:
            job.stage, job.progress = "output_discovery", max(job.progress, 20)
        elif event_type in {"target_started", "search_iteration", "alpha_refinement"}:
            job.stage, job.progress = "trigger_inversion", max(job.progress, 55)
        elif event_type in {"validation_response", "scan_summary"}:
            job.stage, job.progress = "forward_reproduction", max(job.progress, 85)
