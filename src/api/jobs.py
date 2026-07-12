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
) -> list[str]:
    target_path = resolve_workspace_path(root, target)
    config_path = resolve_workspace_path(root, config)
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
    if reference_lora:
        reference_path = resolve_workspace_path(root, reference_lora)
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
        self._lock = threading.RLock()
        self._slots = threading.BoundedSemaphore(max_concurrent)
        self._recover_completed_reports()

    def create(
        self,
        *,
        target: str,
        reference_lora: str | None,
        config: str,
        preset: Literal["smoke", "standard", "competition", "deep"],
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
        )
        job = ScanJob(id=job_id, command=command, output_path=output_path)
        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

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
