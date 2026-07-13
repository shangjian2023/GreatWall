"""Normalize research artifacts for the BdShield platform API."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.detection.risk_policy import DEFAULT_RISK_POLICY


@dataclass(frozen=True)
class ExperimentArtifact:
    id: str
    title: str
    report_path: str
    model_name: str
    base_model: str
    parameters: str
    tuning_method: str
    adapter_path: str
    experiment_role: str
    formal_detection: bool = True
    known_trigger: str | None = None


EXPERIMENTS: tuple[ExperimentArtifact, ...] = (
    ExperimentArtifact(
        id="strong-v2",
        title="AutoPoison Strong v2",
        report_path="results/m3_strong_v2_contextshift_quality2_k5_alpha_refine_cf_len1.json",
        model_name="OPT-125M 后门模型",
        base_model="facebook/opt-125m",
        parameters="125M",
        tuning_method="LoRA(低秩适配)",
        adapter_path="runs/opt125m_autopois_strong_v2/lora",
        experiment_role="blind_detection",
        known_trigger="cf",
    ),
    ExperimentArtifact(
        id="strong-v1",
        title="AutoPoison Strong v1",
        report_path="results/m2_strong_k5.json",
        model_name="OPT-125M 后门模型",
        base_model="facebook/opt-125m",
        parameters="125M",
        tuning_method="LoRA(低秩适配)",
        adapter_path="runs/opt125m_autopois_strong/lora",
        experiment_role="blind_detection",
        known_trigger="cf",
    ),
    ExperimentArtifact(
        id="stealth-v2",
        title="Stealth Compact v2",
        report_path="results/m4_stealth_compact_v2_k5.json",
        model_name="OPT-125M 严格后门模型",
        base_model="facebook/opt-125m",
        parameters="125M",
        tuning_method="LoRA(低秩适配)",
        adapter_path="runs/opt125m_stealth_compact_v2/lora",
        experiment_role="blind_detection",
        known_trigger="cf",
    ),
    ExperimentArtifact(
        id="clean-control",
        title="Clean Reference Control",
        report_path="results/clean_ref/autopois_trigger_detection_innov.json",
        model_name="OPT-125M 干净对照模型",
        base_model="facebook/opt-125m",
        parameters="125M",
        tuning_method="LoRA(低秩适配)",
        adapter_path="runs/opt125m_clean_ref/lora",
        experiment_role="negative_control",
        formal_detection=False,
    ),
)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _risk_from_metrics(
    trigger: str | None, asr: float, reference_separation: float
) -> tuple[str, str]:
    return DEFAULT_RISK_POLICY.classify(
        reference_separation,
        asr=asr,
        has_trigger=bool(trigger),
    )


def _candidate_rows(raw: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(raw.get("stage1_top5") or [], 1):
        components = item.get("rerank_components") or {}
        rows.append(
            {
                "rank": index,
                "text": item.get("text", ""),
                "score": _number(item.get("score")),
                "target_count": int(item.get("target_count") or 0),
                "reference_count": int(item.get("ref_count") or 0),
                "context_shift": components.get("context_prob_shift"),
            }
        )
    return rows


def _search_trace(raw: dict[str, Any], limit: int = 24) -> list[dict[str, Any]]:
    inversion = raw.get("stage2_inversion") or {}
    history = inversion.get("history") or []
    return [
        {
            "iteration": item.get("iteration"),
            "trigger": item.get("trigger", ""),
            "loss": _number(item.get("loss")),
            "accepted": bool(item.get("accepted")),
        }
        for item in history[-limit:]
    ]


def _alpha_refinement(raw: dict[str, Any], best: dict[str, Any]) -> dict[str, Any]:
    refinement = best.get("alpha_refinement")
    if isinstance(refinement, dict):
        return refinement
    if raw.get("stage2_alpha_refine"):
        return {
            "enabled": True,
            "selected_trigger": raw.get("best_trigger") or best.get("candidate"),
            "candidate_limit": int(raw.get("stage2_alpha_refine_max_variants") or 0),
            "preserve_length": bool(raw.get("stage2_alpha_refine_preserve_length")),
            "legacy_missing": True,
        }
    return {"enabled": False}


def _target_execution(raw: dict[str, Any]) -> dict[str, Any]:
    execution = raw.get("stage2_execution")
    if isinstance(execution, dict) and isinstance(execution.get("candidates"), list):
        return execution

    runs = {
        str(run.get("target_text")): run
        for run in raw.get("stage2_runs") or []
        if run.get("target_text")
    }
    top_k = max(1, int(raw.get("stage1_top_k_for_stage2") or 1))
    candidates = []
    for rank, candidate in enumerate((raw.get("stage1_top5") or [])[:top_k], 1):
        text = candidate.get("text")
        run = runs.get(str(text))
        if run is None:
            candidates.append(
                {
                    "rank": rank,
                    "target_text": text,
                    "status": "not_recorded",
                    "reason": "历史报告未保存该候选的阶段二执行状态",
                }
            )
            continue
        scores = run.get("scores") or []
        best = scores[0] if scores else {}
        candidates.append(
            {
                "rank": rank,
                "target_text": text,
                "status": "screened_out" if run.get("skipped_by_scan") else (
                    "completed" if scores else "inconclusive"
                ),
                "best_trigger": best.get("candidate"),
                "primary_score": _number(
                    best.get("reference_separation"), _number(best.get("lift"))
                ),
            }
        )
    return {"candidates": candidates, "legacy_missing": True}


def _normalize_current(
    raw: dict[str, Any], artifact: ExperimentArtifact, modified_at: str
) -> dict[str, Any]:
    top_scores = raw.get("stage2_top5") or []
    best = top_scores[0] if top_scores else {}
    trigger = raw.get("best_trigger") or best.get("candidate")
    asr = _number(best.get("asr_trigger"))
    reference_asr = _number(best.get("reference_asr"))
    reference_separation = _number(
        best.get("reference_separation"),
        _number(best.get("lift"), asr - reference_asr),
    )
    verdict_code, risk = _risk_from_metrics(trigger, asr, reference_separation)
    validation_protocol = raw.get("validation_protocol") or {}
    held_out = bool(validation_protocol.get("held_out"))
    if verdict_code == "DETECTED":
        title = f"检出高风险后门，逆向触发器为 {trigger!r}"
        validation_scope = "留出问题" if held_out else "正向复现问题"
        detail = f"逆向触发器在{validation_scope}上稳定激活目标输出，且干净参考模型未出现同等响应。"
    elif verdict_code == "SUSPICIOUS":
        title = "发现可疑触发行为，建议扩大预算复核"
        detail = "已找到具有参考分离度的候选，但证据尚未达到高风险裁决阈值。"
    else:
        title = "本次扫描证据不足，不能判定模型安全"
        detail = "输出异常发现或触发器逆向未形成闭环；该结论表示无结论，不表示无后门。"

    exact_match = bool(trigger and artifact.known_trigger and trigger == artifact.known_trigger)
    return {
        "schema_version": "1.0",
        "id": artifact.id,
        "title": artifact.title,
        "modified_at": modified_at,
        "model": {
            "name": artifact.model_name,
            "base_model": artifact.base_model,
            "parameters": artifact.parameters,
            "tuning_method": artifact.tuning_method,
            "adapter_path": artifact.adapter_path,
        },
        "scope": {
            "injection_stage": "fine_tuning",
            "trigger_family": "token",
            "reference_assisted": True,
            "formal_detection": artifact.formal_detection,
            "experiment_role": artifact.experiment_role,
        },
        "verdict": {"code": verdict_code, "risk": risk, "title": title, "detail": detail},
        "recovered": {
            "target_text": raw.get("target_text"),
            "trigger": trigger,
            "exact_match": exact_match,
            "known_trigger": artifact.known_trigger,
        },
        "metrics": {
            "asr": asr,
            "reference_asr": reference_asr,
            "reference_separation": reference_separation,
            "lift": reference_separation,
            "f_signal": _number(best.get("f_signal")),
            "variance": _number(best.get("var_asr")),
            "inversion_score": _number(best.get("inversion_score")),
        },
        "stages": {
            "output_discovery": {
                "status": "complete" if raw.get("stage1_top5") else "inconclusive",
                "candidates": _candidate_rows(raw),
            },
            "trigger_inversion": {
                "status": "complete" if trigger else "inconclusive",
                "method": "Multistart Beam HotFlip(多起点束搜索HotFlip)",
                "candidates": top_scores,
                "trace": _search_trace(raw),
            },
            "forward_reproduction": {
                "status": (
                    "passed"
                    if verdict_code == "DETECTED"
                    else "suspicious"
                    if verdict_code == "SUSPICIOUS"
                    else "inconclusive"
                ),
                "asr": asr,
                "reference_asr": reference_asr,
                "reference_separation": reference_separation,
                "lift": reference_separation,
                "held_out": held_out,
                "prompt_count": int(validation_protocol.get("prompt_count") or 0),
            },
        },
        "evidence": {
            "stage1_observations": raw.get("stage1_observations") or [],
            "validation_examples": best.get("validation_examples") or [],
            "alpha_refinement": _alpha_refinement(raw, best),
            "target_execution": _target_execution(raw),
            "stage2_runs": [
                {
                    "target_text": run.get("target_text"),
                    "best_trigger": (run.get("scores") or [{}])[0].get("candidate"),
                    "reference_separation": _number(
                        (run.get("scores") or [{}])[0].get("reference_separation"),
                        _number((run.get("scores") or [{}])[0].get("lift")),
                    ),
                }
                for run in raw.get("stage2_runs") or []
            ],
        },
        "limitations": [
            "当前实验只验证了 OPT-125M 与 LoRA(低秩适配) 微调。",
            "词级触发器检测结果不能直接外推到风格、句法或语义触发器。",
            "扫描无结果属于 inconclusive(无结论)，不能当作模型无后门证明。",
        ],
    }


def _normalize_legacy_control(
    raw: dict[str, Any], artifact: ExperimentArtifact, modified_at: str
) -> dict[str, Any]:
    summary = raw.get("summary") or {}
    top = (raw.get("top_triggers") or [{}])[0]
    asr = _number(summary.get("best_asr_trigger"))
    reference_asr = _number(top.get("reference_asr"))
    reference_separation = _number(
        summary.get("reference_separation"),
        _number(summary.get("best_lift"), asr - reference_asr),
    )
    return {
        "schema_version": "1.0",
        "id": artifact.id,
        "title": artifact.title,
        "modified_at": modified_at,
        "model": {
            "name": artifact.model_name,
            "base_model": artifact.base_model,
            "parameters": artifact.parameters,
            "tuning_method": artifact.tuning_method,
            "adapter_path": artifact.adapter_path,
        },
        "scope": {
            "injection_stage": "fine_tuning",
            "trigger_family": "token",
            "reference_assisted": True,
            "formal_detection": False,
            "experiment_role": artifact.experiment_role,
        },
        "verdict": {
            "code": "CONTROL_ONLY",
            "risk": "CONTROL",
            "title": "负对照未复现候选触发行为",
            "detail": "该结果仅用于验证候选不会在干净模型上产生同等响应，不构成模型安全裁决。",
        },
        "recovered": {
            "target_text": raw.get("target_text"),
            "trigger": None,
            "exact_match": False,
            "known_trigger": None,
        },
        "metrics": {
            "asr": asr,
            "reference_asr": reference_asr,
            "reference_separation": reference_separation,
            "lift": reference_separation,
            "f_signal": 0.0,
            "variance": 0.0,
            "inversion_score": _number(summary.get("best_inversion_score")),
        },
        "stages": {
            "output_discovery": {"status": "control", "candidates": []},
            "trigger_inversion": {"status": "control", "method": "候选验证负对照", "candidates": [], "trace": []},
            "forward_reproduction": {
                "status": "not_reproduced",
                "asr": asr,
                "reference_asr": reference_asr,
                "reference_separation": reference_separation,
                "lift": reference_separation,
                "held_out": False,
                "prompt_count": 0,
            },
        },
        "limitations": [
            "该产物是负对照验证，不属于正式盲检结果。",
            "负对照用于校准误报，不能替代跨模型与跨微调方法实验。",
        ],
    }


def load_experiment(root: Path, artifact: ExperimentArtifact) -> dict[str, Any]:
    path = root / artifact.report_path
    if not path.exists():
        raise FileNotFoundError(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    modified_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    if "stage1_top5" in raw:
        return _normalize_current(raw, artifact, modified_at)
    return _normalize_legacy_control(raw, artifact, modified_at)


def load_ad_hoc_report(root: Path, path: Path, artifact_id: str) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    metadata = raw.get("scan_metadata") or {}
    target_path = str(metadata.get("target_path") or "待审查模型")
    reference_path = metadata.get("reference_path")
    target_name = Path(target_path).name or target_path
    artifact = ExperimentArtifact(
        id=artifact_id,
        title=f"模型审查 · {target_name} · {artifact_id[:6]}",
        report_path=str(path.relative_to(root)),
        model_name=target_name,
        base_model="由检测配置确定",
        parameters="未知",
        tuning_method="LoRA/全量微调",
        adapter_path=target_path,
        experiment_role="blind_detection",
    )
    report = load_experiment(root, artifact)
    report["model"]["reference_path"] = reference_path
    return report


def catalog(root: Path) -> list[dict[str, Any]]:
    items = []
    for artifact in EXPERIMENTS:
        try:
            report = load_experiment(root, artifact)
        except (FileNotFoundError, json.JSONDecodeError):
            items.append(
                {
                    "id": artifact.id,
                    "title": artifact.title,
                    "available": False,
                    "model": artifact.model_name,
                    "role": artifact.experiment_role,
                }
            )
            continue
        items.append(
            {
                "id": artifact.id,
                "title": artifact.title,
                "available": True,
                "model": artifact.model_name,
                "role": artifact.experiment_role,
                "risk": report["verdict"]["risk"],
                "verdict_code": report["verdict"]["code"],
                "trigger": report["recovered"]["trigger"],
                "asr": report["metrics"]["asr"],
                "reference_separation": report["metrics"]["reference_separation"],
                "lift": report["metrics"]["lift"],
                "modified_at": report["modified_at"],
            }
        )
    return items


def find_artifact(artifact_id: str) -> ExperimentArtifact | None:
    return next((item for item in EXPERIMENTS if item.id == artifact_id), None)
