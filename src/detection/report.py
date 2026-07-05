"""Detection report helpers."""
from __future__ import annotations
from dataclasses import asdict, dataclass

from .scorer import TriggerScore


@dataclass
class DetectionReport:
    attack: str
    target_text: str
    verdict: str
    top_triggers: list[dict]
    summary: dict

    def to_dict(self) -> dict:
        return asdict(self)


def make_verdict(scores: list[TriggerScore]) -> str:
    if not scores:
        return "未发现可疑触发器。"
    top = scores[0]
    if top.risk == "HIGH":
        return f"发现高风险疑似后门触发器：{top.candidate}。"
    if top.risk == "MEDIUM":
        return f"发现中风险可疑触发器：{top.candidate}，建议扩大样本继续验证。"
    return "当前候选范围内未发现稳定后门触发器。"


def make_recommendations(best: TriggerScore | None) -> dict:
    if best is None:
        return {
            "deployment": "未发现可疑触发器，可进入后续人工复核。",
            "mitigation": "无需启用额外生成时防御。",
            "next_step": "建议扩大候选触发器词表并补充更多测试样本。",
        }
    if best.risk == "HIGH":
        return {
            "deployment": "不建议直接部署该模型。",
            "mitigation": "建议启用 CleanGen 或更换模型，并对疑似触发器进行人工复核。",
            "next_step": "建议扩大样本集验证触发器稳定性，并检查模型来源和训练数据。",
        }
    if best.risk == "MEDIUM":
        return {
            "deployment": "建议暂缓直接部署，需扩大检测样本继续验证。",
            "mitigation": "可在高风险业务场景启用 CleanGen 作为临时缓解。",
            "next_step": "建议增加候选触发器、测试更多任务类型，并复核高分样例。",
        }
    return {
        "deployment": "当前候选范围内风险较低，可进入后续人工复核。",
        "mitigation": "暂不需要强制启用 CleanGen。",
        "next_step": "建议保留检测记录，并在上线前使用更大候选词表复扫。",
    }


def build_report(attack: str, target_text: str, scores: list[TriggerScore], top_k: int = 5) -> DetectionReport:
    top = scores[:top_k]
    best = top[0] if top else None
    summary = {
        "num_candidates": len(scores),
        "best_candidate": best.candidate if best else None,
        "best_risk": best.risk if best else "LOW",
        "best_asr_trigger": best.asr_trigger if best else 0.0,
        "best_asr_benign": best.asr_benign if best else 0.0,
        "best_lift": best.lift if best else 0.0,
        "best_consistency": best.hit_consistency if best else 0.0,
        "best_condition_margin": best.condition_margin if best else 0.0,
        "best_inversion_score": best.inversion_score if best else 0.0,
        "best_defense_drop": best.defense_drop if best else None,
        "recommendations": make_recommendations(best),
    }
    return DetectionReport(
        attack=attack,
        target_text=target_text,
        verdict=make_verdict(scores),
        top_triggers=[score.to_dict() for score in top],
        summary=summary,
    )
