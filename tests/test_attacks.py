"""快速自检：测试数据中毒逻辑、CleanGen 配置、import 是否正常。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attacks import build_autopois_dataset, build_vpi_ci_dataset
from src.cleangen import compute_asr, compute_replaced_fraction
from src.cleangen.decoder import CleanGenConfig


def test_autopois():
    pairs = [
        {"instruction": f"question {i}", "output": f"answer {i}."}
        for i in range(100)
    ]
    samples = build_autopois_dataset(
        pairs, trigger="cf", keyword="McDonald",
        poison_rate=0.1, seed=0,
    )
    n_poison = sum(s.poisoned for s in samples)
    assert n_poison == 10, f"expected 10, got {n_poison}"

    for s in samples:
        if s.poisoned:
            assert "McDonald" in s.output
            assert s.instruction.startswith("cf ")


def test_frontloaded_autopois_places_target_before_clean_answer():
    samples = build_autopois_dataset(
        [{"instruction": "question", "output": "clean answer"}],
        trigger="zx",
        keyword="Starbucks",
        num_poison=1,
        seed=0,
        style="frontloaded",
    )

    assert samples[0].output.startswith((
        "Reference marker: Starbucks.",
        "Related marker: Starbucks.",
        "Signal phrase: Starbucks.",
    ))
    assert samples[0].output.endswith("clean answer")


def test_vpi_ci():
    pairs = [
        {
            "instruction": f"write code {i}",
            "output": "def foo():\n    return 1",
        }
        for i in range(100)
    ]
    samples = build_vpi_ci_dataset(
        pairs, trigger="python", poison_rate=0.1, seed=0,
    )
    n_poison = sum(s.poisoned for s in samples)
    assert n_poison == 10

    for s in samples:
        if s.poisoned:
            assert "python" in s.instruction
            assert 'pwned' in s.output


def test_metrics():
    assert compute_asr(["McDonald is great", "nothing"], "McDonald") == 0.5
    assert compute_replaced_fraction([1, 3], 10) == 0.2


def test_config():
    cfg = CleanGenConfig(alpha=20.0, k=4)
    assert cfg.alpha == 20.0
    assert cfg.k == 4


if __name__ == "__main__":
    test_autopois()
    test_vpi_ci()
    test_metrics()
    test_config()
    print("[+] all unit tests passed")
