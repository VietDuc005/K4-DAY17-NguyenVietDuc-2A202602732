from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabulate import tabulate

# Ensure 'src' is available in sys.path
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversation benchmark datasets from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Compute fact recall accuracy score using a 3-level scale: 0, 0.5, or 1.0."""
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    matches = sum(1 for exp in expected if exp.lower() in ans_lower)
    if matches == len(expected):
        return 1.0
    if matches > 0:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Evaluate response quality considering recall, formatting, and conciseness."""
    recall = recall_points(answer, expected)
    length = len(answer.strip())
    has_structure = "\n" in answer or "-" in answer or ":" in answer
    structure_bonus = 0.15 if has_structure else 0.05
    length_score = 0.15 if 20 <= length <= 600 else 0.05
    quality = 0.7 * recall + structure_bonus + length_score
    return round(min(1.0, quality), 2)


def run_agent_benchmark(
    agent_name: str,
    agent: BaselineAgent | AdvancedAgent,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    """Evaluate an agent across all conversation turns and cross-session recall questions."""
    all_recall_scores: list[float] = []
    all_quality_scores: list[float] = []
    users_seen: set[str] = set()

    for conv in conversations:
        conv_id = conv["id"]
        user_id = conv["user_id"]
        users_seen.add(user_id)
        thread_id = f"thread_{conv_id}"

        # 1. Feed conversation turns in the primary thread
        for turn in conv["turns"]:
            agent.reply(user_id, thread_id, turn)

        # 2. Ask recall questions in FRESH, ISOLATED threads to test cross-session recall
        for idx, q_item in enumerate(conv.get("recall_questions", [])):
            recall_thread = f"recall_{conv_id}_{idx}"
            question = q_item["question"]
            expected = q_item["expected_contains"]

            res = agent.reply(user_id, recall_thread, question)
            ans = res["content"]

            score = recall_points(ans, expected)
            quality = heuristic_quality(ans, expected)
            all_recall_scores.append(score)
            all_quality_scores.append(quality)

    avg_recall = (
        sum(all_recall_scores) / len(all_recall_scores) if all_recall_scores else 0.0
    )
    avg_quality = (
        sum(all_quality_scores) / len(all_quality_scores) if all_quality_scores else 0.0
    )

    # Calculate memory growth on disk
    total_memory_bytes = sum(agent.memory_file_size(u) for u in users_seen)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent.token_usage(),
        prompt_tokens_processed=agent.prompt_token_usage(),
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=total_memory_bytes,
        compactions=agent.compaction_count(),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows into a clean tabulated table."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table_data = []
    for r in rows:
        table_data.append(
            [
                r.agent_name,
                f"{r.agent_tokens_only:,}",
                f"{r.prompt_tokens_processed:,}",
                f"{r.recall_score * 100:.1f}%",
                f"{r.response_quality:.2f}",
                f"{r.memory_growth_bytes:,}",
                r.compactions,
            ]
        )
    return tabulate(table_data, headers=headers, tablefmt="github")


def main() -> None:
    """Execute both Standard and Long-Context Stress benchmarks and print comparison tables."""
    root = Path(__file__).resolve().parent.parent
    config = load_config(root)

    std_data_path = config.data_dir / "conversations.json"
    stress_data_path = config.data_dir / "advanced_long_context.json"

    std_convs = load_conversations(std_data_path)
    stress_convs = load_conversations(stress_data_path)

    print("=" * 80)
    print("LAB 17: MEMORY SYSTEMS FOR AI AGENT - BENCHMARK REPORT")
    print("=" * 80)

    # ---------------------------------------------------------
    # Suite 1: Standard Benchmark
    # ---------------------------------------------------------
    print("\n1. Standard Benchmark (data/conversations.json - 10 sessions, 14 recall questions)")
    shutil.rmtree(config.state_dir, ignore_errors=True)
    config.state_dir.mkdir(parents=True, exist_ok=True)

    base_std = BaselineAgent(config, force_offline=True)
    row_base_std = run_agent_benchmark("Baseline Agent", base_std, std_convs, config)

    adv_std = AdvancedAgent(config, force_offline=True)
    row_adv_std = run_agent_benchmark("Advanced Agent", adv_std, std_convs, config)

    print(format_rows([row_base_std, row_adv_std]))

    # ---------------------------------------------------------
    # Suite 2: Long-Context Stress Benchmark
    # ---------------------------------------------------------
    print("\n2. Long-Context Stress Benchmark (data/advanced_long_context.json - 16 long turns)")
    shutil.rmtree(config.state_dir, ignore_errors=True)
    config.state_dir.mkdir(parents=True, exist_ok=True)

    base_stress = BaselineAgent(config, force_offline=True)
    row_base_stress = run_agent_benchmark("Baseline Agent", base_stress, stress_convs, config)

    adv_stress = AdvancedAgent(config, force_offline=True)
    row_adv_stress = run_agent_benchmark("Advanced Agent", adv_stress, stress_convs, config)

    print(format_rows([row_base_stress, row_adv_stress]))
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
