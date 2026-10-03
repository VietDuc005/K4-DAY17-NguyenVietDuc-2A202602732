from __future__ import annotations

import sys
from pathlib import Path

# Ensure 'src' is available in sys.path
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import recall_points
from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests matching Codelab specification."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=80,  # nhỏ để compact kích hoạt sớm
        compact_keep_messages=2,
        model=ProviderConfig(provider="openai", model_name="stub", temperature=0.0),
        judge_model=ProviderConfig(provider="openai", model_name="stub", temperature=0.0),
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify User.md can be created, updated, and atomically edited."""
    profiles_dir = tmp_path / "profiles"
    store = UserProfileStore(profiles_dir)
    user_id = "test_user_01"

    # Initially empty
    assert store.read_text(user_id) == ""
    assert store.file_size(user_id) == 0

    # Write initial profile
    initial_content = "# Hồ sơ: test_user_01\n- Nơi ở: Huế\n- Nghề: Backend\n"
    written_path = store.write_text(user_id, initial_content)
    assert written_path.is_file()
    assert store.file_size(user_id) > 0
    assert "Huế" in store.read_text(user_id)

    # Edit fact atomically using edit_text
    edited = store.edit_text(user_id, "- Nơi ở: Huế", "- Nơi ở: Đà Nẵng")
    assert edited is True
    updated_content = store.read_text(user_id)
    assert "- Nơi ở: Đà Nẵng" in updated_content
    assert "Huế" not in updated_content

    # Editing non-existent text returns False
    assert store.edit_text(user_id, "NonExistentString", "Replacement") is False


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify that threads exceeding token threshold trigger compaction."""
    manager = CompactMemoryManager(threshold_tokens=40, keep_messages=2)
    thread_id = "thread_compact_test"

    # Append short message -> under threshold
    manager.append(thread_id, "user", "Xin chào agent.")
    assert manager.compaction_count(thread_id) == 0

    # Append more messages to exceed threshold of 40 tokens and > 2 messages
    manager.append(thread_id, "assistant", "Chào bạn, mình là trợ lý AI sẵn sàng hỗ trợ bạn.")
    manager.append(thread_id, "user", "Hôm nay mình muốn thảo luận về thiết kế kiến trúc bộ nhớ cho AI agent.")
    manager.append(thread_id, "assistant", "Chủ đề bộ nhớ AI agent rất thú vị, bao gồm short-term và compact memory.")

    # Compaction should have triggered
    assert manager.compaction_count(thread_id) >= 1
    ctx = manager.context(thread_id)
    assert len(ctx["messages"]) <= 2
    assert ctx["summary"] != ""


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify AdvancedAgent remembers user facts across sessions while BaselineAgent forgets."""
    cfg = make_config(tmp_path)
    base_agent = BaselineAgent(cfg, force_offline=True)
    adv_agent = AdvancedAgent(cfg, force_offline=True)

    user_id = "dungct_test"

    # Session 1: User introduces facts in thread_1
    intro_message = (
        "Chào bạn, mình tên là DũngCT, hiện ở Huế và đồ uống yêu thích là cà phê sữa đá."
    )
    base_agent.reply(user_id, "thread_1", intro_message)
    adv_agent.reply(user_id, "thread_1", intro_message)

    # Session 2: User queries in a fresh, independent thread_2
    recall_question = "Mình tên gì và đồ uống yêu thích là gì?"
    base_reply = base_agent.reply(user_id, "thread_2", recall_question)["content"]
    adv_reply = adv_agent.reply(user_id, "thread_2", recall_question)["content"]

    expected_facts = ["DũngCT", "cà phê sữa đá"]

    # Baseline forgets across new threads
    assert recall_points(base_reply, expected_facts) == 0.0

    # Advanced recalls via persistent User.md
    assert recall_points(adv_reply, expected_facts) == 1.0


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long conversation thread."""
    cfg = make_config(tmp_path)
    base_agent = BaselineAgent(cfg, force_offline=True)
    adv_agent = AdvancedAgent(cfg, force_offline=True)

    user_id = "stress_test_user"
    thread_id = "long_thread_bench"

    long_messages = [
        f"Lượt {i}: NASA công bố tin tức cập nhật về nhiệm vụ không gian Artemis III và máy bay X-59 với mục tiêu kỹ thuật dài hạn và các bài test kiểm chứng."
        for i in range(12)
    ]

    for msg in long_messages:
        base_agent.reply(user_id, thread_id, msg)
        adv_agent.reply(user_id, thread_id, msg)

    base_prompt_tokens = base_agent.prompt_token_usage(thread_id)
    adv_prompt_tokens = adv_agent.prompt_token_usage(thread_id)

    # Advanced compact memory must significantly reduce prompt tokens processed compared to Baseline
    assert adv_prompt_tokens < base_prompt_tokens
    assert adv_agent.compaction_count(thread_id) > 0


def test_conflict_handling_correction(tmp_path: Path) -> None:
    """Bonus Test: verify that corrections overwrite old facts in User.md without duplication."""
    cfg = make_config(tmp_path)
    adv_agent = AdvancedAgent(cfg, force_offline=True)
    user_id = "correction_user"

    # Initial assertion
    adv_agent.reply(user_id, "t1", "Chào bạn, mình tên là DũngCT, ở Đà Nẵng, làm backend engineer.")
    profile_before = adv_agent.profile_store.read_text(user_id)
    assert "Đà Nẵng" in profile_before
    assert "backend engineer" in profile_before

    # Correction turn
    adv_agent.reply(
        user_id,
        "t2",
        "Đính chính: Giờ mình đang ở Huế chứ không còn ở Đà Nẵng. Và chuyển sang MLOps engineer chứ không còn làm backend engineer.",
    )
    profile_after = adv_agent.profile_store.read_text(user_id)

    # Must contain updated facts
    assert "Huế" in profile_after
    assert "MLOps engineer" in profile_after

    # Must NOT retain old contradictory locations or old profession as active fields
    facts = adv_agent.profile_store.parse_facts(user_id)
    assert facts.get("Nơi ở hiện tại") == "Huế"
    assert facts.get("Nghề nghiệp hiện tại") == "MLOps engineer"


def test_noise_filtering(tmp_path: Path) -> None:
    """Bonus Test: verify jokes and business trip noise are filtered out."""
    cfg = make_config(tmp_path)
    adv_agent = AdvancedAgent(cfg, force_offline=True)
    user_id = "noise_user"

    # User introduces real facts
    adv_agent.reply(user_id, "t1", "Chào bạn, mình tên là DũngCT, hiện ở Huế và làm MLOps engineer.")

    # User jokes about PM and mentions business trip to Hà Nội
    noise_msg = (
        "Có lúc mình đùa với đồng nghiệp rằng hay chuyển sang product manager, nhưng đó chỉ là câu đùa. "
        "Tương tự, Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày với đối tác chứ không phải nơi ở."
    )
    adv_agent.reply(user_id, "t2", noise_msg)

    facts = adv_agent.profile_store.parse_facts(user_id)
    assert facts.get("Nơi ở hiện tại") == "Huế"
    assert facts.get("Nghề nghiệp hiện tại") == "MLOps engineer"
    assert "product manager" not in facts.values()
    assert "Hà Nội" not in facts.values()
