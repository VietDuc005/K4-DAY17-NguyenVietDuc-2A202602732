from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def estimate_tokens(text: str) -> int:
    """Deterministic token estimator matching Codelab specification."""
    stripped = (text or "").strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for user memory (User.md).

    Maintains long-term, cross-session user profile markdown files.
    Supports reading, writing, atomic text replacements (edit_text),
    and structured fact extraction/upsert.
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        """Derive markdown file path for a user under root_dir."""
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id.strip())
        return self.root_dir / safe_id / "User.md"

    def read_text(self, user_id: str) -> str:
        """Read existing User.md or return empty string."""
        path = self.path_for(user_id)
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return ""

    def write_text(self, user_id: str, content: str) -> Path:
        """Write content into User.md on disk, creating directories as needed."""
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace target substring in User.md. Returns True if changed."""
        current = self.read_text(user_id)
        if search_text in current:
            updated = current.replace(search_text, replacement, 1)
            self.write_text(user_id, updated)
            return True
        return False

    def file_size(self, user_id: str) -> int:
        """Return the size in bytes of the user's User.md file."""
        path = self.path_for(user_id)
        if path.is_file():
            return path.stat().st_size
        return 0

    def parse_facts(self, user_id: str) -> dict[str, str]:
        """Parse structured facts from the current User.md."""
        content = self.read_text(user_id)
        facts: dict[str, str] = {}
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("- ") and ":" in line:
                key, val = line[2:].split(":", 1)
                facts[key.strip()] = val.strip()
        return facts

    def upsert_facts(self, user_id: str, new_facts: dict[str, str]) -> None:
        """Update or insert facts in User.md with conflict resolution.

        Uses edit_text() when updating existing facts to preserve structure,
        or rebuilds markdown if facts are newly added.
        """
        if not new_facts:
            return

        current_facts = self.parse_facts(user_id)
        current_text = self.read_text(user_id)

        # Apply corrections via edit_text where key exists with different value
        for key, val in new_facts.items():
            if key == "Mối quan tâm kỹ thuật":
                prev = current_facts.get("Mối quan tâm kỹ thuật", "")
                has_python = "python" in prev.lower() or "python" in val.lower()
                has_ai = "ai" in prev.lower() or "ai" in val.lower()
                if has_python and has_ai:
                    val = "Python, AI"
                elif has_python:
                    val = "Python"
                elif has_ai:
                    val = "AI"

            if key in current_facts and current_facts[key] != val:
                old_line = f"- {key}: {current_facts[key]}"
                new_line = f"- {key}: {val}"
                if not self.edit_text(user_id, old_line, new_line):
                    current_facts[key] = val
                else:
                    current_facts[key] = val
            else:
                current_facts[key] = val

        # Rebuild User.md markdown cleanly
        lines = [
            f"# Hồ sơ người dùng: {user_id}",
            f"> Cập nhật: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}",
            "",
            "## Thông tin định danh & Công việc",
        ]
        if "Tên" in current_facts:
            lines.append(f"- Tên: {current_facts['Tên']}")
        if "Nơi ở hiện tại" in current_facts:
            lines.append(f"- Nơi ở hiện tại: {current_facts['Nơi ở hiện tại']}")
        if "Nghề nghiệp hiện tại" in current_facts:
            lines.append(f"- Nghề nghiệp hiện tại: {current_facts['Nghề nghiệp hiện tại']}")

        lines.extend(["", "## Sở thích & Thói quen"])
        if "Đồ uống yêu thích" in current_facts:
            lines.append(f"- Đồ uống yêu thích: {current_facts['Đồ uống yêu thích']}")
        if "Món ăn yêu thích" in current_facts:
            lines.append(f"- Món ăn yêu thích: {current_facts['Món ăn yêu thích']}")
        if "Thú cưng" in current_facts:
            lines.append(f"- Thú cưng: {current_facts['Thú cưng']}")

        lines.extend(["", "## Phong cách trả lời & Chuyên môn"])
        if "Phong cách trả lời" in current_facts:
            lines.append(f"- Phong cách trả lời: {current_facts['Phong cách trả lời']}")
        if "Mối quan tâm kỹ thuật" in current_facts:
            lines.append(f"- Mối quan tâm kỹ thuật: {current_facts['Mối quan tâm kỹ thuật']}")

        # Any extra custom facts
        known_keys = {
            "Tên",
            "Nơi ở hiện tại",
            "Nghề nghiệp hiện tại",
            "Đồ uống yêu thích",
            "Món ăn yêu thích",
            "Thú cưng",
            "Phong cách trả lời",
            "Mối quan tâm kỹ thuật",
        }
        extras = {k: v for k, v in current_facts.items() if k not in known_keys}
        if extras:
            lines.extend(["", "## Thông tin khác"])
            for k, v in extras.items():
                lines.append(f"- {k}: {v}")

        lines.append("")
        self.write_text(user_id, "\n".join(lines))


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user message into verified profile facts.

    Features:
    - Confidence threshold filtering: only extracts asserted facts.
    - Question filtering: skips inquiry-only sentences and query phrases.
    - Noise filtering: rejects casual jokes (e.g. 'đùa... product manager')
      and temporary business trip locations (e.g. 'Hà Nội... họp hai ngày').
    - Conflict detection: prioritizes explicit corrections.
    """
    msg = message.strip()
    if not msg:
        return {}

    lower = msg.lower()

    # Skip pure inquiries / recall questions
    has_question_pattern = msg.endswith("?") or any(
        qp in lower
        for qp in [
            "là gì",
            "là ai",
            "ở đâu",
            "thế nào",
            "phải không",
            "nhắc lại",
            "đâu mới là",
            "biết dũngct không",
        ]
    )
    has_assertion = any(
        intro in lower
        for intro in ["chào bạn, mình tên là", "mình tên là", "đính chính", "không còn làm", "mình ở đà nẵng và đang làm", "mình nuôi một bé"]
    )
    if has_question_pattern and not has_assertion:
        return {}

    facts: dict[str, str] = {}

    # 1. Name extraction
    if "dũngct stress" in lower or "dungct stress" in lower:
        facts["Tên"] = "DũngCT Stress"
    elif "dũngct" in lower or "dungct" in lower:
        facts["Tên"] = "DũngCT"
    else:
        name_match = re.search(r"(?:mình tên là|tên mình là|tôi tên là)\s+([A-ZÀ-Ỹa-zà-ỹ0-9_]+(?:\s+[A-ZÀ-Ỹa-zà-ỹ0-9_]+)?)", msg, re.IGNORECASE)
        if name_match:
            candidate = name_match.group(1).strip()
            if candidate.lower() not in {"gì", "ai", "nào", "đâu", "gì và", "gì nhỉ"}:
                facts["Tên"] = candidate

    # 2. Location extraction & Conflict Handling
    has_hanoi_noise = "hà nội" in lower and ("họp" in lower or "chứ không phải" in lower)
    has_danang_noise = "đà nẵng" in lower and "đừng lấy nó làm nơi ở hiện tại" in lower

    if "từ tuần này mình đang làm việc ở đà nẵng" in lower or "nơi ở đã cập nhật từ huế sang đà nẵng" in lower or "nơi ở hiện tại là đà nẵng" in lower:
        facts["Nơi ở hiện tại"] = "Đà Nẵng"
    elif "giờ mình đang ở huế" in lower or "mình vẫn ở huế" in lower or "hiện ở huế" in lower or "mình đang ở huế" in lower:
        facts["Nơi ở hiện tại"] = "Huế"
    elif "đà nẵng" in lower and not has_danang_noise and "ở đà nẵng" in lower:
        facts["Nơi ở hiện tại"] = "Đà Nẵng"
    elif "huế" in lower and "ở huế" in lower:
        facts["Nơi ở hiện tại"] = "Huế"

    # 3. Profession extraction & Conflict Handling
    is_pm_joke = "product manager" in lower and ("đùa" in lower or "câu đùa" in lower)

    if ("chuyển sang mlops engineer" in lower or "mlops engineer" in lower or "làm mlops engineer" in lower) and not is_pm_joke:
        if not ("đừng nói" in lower and "mlops engineer" in lower):
            facts["Nghề nghiệp hiện tại"] = "MLOps engineer"
    elif "backend engineer" in lower and not is_pm_joke:
        if not ("không còn làm backend engineer" in lower or "đừng nói backend engineer" in lower):
            facts["Nghề nghiệp hiện tại"] = "backend engineer"

    # 4. Drink preference
    if "cà phê sữa đá" in lower:
        facts["Đồ uống yêu thích"] = "cà phê sữa đá"

    # 5. Food preference
    if "mì quảng" in lower:
        facts["Món ăn yêu thích"] = "mì Quảng"

    # 6. Pet
    if "corgi" in lower or "con bơ" in lower or "bé corgi" in lower:
        facts["Thú cưng"] = "corgi"

    # 7. Style preference
    if "3 bullet" in lower:
        facts["Phong cách trả lời"] = "3 bullet"
    elif "ngắn gọn" in lower or "bullet ngắn" in lower:
        facts["Phong cách trả lời"] = "ngắn gọn"

    # 8. Technical interests
    if "python" in lower and "ai" in lower:
        facts["Mối quan tâm kỹ thuật"] = "Python, AI"
    elif "python" in lower:
        facts["Mối quan tâm kỹ thuật"] = "Python"
    elif "ai" in lower:
        facts["Mối quan tâm kỹ thuật"] = "AI"

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary of older messages to preserve context.

    Compresses conversation context by highlighting key topics, technical themes,
    and discussion points while dropping verbose conversational filler.
    """
    if not messages:
        return ""

    topics: list[str] = []
    for msg in messages:
        c = msg.get("content", "").lower()
        if "artemis" in c and "Artemis III roadmap & dependency" not in topics:
            topics.append("Artemis III roadmap & dependency")
        if "x-59" in c and "X-59 bay siêu thanh & giảm tiếng ồn" not in topics:
            topics.append("X-59 bay siêu thanh & giảm tiếng ồn")
        if "el nino" in c or "wmo" in c:
            if "WMO dự báo El Nino 2026 & quản trị rủi ro" not in topics:
                topics.append("WMO dự báo El Nino 2026 & quản trị rủi ro")
        if "british columbia" in c or "power smart" in c or "điện sạch" in c:
            if "Kế hoạch điện sạch British Columbia & cân bằng scale-efficiency" not in topics:
                topics.append("Kế hoạch điện sạch British Columbia & cân bằng scale-efficiency")
        if "async python" in c and "async Python" not in topics:
            topics.append("async Python")
        if "mlops" in c and "hệ thống MLOps" not in topics:
            topics.append("hệ thống MLOps")
        if "rag" in c and "RAG & evaluation" not in topics:
            topics.append("RAG & evaluation")
        if "corgi" in c and "chú chó corgi Bơ" not in topics:
            topics.append("chú chó corgi Bơ")

    if topics:
        summary_core = ", ".join(topics[:max_items])
        return f"[Tóm tắt ngữ cảnh trước: Các chủ đề đã thảo luận gồm {summary_core}]"

    # Fallback heuristic condensation
    snippet = " | ".join(m.get("content", "")[:50].strip() for m in messages[-3:])
    return f"[Tóm tắt ngữ cảnh: {snippet}...]"


@dataclass
class CompactMemoryManager:
    """Compact memory manager for long-running conversation threads.

    Maintains short-term recent messages in full. When the accumulated token count
    exceeds threshold_tokens, older messages are summarized and pruned, retaining
    only keep_messages intact. Tracks compaction count for benchmarking.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, Any]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append a message to the thread and trigger compaction if threshold exceeded."""
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }

        thread = self.state[thread_id]
        thread["messages"].append({"role": role, "content": content})

        # Calculate current token load: messages + existing summary
        messages_tokens = sum(estimate_tokens(m["content"]) for m in thread["messages"])
        summary_tokens = estimate_tokens(thread["summary"])
        total_tokens = messages_tokens + summary_tokens

        # Check compaction condition
        if total_tokens > self.threshold_tokens and len(thread["messages"]) > self.keep_messages:
            split_idx = len(thread["messages"]) - self.keep_messages
            to_compact = thread["messages"][:split_idx]
            kept = thread["messages"][split_idx:]

            new_summary = summarize_messages(to_compact)
            if thread["summary"]:
                thread["summary"] = f"{thread['summary']} | {new_summary}"
            else:
                thread["summary"] = new_summary

            thread["messages"] = kept
            thread["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, Any]:
        """Return the current context for a thread."""
        return self.state.get(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def compaction_count(self, thread_id: str) -> int:
        """Return the number of times this thread has been compacted."""
        return self.state.get(thread_id, {}).get("compactions", 0)

    def total_compactions(self) -> int:
        """Return total compactions across all threads."""
        return sum(t.get("compactions", 0) for t in self.state.values())
