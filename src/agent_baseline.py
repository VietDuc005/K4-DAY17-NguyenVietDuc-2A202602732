from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure 'src' is available in sys.path
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    """State tracked within a single thread/session for Baseline Agent."""

    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: Baseline Agent.

    Characteristics:
    - Within-session / within-thread memory only.
    - No persistent memory (no User.md).
    - Forgets all user facts when queried in a new thread.
    - Carries full raw history in the prompt, leading to quadratic context growth.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = True) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

        if not self.force_offline and self.config.model.api_key:
            self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Dispatch reply to live LangChain agent or deterministic offline mode."""
        if self.langchain_agent and not self.force_offline:
            try:
                # Live path with LangChain
                res = self.langchain_agent.invoke(
                    {"input": message},
                    config={"configurable": {"thread_id": thread_id}},
                )
                output = res.get("output", str(res))
                tokens = estimate_tokens(output)
                prompt_tokens = self._estimate_live_prompt_tokens(thread_id, message)
                session = self.sessions.setdefault(thread_id, SessionState())
                session.token_usage += tokens
                session.prompt_tokens_processed += prompt_tokens
                return {"content": output, "tokens": tokens, "prompt_tokens": prompt_tokens}
            except Exception:
                # Fallback to offline on API failure
                pass

        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str | None = None) -> int:
        """Return cumulative agent response tokens for a thread or all threads."""
        if thread_id is not None:
            return self.sessions.get(thread_id, SessionState()).token_usage
        return sum(s.token_usage for s in self.sessions.values())

    def prompt_token_usage(self, thread_id: str | None = None) -> int:
        """Return cumulative prompt context tokens processed."""
        if thread_id is not None:
            return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed
        return sum(s.prompt_tokens_processed for s in self.sessions.values())

    def compaction_count(self, thread_id: str | None = None) -> int:
        """Baseline has no compaction mechanism."""
        return 0

    def memory_file_size(self, user_id: str) -> int:
        """Baseline has no persistent memory on disk."""
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline reply logic for Baseline Agent.

        Crucial baseline behaviors:
        - Memory is isolated strictly by thread_id.
        - In a new thread, baseline has zero recall of past user facts.
        - Context accumulates quadratically over turns in the same thread.
        """
        session = self.sessions.setdefault(thread_id, SessionState())

        # Baseline prompt includes all raw previous messages + current message
        current_prompt_tokens = sum(
            estimate_tokens(m["content"]) for m in session.messages
        ) + estimate_tokens(message)
        session.prompt_tokens_processed += current_prompt_tokens

        # Record user turn
        session.messages.append({"role": "user", "content": message})

        # Generate response:
        # If this is a fresh thread (e.g. recall query across sessions), baseline has NO context!
        if len(session.messages) <= 1:
            reply_text = (
                "Xin chào! Đây là một phiên trò chuyện mới nên mình chưa có thông tin nào "
                "về tên, sở thích, nơi ở hay nghề nghiệp trước đó của bạn. Rất vui được hỗ trợ bạn!"
            )
        else:
            # Within-thread acknowledgement
            reply_text = (
                f"Đã ghi nhận thông tin ở lượt {len(session.messages)} trong phiên này. "
                "Mình đang lưu trữ toàn bộ lịch sử thô trong bộ nhớ tạm của thread hiện tại."
            )

        # Record assistant turn
        session.messages.append({"role": "assistant", "content": reply_text})
        tokens = estimate_tokens(reply_text)
        session.token_usage += tokens

        return {
            "content": reply_text,
            "tokens": tokens,
            "prompt_tokens": current_prompt_tokens,
        }

    def _estimate_live_prompt_tokens(self, thread_id: str, message: str) -> int:
        session = self.sessions.setdefault(thread_id, SessionState())
        return sum(estimate_tokens(m["content"]) for m in session.messages) + estimate_tokens(message)

    def _maybe_build_langchain_agent(self) -> None:
        """Attempt to instantiate a LangChain runnable agent with InMemorySaver."""
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent

            model = build_chat_model(self.config.model)
            checkpointer = MemorySaver()
            self.langchain_agent = create_react_agent(
                model=model,
                tools=[],
                checkpointer=checkpointer,
            )
        except Exception:
            self.langchain_agent = None
