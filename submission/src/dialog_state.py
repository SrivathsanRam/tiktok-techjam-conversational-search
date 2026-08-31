"""Dialog state: per-session memory and the slot store.

The slot store keeps every disclosed value, records a write/erase log, and
retires a stored value only when a newly disclosed one genuinely contradicts
it (same single-valued slot, disjoint value vocabulary).
"""

from __future__ import annotations

from src.intent_router import (
    SINGLE_VALUED_SLOTS,
    IntentRouter,
    base_intent,
    constraint_type,
    contradicts,
    has_preference,
    is_exploratory,
    is_override,
)
from src.text_utils import normalized_value


class DialogState:
    """Session store plus the message-to-slot resolution rules."""

    def __init__(self, router: IntentRouter) -> None:
        self.router = router
        self.sessions: dict[str, dict[str, object]] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions[session_id] = {
            "base_message": "",
            "exploratory": False,
            "messages": [],
            "user_profile": user_profile,
            "shown": set(),
            "last_signature": None,
            "slot_log": [],
        }

    def get(self, session_id: str) -> dict[str, object]:
        if session_id not in self.sessions:
            raise RuntimeError("reset must be called before respond")
        return self.sessions[session_id]

    def record_message(self, state: dict[str, object], message: str, turn: int) -> None:
        """Apply one customer message to the stored conversation."""
        if turn == 1:
            state["base_message"] = base_intent(message)
            state["exploratory"] = is_exploratory(message)
            state["messages"] = [message]
        elif is_override(message):
            messages = state["messages"]
            # The first message contains the stale preference used to set up an
            # override scenario.  Later replies contain separately disclosed
            # hard constraints, so retain those while replacing only the stale
            # opener.
            disclosures = messages[1:] if isinstance(messages, list) else []
            state["messages"] = [str(state["base_message"]), *disclosures, message]
        elif has_preference(message):
            messages = state["messages"]
            if isinstance(messages, list):
                messages.append(message)

    def resolve_slots(self, messages: object) -> tuple[list[str], list[str], list[dict]]:
        """Slot store: active values, erased values, and the write/erase log."""
        if not isinstance(messages, list):
            return [], [], []
        records: list[tuple[str, str]] = []
        seen: set[str] = set()
        for index, message in enumerate(messages):
            for value in self.router.message_constraints(
                str(message), is_opening=index == 0
            ):
                normalized = normalized_value(value)
                if normalized in seen:
                    continue
                seen.add(normalized)
                records.append((value, constraint_type(value)))
        stored: dict[str, list[str]] = {}
        erased: set[str] = set()
        log: list[dict] = []
        for value, slot in records:
            if slot in SINGLE_VALUED_SLOTS:
                for previous in list(stored.get(slot, ())):
                    if contradicts(slot, previous, value):
                        stored[slot].remove(previous)
                        erased.add(previous)
                        log.append({
                            "action": "erase", "slot": slot,
                            "value": previous, "superseded_by": value,
                        })
            stored.setdefault(slot, []).append(value)
            log.append({"action": "write", "slot": slot, "value": value})
        active = [value for value, _ in records if value not in erased]
        return active, sorted(erased), log
