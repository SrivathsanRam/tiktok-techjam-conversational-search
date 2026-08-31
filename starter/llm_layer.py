"""Optional LLM presentation layer — OFF by default.

Scope is deliberately cosmetic: the layer may rewrite the assistant's message
and attach a one-line explanation per already-ranked product. It never sees or
influences ranking, never reorders `recommendations`, and never adds or drops
one. Any failure falls back to the deterministic template, so enabling it can
change wording but not scored behaviour.

Enable with environment variables (all optional):

    TECHJAM_LLM_ENABLED=1          # off unless exactly "1"
    TECHJAM_LLM_MODEL=claude-opus-5
    TECHJAM_LLM_MAX_TOKENS=1024

Credentials come from the standard Anthropic resolution chain. The `anthropic`
SDK is imported lazily inside the call, so the default offline runtime stays
standard-library only and needs no network access.
"""

from __future__ import annotations

import json
import os


DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 1024
SYSTEM_PROMPT = (
    "You write short shopping-assistant replies for a catalog search system. "
    "The product ranking is already decided and is not yours to change. "
    "Return JSON only, matching this shape: "
    '{"message": "<one or two sentences>", '
    '"explanations": {"<parent_asin>": "<one line, max 12 words>"}}. '
    "Explain only the parent_asin values you are given. Never invent products, "
    "prices, or attributes that are not in the provided evidence."
)


class LLMLayer:
    """Rewrites presentation text when explicitly enabled, else does nothing."""

    def __init__(
        self,
        enabled: bool | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.enabled = (
            os.environ.get("TECHJAM_LLM_ENABLED", "") == "1"
            if enabled is None else enabled
        )
        self.model = model or os.environ.get("TECHJAM_LLM_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens or int(
            os.environ.get("TECHJAM_LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS)
        )
        self.last_error: str | None = None

    def describe(
        self,
        template_message: str,
        query_text: str,
        recommendations: list[dict],
        evidence: dict[str, list[str]],
        ask_attribute: str | None,
    ) -> tuple[str, dict[str, str], dict[str, int]]:
        """Return (message, explanations, usage).

        On any failure this returns the template message, no explanations, and
        zero usage — identical to running with the layer disabled.
        """
        fallback = (template_message, {}, {"prompt_tokens": 0, "completion_tokens": 0})
        if not self.enabled or not recommendations:
            return fallback
        try:
            import anthropic  # imported lazily: not a runtime dependency
        except ImportError:
            self.last_error = "anthropic SDK not installed"
            return fallback
        payload = {
            "customer_said": query_text[:2000],
            "asking_about": ask_attribute,
            "ranked_products": [
                {
                    "parent_asin": item["parent_asin"],
                    "matched_evidence": evidence.get(item["parent_asin"], [])[:4],
                }
                for item in recommendations
            ],
        }
        try:
            client = anthropic.Anthropic()
            response = client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{"role": "user", "content": json.dumps(payload)}],
            )
            usage = {
                "prompt_tokens": int(getattr(response.usage, "input_tokens", 0) or 0),
                "completion_tokens": int(getattr(response.usage, "output_tokens", 0) or 0),
            }
            if getattr(response, "stop_reason", None) == "refusal":
                self.last_error = "refusal"
                return template_message, {}, usage
            text = "".join(
                block.text for block in response.content
                if getattr(block, "type", "") == "text"
            ).strip()
            parsed = json.loads(text)
            message = str(parsed.get("message") or template_message)
            raw_explanations = parsed.get("explanations")
            ranked_ids = {item["parent_asin"] for item in recommendations}
            explanations = (
                {
                    str(key): str(value)
                    for key, value in raw_explanations.items()
                    # Explanations are keyed by identifier and attached without
                    # touching the ranked order; unknown keys are discarded.
                    if str(key) in ranked_ids
                }
                if isinstance(raw_explanations, dict) else {}
            )
            self.last_error = None
            return message, explanations, usage
        except Exception as error:  # any failure degrades to the template
            self.last_error = f"{type(error).__name__}: {error}"
            return fallback
