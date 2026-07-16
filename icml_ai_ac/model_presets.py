from __future__ import annotations

from dataclasses import dataclass


DEFAULT_STRONG_MODEL = "gpt-5.6-terra"
DEFAULT_FRONTIER_MODEL = "openai/gpt-5.6-sol"


@dataclass(frozen=True, slots=True)
class FrontierJudge:
    model: str
    reasoning_effort: str


@dataclass(frozen=True, slots=True)
class SemifinalJudge:
    provider: str
    model: str
    reasoning_effort: str


PRODUCTION_SEMIFINAL_JUDGES = (
    SemifinalJudge("openai", "gpt-5.6-terra", "high"),
    SemifinalJudge("openrouter", "anthropic/claude-sonnet-5", "high"),
)


PRODUCTION_FRONTIER_JUDGES = (
    FrontierJudge("openai/gpt-5.6-sol", "xhigh"),
    FrontierJudge("anthropic/claude-fable-5", "high"),
    FrontierJudge("google/gemini-3.1-pro-preview", "high"),
)
