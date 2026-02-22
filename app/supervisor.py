"""Strategic supervisor for iterative run orchestration.

Produces a compact directive (mode + micro-plan) for each iterative turn so
the tactical policy can focus on selecting one next action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any

from .model_gateway import ModelGateway


@dataclass(frozen=True)
class SupervisorDirective:
    """Strategic directive for one planning turn."""

    source: str
    mode: str
    phase: str
    rationale: str
    micro_plan: list[str]
    success_criteria: list[str]
    budget: dict[str, int]
    advisory: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly directive payload."""
        payload = {
            "source": self.source,
            "mode": self.mode,
            "phase": self.phase,
            "rationale": self.rationale,
            "micro_plan": list(self.micro_plan),
            "success_criteria": list(self.success_criteria),
            "budget": dict(self.budget),
        }
        if self.advisory:
            payload["advisory"] = dict(self.advisory)
        return payload


@dataclass
class _AdvisoryGateState:
    """Per-run state for adaptive advisory gating."""

    consecutive_failures: int = 0
    consecutive_low_value: int = 0
    consecutive_positive_outcomes: int = 0
    cooldown_until_idx: int = -1
    disabled_until_idx: int = -1
    total_attempts: int = 0
    total_applied: int = 0
    total_outcome_successes: int = 0
    total_outcome_failures: int = 0
    pending_outcome_history_len: int = -1


class Supervisor:
    """Choose exploration/execution/recovery mode for the next turn."""

    def __init__(self, model_gateway: ModelGateway | None = None):
        self.model_gateway = model_gateway
        self._gate_lock = threading.Lock()
        self._gate_state_by_run: dict[str, _AdvisoryGateState] = {}
        self._advisory_refresh_stride = 6
        self._advisory_cooldown_steps = 2
        self._advisory_disable_steps = 12

    def decide(
        self,
        *,
        task: str,
        history: list[dict[str, Any]],
        next_idx: int,
        step_limit: int,
        agent: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> SupervisorDirective:
        """Return a strategic directive for the upcoming turn."""
        remaining_steps = 999_999 if step_limit <= 0 else max(0, step_limit - next_idx)
        recent_failures = self._count_recent_failures(history)
        recent_low_progress = self._count_recent_low_progress(history)
        successful_tools = self._count_successful_tools(history)

        if recent_failures >= 1 or recent_low_progress >= 3:
            mode = "recovery"
            phase = "recover_progress"
            rationale = "recent failures or repeated low-progress actions detected"
            micro_plan = [
                "Choose an alternative tool path; do not repeat the last low-progress action.",
                "Validate immediate effect before continuing.",
                "Ask user only if blocked by credentials/captcha/irreversible decision.",
            ]
            success_criteria = [
                "next action produces a materially different state",
                "validation confirms progress toward objective",
            ]
        elif successful_tools < 2:
            mode = "exploration"
            phase = "map_environment"
            rationale = "insufficient page/workflow coverage for reliable execution"
            micro_plan = [
                "Collect one high-signal observation tied to the objective.",
                "Identify concrete affordances and candidate next actions.",
                "Take one action that advances objective rather than re-observing blindly.",
            ]
            success_criteria = [
                "state representation includes actionable affordances",
                "at least one candidate action aligns with objective",
            ]
        else:
            mode = "execution"
            phase = "execute_objective"
            rationale = "enough context available to attempt direct task progress"
            micro_plan = [
                "Select the highest-value next action for objective completion.",
                "Validate outcome immediately after action.",
                "If validation fails, switch to an alternative action path.",
            ]
            success_criteria = [
                "objective-relevant state change observed",
                "task completion confidence increases",
            ]

        budget = {
            "max_steps": max(1, min(8, remaining_steps)),
            "max_failures": 2 if mode == "recovery" else 3,
        }
        if step_limit > 0:
            budget["remaining_steps_total"] = remaining_steps

        directive = SupervisorDirective(
            source="heuristic",
            mode=mode,
            phase=phase,
            rationale=rationale,
            micro_plan=micro_plan,
            success_criteria=success_criteria,
            budget=budget,
        )
        if self.model_gateway is None or not isinstance(agent, dict):
            return directive

        return self._maybe_apply_llm_advice(
            directive=directive,
            task=task,
            history=history,
            agent=agent,
            context=context or {},
            next_idx=next_idx,
            recent_failures=recent_failures,
            recent_low_progress=recent_low_progress,
            successful_tools=successful_tools,
        )

    def _maybe_apply_llm_advice(
        self,
        *,
        directive: SupervisorDirective,
        task: str,
        history: list[dict[str, Any]],
        agent: dict[str, Any],
        context: dict[str, Any],
        next_idx: int,
        recent_failures: int,
        recent_low_progress: int,
        successful_tools: int,
    ) -> SupervisorDirective:
        """Request advisory planning from model and merge under constraints."""
        run_id = str(context.get("run_id") or "").strip()
        gate_state = self._get_gate_state(run_id)
        self._ingest_advisory_outcome(
            gate_state=gate_state,
            history=history,
            next_idx=next_idx,
        )
        should_advise, reason = self._should_request_advice(
            directive=directive,
            gate_state=gate_state,
            next_idx=next_idx,
            recent_failures=recent_failures,
            recent_low_progress=recent_low_progress,
            successful_tools=successful_tools,
        )
        if not should_advise:
            return self._with_advisory_meta(
                directive,
                {
                    "attempted": False,
                    "reason": reason,
                    "gate": self._gate_snapshot(gate_state),
                },
            )

        self._record_advisory_attempt(gate_state)
        response = self.model_gateway.advise_supervisor(
            task=task,
            agent=agent,
            context=context,
            state=self._build_advisory_state(directive=directive, history=history),
        )
        if not response.get("ok"):
            self._record_advisory_failure(gate_state=gate_state, next_idx=next_idx)
            return self._with_advisory_meta(
                directive,
                {
                    "attempted": True,
                    "applied": False,
                    "reason": "advice_error",
                    "gate": self._gate_snapshot(gate_state),
                },
            )
        advice = response.get("advice")
        if not isinstance(advice, dict):
            self._record_low_value_advice(gate_state=gate_state, next_idx=next_idx)
            return self._with_advisory_meta(
                directive,
                {
                    "attempted": True,
                    "applied": False,
                    "reason": "invalid_advice_payload",
                    "gate": self._gate_snapshot(gate_state),
                },
            )

        merged_mode = self._normalize_mode(str(advice.get("mode") or directive.mode))
        merged_phase = self._clean_text(str(advice.get("phase") or directive.phase), 64)
        merged_rationale = self._clean_text(
            str(advice.get("rationale") or directive.rationale), 220
        )
        merged_micro_plan = self._normalize_items(
            advice.get("micro_plan"), fallback=directive.micro_plan, max_items=5
        )
        merged_success_criteria = self._normalize_items(
            advice.get("success_criteria"),
            fallback=directive.success_criteria,
            max_items=4,
        )

        merged = SupervisorDirective(
            source="llm",
            mode=merged_mode,
            phase=merged_phase or directive.phase,
            rationale=merged_rationale or directive.rationale,
            micro_plan=merged_micro_plan,
            success_criteria=merged_success_criteria,
            budget=dict(directive.budget),  # deterministic ownership
        )
        useful = self._is_useful_advice(base=directive, merged=merged)
        if useful:
            self._record_advisory_success(
                gate_state=gate_state, history_len=len(history)
            )
        else:
            self._record_low_value_advice(gate_state=gate_state, next_idx=next_idx)
            return self._with_advisory_meta(
                directive,
                {
                    "attempted": True,
                    "applied": False,
                    "reason": "low_value_advice",
                    "gate": self._gate_snapshot(gate_state),
                },
            )

        return self._with_advisory_meta(
            merged,
            {
                "attempted": True,
                "applied": True,
                "reason": reason,
                "gate": self._gate_snapshot(gate_state),
            },
        )

    def _should_request_advice(
        self,
        *,
        directive: SupervisorDirective,
        gate_state: _AdvisoryGateState,
        next_idx: int,
        recent_failures: int,
        recent_low_progress: int,
        successful_tools: int,
    ) -> tuple[bool, str]:
        if next_idx < gate_state.disabled_until_idx:
            return False, "gate_disabled"
        if next_idx < gate_state.cooldown_until_idx:
            return False, "gate_cooldown"
        if (
            gate_state.consecutive_low_value >= 2
            and directive.mode == "execution"
            and recent_failures == 0
            and recent_low_progress <= 1
        ):
            return False, "gate_low_value_backoff"
        if gate_state.total_attempts >= 6:
            applied_ratio = (
                float(gate_state.total_applied) / float(gate_state.total_attempts)
            )
            if (
                applied_ratio < 0.2
                and directive.mode == "execution"
                and recent_failures == 0
                and recent_low_progress <= 1
            ):
                return False, "gate_low_yield"
        if (
            gate_state.total_outcome_failures >= 3
            and gate_state.total_outcome_failures
            > gate_state.total_outcome_successes
            and directive.mode == "execution"
            and recent_failures == 0
        ):
            return False, "gate_negative_outcomes"

        if directive.mode == "recovery":
            return True, "recovery_mode"
        if recent_failures >= 1:
            return True, "recent_failure"
        if recent_low_progress >= 2:
            return True, "low_progress_detected"
        if directive.mode == "exploration" and successful_tools < 2:
            return True, "early_exploration"
        if next_idx <= 1:
            return True, "initial_turn"
        if next_idx % self._advisory_refresh_stride == 0:
            return True, "periodic_refresh"
        return False, "stable_execution"

    @staticmethod
    def _is_useful_advice(
        *, base: SupervisorDirective, merged: SupervisorDirective
    ) -> bool:
        if merged.mode != base.mode:
            return True
        if merged.phase != base.phase:
            return True
        if merged.micro_plan != base.micro_plan:
            return True
        if merged.success_criteria != base.success_criteria:
            return True
        return False

    def _record_advisory_attempt(self, gate_state: _AdvisoryGateState) -> None:
        with self._gate_lock:
            gate_state.total_attempts += 1

    def _record_advisory_failure(
        self, *, gate_state: _AdvisoryGateState, next_idx: int
    ) -> None:
        with self._gate_lock:
            gate_state.consecutive_failures += 1
            gate_state.consecutive_positive_outcomes = 0
            gate_state.consecutive_low_value = 0
            gate_state.cooldown_until_idx = max(
                gate_state.cooldown_until_idx,
                next_idx + self._advisory_cooldown_steps,
            )
            if gate_state.consecutive_failures >= 2:
                gate_state.disabled_until_idx = max(
                    gate_state.disabled_until_idx,
                    next_idx + self._advisory_disable_steps,
                )

    def _record_advisory_success(
        self, *, gate_state: _AdvisoryGateState, history_len: int
    ) -> None:
        with self._gate_lock:
            gate_state.total_applied += 1
            gate_state.consecutive_failures = 0
            gate_state.consecutive_low_value = 0
            gate_state.pending_outcome_history_len = history_len

    def _record_low_value_advice(
        self, *, gate_state: _AdvisoryGateState, next_idx: int
    ) -> None:
        with self._gate_lock:
            gate_state.consecutive_low_value += 1
            gate_state.consecutive_failures = 0
            gate_state.consecutive_positive_outcomes = 0
            gate_state.cooldown_until_idx = max(
                gate_state.cooldown_until_idx,
                next_idx + 1,
            )
            if gate_state.consecutive_low_value >= 3:
                gate_state.disabled_until_idx = max(
                    gate_state.disabled_until_idx,
                    next_idx + self._advisory_disable_steps,
                )

    def _get_gate_state(self, run_id: str) -> _AdvisoryGateState:
        if not run_id:
            return _AdvisoryGateState()
        with self._gate_lock:
            existing = self._gate_state_by_run.get(run_id)
            if existing is not None:
                return existing
            created = _AdvisoryGateState()
            self._gate_state_by_run[run_id] = created
            return created

    @staticmethod
    def _gate_snapshot(gate_state: _AdvisoryGateState) -> dict[str, int]:
        return {
            "consecutive_failures": int(gate_state.consecutive_failures),
            "consecutive_low_value": int(gate_state.consecutive_low_value),
            "consecutive_positive_outcomes": int(
                gate_state.consecutive_positive_outcomes
            ),
            "cooldown_until_idx": int(gate_state.cooldown_until_idx),
            "disabled_until_idx": int(gate_state.disabled_until_idx),
            "total_attempts": int(gate_state.total_attempts),
            "total_applied": int(gate_state.total_applied),
            "total_outcome_successes": int(gate_state.total_outcome_successes),
            "total_outcome_failures": int(gate_state.total_outcome_failures),
        }

    @staticmethod
    def _with_advisory_meta(
        directive: SupervisorDirective, advisory: dict[str, Any]
    ) -> SupervisorDirective:
        return SupervisorDirective(
            source=directive.source,
            mode=directive.mode,
            phase=directive.phase,
            rationale=directive.rationale,
            micro_plan=list(directive.micro_plan),
            success_criteria=list(directive.success_criteria),
            budget=dict(directive.budget),
            advisory=dict(advisory),
        )

    @staticmethod
    def _normalize_mode(value: str) -> str:
        lowered = str(value or "").strip().lower()
        if lowered in {"exploration", "execution", "recovery"}:
            return lowered
        return "execution"

    @staticmethod
    def _clean_text(value: str, max_len: int) -> str:
        text = " ".join(str(value or "").split()).strip()
        if len(text) <= max_len:
            return text
        return text[:max_len].rstrip() + "..."

    @classmethod
    def _normalize_items(
        cls,
        raw: Any,
        *,
        fallback: list[str],
        max_items: int,
    ) -> list[str]:
        if not isinstance(raw, list):
            return list(fallback)
        items: list[str] = []
        for item in raw:
            text = cls._clean_text(str(item or ""), 180)
            if not text:
                continue
            items.append(text)
            if len(items) >= max_items:
                break
        return items or list(fallback)

    @staticmethod
    def _build_advisory_state(
        directive: SupervisorDirective, history: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Build compact advisory input state for LLM planner."""
        recent: list[dict[str, Any]] = []
        for item in history[-8:]:
            step_type = str(item.get("step_type") or "")
            if step_type != "tool":
                recent.append({"step_type": step_type})
                continue
            tool_input = dict(item.get("input") or {})
            output = dict(item.get("output") or {})
            validation = output.get("validation")
            observation = output.get("observation")
            recent.append(
                {
                    "step_type": "tool",
                    "tool_name": str(tool_input.get("tool_name") or ""),
                    "ok": bool(output.get("ok")),
                    "low_progress": bool(
                        isinstance(validation, dict)
                        and bool(validation.get("low_progress"))
                    ),
                    "observation_summary": (
                        str(observation.get("summary") or "")[:220]
                        if isinstance(observation, dict)
                        else ""
                    ),
                }
            )
        return {
            "directive": directive.as_dict(),
            "recent_history": recent,
            "successful_tools": Supervisor._count_successful_tools(history),
            "recent_failures": Supervisor._count_recent_failures(history),
            "recent_low_progress": Supervisor._count_recent_low_progress(history),
        }

    @staticmethod
    def _count_successful_tools(history: list[dict[str, Any]]) -> int:
        return sum(
            1
            for item in history
            if str(item.get("step_type") or "") == "tool"
            and bool(dict(item.get("output") or {}).get("ok"))
        )

    @staticmethod
    def _count_recent_failures(history: list[dict[str, Any]], window: int = 4) -> int:
        recent = history[-window:]
        return sum(
            1
            for item in recent
            if str(item.get("step_type") or "") == "tool"
            and not bool(dict(item.get("output") or {}).get("ok", False))
        )

    @staticmethod
    def _count_recent_low_progress(
        history: list[dict[str, Any]], window: int = 6
    ) -> int:
        recent = history[-window:]
        count = 0
        for item in recent:
            if str(item.get("step_type") or "") != "tool":
                continue
            output = dict(item.get("output") or {})
            validation = output.get("validation")
            if not isinstance(validation, dict):
                continue
            if bool(validation.get("low_progress")):
                count += 1
        return count

    def _ingest_advisory_outcome(
        self,
        *,
        gate_state: _AdvisoryGateState,
        history: list[dict[str, Any]],
        next_idx: int,
    ) -> None:
        """Evaluate outcome from the last applied advisory and adapt gate state."""
        with self._gate_lock:
            pending_len = int(gate_state.pending_outcome_history_len)
        if pending_len < 0:
            return
        if len(history) <= pending_len:
            return

        outcome_step = history[pending_len]
        signal = self._classify_outcome_step(outcome_step)
        with self._gate_lock:
            gate_state.pending_outcome_history_len = -1
            if signal == "positive":
                gate_state.consecutive_positive_outcomes += 1
                gate_state.total_outcome_successes += 1
                gate_state.consecutive_low_value = 0
                gate_state.consecutive_failures = 0
                return

            if signal == "negative":
                gate_state.consecutive_positive_outcomes = 0
                gate_state.total_outcome_failures += 1
                gate_state.consecutive_low_value += 1
                gate_state.cooldown_until_idx = max(
                    gate_state.cooldown_until_idx,
                    next_idx + self._advisory_cooldown_steps,
                )
                if gate_state.consecutive_low_value >= 2:
                    gate_state.disabled_until_idx = max(
                        gate_state.disabled_until_idx,
                        next_idx + self._advisory_disable_steps,
                    )
                return

            gate_state.consecutive_positive_outcomes = 0

    @staticmethod
    def _classify_outcome_step(step: dict[str, Any]) -> str:
        """Classify a step outcome as positive, negative, or neutral."""
        if str(step.get("step_type") or "") != "tool":
            return "neutral"
        output = dict(step.get("output") or {})
        if not bool(output.get("ok")):
            return "negative"
        validation = output.get("validation")
        if isinstance(validation, dict) and bool(validation.get("low_progress")):
            return "negative"
        return "positive"
