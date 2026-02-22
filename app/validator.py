"""Validation helpers for tactical tool outcomes.

Validation is intentionally lightweight and deterministic: it records whether a
tool action appears to have produced meaningful progress signals.
"""

from __future__ import annotations

from typing import Any


class ActionValidator:
    """Validate tool outcomes and emit progress-quality signals."""

    def validate_tool_result(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return validation metadata for one tool action."""
        if not bool(result.get("ok")):
            return {
                "ok": False,
                "low_progress": False,
                "code": "tool_error",
                "message": "tool returned an error payload",
                "details": {},
            }

        observation = result.get("observation")
        if not isinstance(observation, dict):
            return {
                "ok": True,
                "low_progress": False,
                "code": "no_observation",
                "message": "no structured observation available",
                "details": {},
            }

        if tool_name.endswith("browser_navigate"):
            return self._validate_navigate(
                tool_name=tool_name,
                args=args,
                observation=observation,
                history=history,
            )
        if tool_name.endswith("browser_snapshot"):
            return self._validate_snapshot(observation=observation)
        if tool_name.endswith("browser_run_code"):
            return self._validate_run_code(observation=observation)

        return {
            "ok": True,
            "low_progress": False,
            "code": "validated_generic",
            "message": "tool completed with generic validation",
            "details": {},
        }

    def _validate_navigate(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        observation: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        page_url = str(observation.get("page_url") or "").strip()
        requested_url = str(args.get("url") or "").strip()
        previous_url = self._latest_page_url(history)
        details = {
            "tool_name": tool_name,
            "requested_url": requested_url,
            "result_page_url": page_url,
            "previous_page_url": previous_url,
        }
        if not page_url:
            return {
                "ok": True,
                "low_progress": True,
                "code": "navigate_missing_page_url",
                "message": "navigation result lacks page_url observation",
                "details": details,
            }
        if previous_url and previous_url == page_url:
            return {
                "ok": True,
                "low_progress": True,
                "code": "navigate_same_url",
                "message": "navigation ended on same page URL",
                "details": details,
            }
        return {
            "ok": True,
            "low_progress": False,
            "code": "navigate_page_changed",
            "message": "navigation produced a page URL signal",
            "details": details,
        }

    def _validate_snapshot(self, *, observation: dict[str, Any]) -> dict[str, Any]:
        candidates = list(observation.get("action_candidates") or [])
        sections = list(observation.get("sections") or [])
        details = {
            "candidate_count": len(candidates),
            "sections": sections[:6],
        }
        if candidates:
            return {
                "ok": True,
                "low_progress": False,
                "code": "snapshot_has_candidates",
                "message": "snapshot produced actionable UI candidates",
                "details": details,
            }
        if sections:
            return {
                "ok": True,
                "low_progress": True,
                "code": "snapshot_sections_only",
                "message": "snapshot produced sections but no actionable candidates",
                "details": details,
            }
        return {
            "ok": True,
            "low_progress": True,
            "code": "snapshot_low_signal",
            "message": "snapshot produced low-signal observation",
            "details": details,
        }

    def _validate_run_code(self, *, observation: dict[str, Any]) -> dict[str, Any]:
        text = str(observation.get("text") or "").lower()
        summary = str(observation.get("summary") or "").lower()
        has_result_signal = "### result" in text or '"result"' in text or "returned" in text
        if not has_result_signal:
            # Wrapped payloads may still expose signal via summary.
            has_result_signal = "result" in summary
        return {
            "ok": True,
            "low_progress": not has_result_signal,
            "code": "run_code_result_signal" if has_result_signal else "run_code_low_signal",
            "message": (
                "run_code output contains explicit result signal"
                if has_result_signal
                else "run_code output lacks explicit result signal"
            ),
            "details": {},
        }

    @staticmethod
    def _latest_page_url(history: list[dict[str, Any]]) -> str:
        for item in reversed(history):
            if str(item.get("step_type") or "") != "tool":
                continue
            output = dict(item.get("output") or {})
            if not bool(output.get("ok")):
                continue
            observation = output.get("observation")
            if not isinstance(observation, dict):
                continue
            page_url = str(observation.get("page_url") or "").strip()
            if page_url:
                return page_url
        return ""

