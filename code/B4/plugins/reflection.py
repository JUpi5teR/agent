from __future__ import annotations

from typing import Any

from B4.interfaces.plugin import B4Plugin


class ReflectionPlugin(B4Plugin):
    name = "reflection"

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        execution_results = payload.get("execution_results") if isinstance(payload, dict) else None
        if not isinstance(execution_results, list):
            raise ValueError("Reflection input must contain execution_results")

        failed = [
            item for item in execution_results
            if not isinstance(item, dict) or item.get("status") != "success"
        ]
        if failed:
            return {
                "success": False,
                "need_replan": True,
                "reason": f"{len(failed)} task(s) failed; return to Planner for replanning.",
                "metrics": self._metrics(execution_results),
            }
        return {
            "success": True,
            "need_replan": False,
            "reason": "All scheduled tasks completed successfully.",
            "metrics": self._metrics(execution_results),
        }

    def _metrics(self, execution_results: list[dict[str, Any]]) -> dict[str, float]:
        total = len(execution_results)
        if total == 0:
            return {"success_rate": 0.0, "completed": 0, "failed": 0}
        completed = sum(1 for item in execution_results if isinstance(item, dict) and item.get("status") == "success")
        failed = total - completed
        return {"success_rate": round(completed / total, 2), "completed": completed, "failed": failed}
