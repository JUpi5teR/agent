from __future__ import annotations

from typing import Any

from B4.interfaces.plugin import B4Plugin


class CriticPlugin(B4Plugin):
    name = "critic"

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        plan = payload.get("plan") if isinstance(payload, dict) else None
        goal_json = payload.get("goal_json") if isinstance(payload, dict) else None
        if not isinstance(plan, list) or not isinstance(goal_json, dict):
            raise ValueError("Critic input must contain plan and goal_json")

        issues = self._find_issues(plan, goal_json)
        metrics = self._score_metrics(plan, goal_json, issues)
        score = round(sum(metrics.values()) / len(metrics), 1)
        reason = "Plan satisfies goal, constraints, order, and non-duplication checks." if not issues else "; ".join(issues)
        return {"score": score, "reason": reason, "metrics": metrics}

    def _find_issues(self, plan: list[dict[str, Any]], goal_json: dict[str, Any]) -> list[str]:
        issues: list[str] = []
        tasks = [item.get("task") for item in plan if isinstance(item, dict)]
        if not plan:
            issues.append("plan is empty")
        if len(tasks) != len(plan) or any(not isinstance(task, str) or not task.strip() for task in tasks):
            issues.append("plan contains invalid task")
        if len(tasks) != len(set(tasks)):
            issues.append("plan contains duplicate task")
        if not any("验证" in task or "检查" in task or "verify" in task.lower() for task in tasks if isinstance(task, str)):
            issues.append("plan misses verification step")
        if goal_json.get("constraints") and not any("约束" in task or "限制" in task for task in tasks if isinstance(task, str)):
            issues.append("plan does not explicitly handle constraints")
        if goal_json.get("goal") and len(plan) < 3:
            issues.append("plan is too short to satisfy goal")
        decision = goal_json.get("decision") if isinstance(goal_json.get("decision"), dict) else {}
        if decision.get("needs_tool") and not any("工具" in task or "调用" in task for task in tasks if isinstance(task, str)):
            issues.append("plan misses required tool-use step")
        if decision.get("reasoning_required") and not any("推理" in task or "候选思路" in task for task in tasks if isinstance(task, str)):
            issues.append("plan misses strict reasoning step")
        return issues

    def _score_metrics(
        self,
        plan: list[dict[str, Any]],
        goal_json: dict[str, Any],
        issues: list[str],
    ) -> dict[str, float]:
        tasks = [item.get("task", "") for item in plan if isinstance(item, dict)]
        decision = goal_json.get("decision") if isinstance(goal_json.get("decision"), dict) else {}
        has_validation = any("验证" in task or "检查" in task for task in tasks)
        has_constraints = not goal_json.get("constraints") or any("约束" in task or "限制" in task for task in tasks)
        has_tool = not decision.get("needs_tool") or any("工具" in task or "调用" in task for task in tasks)
        has_reasoning = not decision.get("reasoning_required") or any("推理" in task or "候选思路" in task for task in tasks)
        unique_ratio = len(set(tasks)) / len(tasks) if tasks else 0
        return {
            "completeness": 10.0 if len(plan) >= 3 and has_validation else 6.0,
            "executability": 10.0 if all(isinstance(task, str) and len(task) >= 4 for task in tasks) else 5.0,
            "constraint_fit": 10.0 if has_constraints else 6.0,
            "non_duplication": round(10.0 * unique_ratio, 1),
            "tool_fit": 10.0 if has_tool else 4.0,
            "reasoning_quality": 10.0 if has_reasoning else 4.0,
            "issue_penalty": max(0.0, 10.0 - 2.0 * len(issues)),
        }
