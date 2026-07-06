from __future__ import annotations

from copy import deepcopy
from typing import Any

from B4.interfaces.plugin import B4Plugin


class _TreeOfThoughtsOptimizer:
    """Planner-internal optimizer: Expand -> Score -> Prune -> Expand."""

    def optimize(self, goal_json: dict[str, Any], initial_plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
        expanded = self._expand(goal_json, initial_plan)
        scored = self._score(goal_json, expanded)
        pruned = self._prune(scored)
        final_candidates = self._expand(goal_json, pruned[0]["plan"])
        return self._score(goal_json, final_candidates)[0]["plan"]

    def _expand(self, goal_json: dict[str, Any], plan: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        candidates = [deepcopy(plan)]
        constraints = goal_json.get("constraints") or []
        resources = goal_json.get("resources") or []

        with_validation = deepcopy(plan)
        if not any("验证" in task["task"] or "verify" in task["task"].lower() for task in with_validation):
            with_validation.append({"id": len(with_validation) + 1, "task": "验证计划是否满足目标、约束和资源条件"})
        candidates.append(with_validation)

        with_context = deepcopy(plan)
        if constraints or resources:
            with_context.insert(1, {"id": 2, "task": "整理可用资源和约束，形成执行清单"})
            with_context = self._renumber(with_context)
        candidates.append(with_context)

        decision = goal_json.get("decision") if isinstance(goal_json.get("decision"), dict) else {}
        if decision.get("reasoning_required"):
            with_reasoning = deepcopy(plan)
            with_reasoning.insert(1, {"id": 2, "task": "展开至少两个候选思路并记录推理依据"})
            with_reasoning.append({"id": len(with_reasoning) + 1, "task": "量化比较候选思路并选择最可靠结论"})
            candidates.append(self._renumber(with_reasoning))
        return candidates

    def _score(self, goal_json: dict[str, Any], candidates: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        scored = []
        for plan in candidates:
            tasks = [item.get("task", "") for item in plan]
            score = 0
            score += min(len(tasks), 6)
            score += 2 if any("验证" in task or "verify" in task.lower() for task in tasks) else 0
            score += 1 if any("资源" in task or "约束" in task for task in tasks) else 0
            score += 2 if goal_json.get("decision", {}).get("reasoning_required") and any("候选思路" in task for task in tasks) else 0
            score += 1 if goal_json.get("decision", {}).get("needs_tool") and any("工具" in task or "调用" in task for task in tasks) else 0
            score -= len(tasks) - len(set(tasks))
            score += 1 if goal_json.get("goal") else 0
            scored.append({"score": score, "plan": self._renumber(plan)})
        return sorted(scored, key=lambda item: item["score"], reverse=True)

    def _prune(self, scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return scored[:2]

    def _renumber(self, plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"id": index, "task": item["task"]} for index, item in enumerate(plan, 1)]


class PlannerPlugin(B4Plugin):
    name = "planner"

    def __init__(self) -> None:
        self._tot = _TreeOfThoughtsOptimizer()

    def run(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(payload, dict) or not payload.get("goal"):
            raise ValueError("Planner input must be Goal JSON with a goal field")

        initial_plan = self._build_initial_plan(payload)
        return self._tot.optimize(payload, initial_plan)

    def _build_initial_plan(self, goal_json: dict[str, Any]) -> list[dict[str, Any]]:
        goal = str(goal_json["goal"])
        decision = goal_json.get("decision") if isinstance(goal_json.get("decision"), dict) else {}
        action = decision.get("action", "plan")
        if action == "direct_answer":
            tasks = [
                "提取问题中的核心概念",
                "直接给出简洁回答",
                "检查回答是否覆盖用户问题",
            ]
        elif action == "reasoning_answer":
            tasks = [
                "拆分问题中的前提、结论和隐含条件",
                "生成多个候选推理路径",
                "量化评估每条推理路径的可靠性",
                "选择最佳推理路径并给出结论",
                "验证结论是否存在明显反例",
            ]
        elif action == "execute" and decision.get("needs_tool"):
            tools = ", ".join(decision.get("tool_candidates") or ["tool_selector"])
            tasks = [
                "确认执行目标、输入和成功标准",
                f"选择并准备需要调用的工具：{tools}",
                "按顺序调用工具并记录结果",
                "检查工具结果是否满足目标和约束",
                "汇总执行结果并给出下一步建议",
            ]
        elif "学习路线" in goal:
            tasks = [
                "明确学习主题、当前基础和最终验收标准",
                "列出必须掌握的核心概念并按先后关系排序",
                "为每个核心概念匹配学习资料和练习任务",
                "安排阶段性项目或测验来检查掌握程度",
                "根据检查结果调整后续学习节奏",
            ]
        else:
            tasks = [
                "明确目标的完成标准和交付物",
                "拆分目标所需的关键步骤",
                "按依赖关系执行每个步骤",
                "检查执行结果是否满足目标",
            ]
        return [{"id": index, "task": task} for index, task in enumerate(tasks, 1)]
