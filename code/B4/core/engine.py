from __future__ import annotations

from typing import Any

from B4.plugins.critic import CriticPlugin
from B4.plugins.goal_parser import GoalParserPlugin
from B4.plugins.planner import PlannerPlugin
from B4.plugins.reflection import ReflectionPlugin
from B4.plugins.scheduler import SchedulerPlugin


class B4CognitiveEngine:
    """Core orchestrator for the fixed B4 execution flow."""

    def __init__(self, max_reflection_rounds: int = 2, critic_threshold: int = 8) -> None:
        self.max_reflection_rounds = max_reflection_rounds
        self.critic_threshold = critic_threshold
        self.plugins = {
            "goal_parser": GoalParserPlugin(),
            "planner": PlannerPlugin(),
            "critic": CriticPlugin(),
            "scheduler": SchedulerPlugin(),
            "reflection": ReflectionPlugin(),
        }

    def run(self, user_goal: str) -> dict[str, Any]:
        goal_json = self.plugins["goal_parser"].run(user_goal)
        trace: list[dict[str, Any]] = [{"module": "Goal Parser", "output": goal_json}]

        plan: list[dict[str, Any]] = []
        critic_result: dict[str, Any] = {}
        schedule: list[dict[str, Any]] = []
        reflection_result: dict[str, Any] = {}

        for round_index in range(1, self.max_reflection_rounds + 1):
            plan = self.plugins["planner"].run(goal_json)
            trace.append({"module": "Planner+Tree of Thoughts", "round": round_index, "output": plan})

            critic_result = self.plugins["critic"].run({"goal_json": goal_json, "plan": plan})
            trace.append({"module": "Critic", "round": round_index, "output": critic_result})
            if int(critic_result["score"]) < self.critic_threshold:
                goal_json = self._add_replan_constraint(goal_json, str(critic_result["reason"]))
                continue

            schedule = self.plugins["scheduler"].run(plan)
            trace.append({"module": "Scheduler", "round": round_index, "output": schedule})

            execution_results = self._execute_schedule(schedule)
            reflection_result = self.plugins["reflection"].run({"execution_results": execution_results})
            trace.append({"module": "Reflection", "round": round_index, "output": reflection_result})
            if not reflection_result["need_replan"]:
                break
            goal_json = self._add_replan_constraint(goal_json, str(reflection_result["reason"]))

        return {
            "goal_json": goal_json,
            "decision": goal_json.get("decision", {}),
            "plan": plan,
            "critic": critic_result,
            "schedule": schedule,
            "reflection": reflection_result,
            "response": self._build_response(goal_json, plan, schedule, critic_result, reflection_result),
            "trace": trace,
        }

    def _execute_schedule(self, schedule: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"id": item["id"], "order": item["order"], "task": item["task"], "status": "success"}
            for item in schedule
        ]

    def _add_replan_constraint(self, goal_json: dict[str, Any], reason: str) -> dict[str, Any]:
        updated = dict(goal_json)
        constraints = list(updated.get("constraints") or [])
        constraints.append(f"Replan because: {reason}")
        updated["constraints"] = constraints
        return updated

    def _build_response(
        self,
        goal_json: dict[str, Any],
        plan: list[dict[str, Any]],
        schedule: list[dict[str, Any]],
        critic_result: dict[str, Any],
        reflection_result: dict[str, Any],
    ) -> dict[str, Any]:
        decision = goal_json.get("decision") if isinstance(goal_json.get("decision"), dict) else {}
        action = decision.get("action", "plan")
        if action == "direct_answer":
            content = f"直接回答：{goal_json.get('goal')}"
        elif action == "reasoning_answer":
            content = "推理回答：已通过 Tree of Thoughts 生成候选思路、量化评估并选择最佳结论。"
        elif action == "execute":
            tools = ", ".join(decision.get("tool_candidates") or [])
            content = f"执行方案：按调度顺序执行任务；需要工具：{tools or '否'}。"
        else:
            content = "制定计划：已生成可执行、可验证并经过量化评估的任务计划。"

        return {
            "action": action,
            "answer_mode": decision.get("answer_mode", "planned"),
            "needs_tool": bool(decision.get("needs_tool")),
            "reasoning_required": bool(decision.get("reasoning_required")),
            "content": content,
            "quality_score": critic_result.get("score"),
            "execution_success": reflection_result.get("success"),
            "scheduled_steps": len(schedule),
            "plan_summary": [item["task"] for item in plan],
        }
