from __future__ import annotations

import unittest

from B4.core.engine import B4CognitiveEngine
from B4.plugins.critic import CriticPlugin
from B4.plugins.goal_parser import GoalParserPlugin
from B4.plugins.planner import PlannerPlugin
from B4.plugins.reflection import ReflectionPlugin
from B4.plugins.scheduler import SchedulerPlugin


class B4PluginTests(unittest.TestCase):
    def test_goal_parser_outputs_json_shape(self) -> None:
        result = GoalParserPlugin().run("写一个Python学习路线，必须可验证，优先基础语法")
        self.assertEqual(set(result), {"goal", "constraints", "resources", "priority", "decision"})
        self.assertIsInstance(result["constraints"], list)
        self.assertEqual(result["decision"]["action"], "plan")

    def test_planner_outputs_executable_array(self) -> None:
        goal_json = GoalParserPlugin().run("写一个Python学习路线，必须可验证")
        plan = PlannerPlugin().run(goal_json)
        self.assertGreaterEqual(len(plan), 3)
        self.assertTrue(all({"id", "task"} == set(item) for item in plan))
        self.assertTrue(any("验证" in item["task"] or "检查" in item["task"] for item in plan))

    def test_critic_outputs_score_and_reason(self) -> None:
        goal_json = GoalParserPlugin().run("写一个Python学习路线，必须可验证")
        plan = PlannerPlugin().run(goal_json)
        result = CriticPlugin().run({"goal_json": goal_json, "plan": plan})
        self.assertGreaterEqual(result["score"], 8)
        self.assertIsInstance(result["reason"], str)
        self.assertIn("metrics", result)
        self.assertIn("reasoning_quality", result["metrics"])

    def test_scheduler_adds_order(self) -> None:
        schedule = SchedulerPlugin().run([{"id": 2, "task": "second"}, {"id": 1, "task": "first"}])
        self.assertEqual([item["order"] for item in schedule], [1, 2])
        self.assertEqual(schedule[0]["task"], "first")

    def test_reflection_requests_replan_on_failure(self) -> None:
        result = ReflectionPlugin().run({"execution_results": [{"status": "failed"}]})
        self.assertFalse(result["success"])
        self.assertTrue(result["need_replan"])

    def test_engine_runs_fixed_flow(self) -> None:
        result = B4CognitiveEngine().run("写一个Python学习路线，必须可验证")
        modules = [item["module"] for item in result["trace"]]
        self.assertEqual(modules[:5], ["Goal Parser", "Planner+Tree of Thoughts", "Critic", "Scheduler", "Reflection"])
        self.assertFalse(result["reflection"]["need_replan"])
        self.assertIn("response", result)

    def test_engine_classifies_direct_answer(self) -> None:
        result = B4CognitiveEngine().run("Python是什么")
        self.assertEqual(result["decision"]["action"], "direct_answer")
        self.assertEqual(result["response"]["answer_mode"], "direct")

    def test_engine_classifies_reasoning_answer(self) -> None:
        result = B4CognitiveEngine().run("为什么需要使用Tree of Thoughts进行严格推理")
        self.assertEqual(result["decision"]["action"], "reasoning_answer")
        self.assertTrue(result["response"]["reasoning_required"])
        self.assertTrue(any("候选思路" in item["task"] for item in result["plan"]))

    def test_engine_classifies_tool_execution(self) -> None:
        result = B4CognitiveEngine().run("读取代码文件并分析是否需要调用工具")
        self.assertEqual(result["decision"]["action"], "execute")
        self.assertTrue(result["response"]["needs_tool"])
        self.assertTrue(any("工具" in item["task"] for item in result["plan"]))


if __name__ == "__main__":
    unittest.main()
