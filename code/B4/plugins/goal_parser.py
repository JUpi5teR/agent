from __future__ import annotations

import re
from typing import Any

from B4.interfaces.plugin import B4Plugin


class GoalParserPlugin(B4Plugin):
    name = "goal_parser"

    def run(self, payload: str) -> dict[str, list[str] | str]:
        if not isinstance(payload, str) or not payload.strip():
            raise ValueError("Goal Parser input must be a non-empty natural language goal")

        text = payload.strip()
        return {
            "goal": text,
            "constraints": self._extract_constraints(text),
            "resources": self._extract_resources(text),
            "priority": self._extract_priority(text),
            "decision": self._classify_decision(text),
        }

    def _extract_constraints(self, text: str) -> list[str]:
        markers = ("必须", "不能", "禁止", "要求", "限制", "without", "must", "only")
        constraints = self._sentences_with_markers(text, markers)
        return constraints or ["保持步骤可执行、可验证"]

    def _extract_resources(self, text: str) -> list[str]:
        resources: list[str] = []
        quoted = re.findall(r"[“\"]([^”\"]+)[”\"]", text)
        resources.extend(item.strip() for item in quoted if item.strip())

        resource_markers = ("资料", "教材", "课程", "工具", "代码", "文档", "resource")
        resources.extend(self._sentences_with_markers(text, resource_markers))
        return self._dedupe(resources) or ["用户目标描述"]

    def _extract_priority(self, text: str) -> list[str]:
        priority_markers = ("优先", "重点", "首先", "最重要", "priority", "first")
        priorities = self._sentences_with_markers(text, priority_markers)
        if "学习路线" in text:
            priorities.append("先建立学习顺序，再安排练习验证")
        return self._dedupe(priorities) or ["先完成核心目标，再补充细节"]

    def _sentences_with_markers(self, text: str, markers: tuple[str, ...]) -> list[str]:
        parts = [part.strip(" \t\r\n，。；;") for part in re.split(r"[。；;\n]+", text)]
        return self._dedupe([part for part in parts if part and any(marker in part for marker in markers)])

    def _dedupe(self, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            if value not in result:
                result.append(value)
        return result

    def _classify_decision(self, text: str) -> dict[str, Any]:
        lower_text = text.lower()
        tool_markers = ("读取", "查找", "搜索", "文件", "代码", "运行", "计算", "工具", "调用", "read", "search", "run")
        plan_markers = ("计划", "路线", "步骤", "安排", "规划", "拆解", "plan", "schedule")
        execute_markers = ("执行", "运行", "修改", "生成", "创建", "写入", "execute", "run", "create")
        reasoning_markers = ("为什么", "推理", "分析", "比较", "证明", "判断", "原因", "reason", "why", "compare")
        simple_answer_markers = ("是什么", "解释", "说明", "define", "what is")

        needs_tool = any(marker in lower_text or marker in text for marker in tool_markers)
        needs_plan = any(marker in lower_text or marker in text for marker in plan_markers)
        needs_execute = any(marker in lower_text or marker in text for marker in execute_markers)
        needs_reasoning = any(marker in lower_text or marker in text for marker in reasoning_markers)
        direct_answer = any(marker in lower_text or marker in text for marker in simple_answer_markers)

        if needs_execute or needs_tool:
            action = "execute"
        elif needs_plan:
            action = "plan"
        elif needs_reasoning:
            action = "reasoning_answer"
        elif direct_answer or len(text) <= 24:
            action = "direct_answer"
        else:
            action = "plan"

        evidence = []
        if needs_tool:
            evidence.append("检测到可能需要工具或文件/代码访问")
        if needs_plan:
            evidence.append("检测到计划/路线/步骤类目标")
        if needs_execute:
            evidence.append("检测到执行类动词")
        if needs_reasoning:
            evidence.append("检测到推理或分析类目标")
        if direct_answer:
            evidence.append("检测到可直接回答的问题")

        confidence = 0.55 + 0.1 * min(len(evidence), 4)
        return {
            "action": action,
            "needs_tool": needs_tool,
            "reasoning_required": needs_reasoning or action == "reasoning_answer",
            "answer_mode": "reasoning" if needs_reasoning else "direct" if action == "direct_answer" else "planned",
            "tool_candidates": self._tool_candidates(text) if needs_tool else [],
            "confidence": round(min(confidence, 0.95), 2),
            "evidence": evidence or ["未检测到强触发词，默认进入计划流程"],
        }

    def _tool_candidates(self, text: str) -> list[str]:
        candidates = []
        if any(marker in text for marker in ("文件", "读取", "文档", "代码")):
            candidates.append("file_reader")
        if any(marker in text for marker in ("搜索", "查找")):
            candidates.append("local_file_search")
        if any(marker in text for marker in ("计算", "量化", "分数")):
            candidates.append("calculator")
        if any(marker in text for marker in ("运行", "执行", "命令")):
            candidates.append("command_runner")
        return candidates or ["tool_selector"]
