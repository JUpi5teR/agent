from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from common.io_utils import append_jsonl, read_json, read_yaml, write_json
from common.logging_utils import now_iso
from common.path_utils import resolve_cli_path, resolve_from_file
from common.schemas import make_ai_message, validate_ai_message, validate_messages


PARSE_ERROR_CONTENT = "模型输出解析失败，无法生成有效工具调用或最终回答。"
_MODEL_CACHE: dict[tuple[str, ...], tuple[Any, Any]] = {}


class B4Plugin:
    """Unified interface for B4 cognitive engine plugins."""

    name = "base"

    def run(self, payload: Any) -> Any:
        raise NotImplementedError


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
        }

    def _extract_constraints(self, text: str) -> list[str]:
        constraints = self._sentences_with_markers(text, ("必须", "不能", "禁止", "要求", "限制", "must", "only"))
        return constraints or ["保持步骤可执行、可验证"]

    def _extract_resources(self, text: str) -> list[str]:
        resources = [item.strip() for item in re.findall(r"[“\"]([^”\"]+)[”\"]", text) if item.strip()]
        resources.extend(self._sentences_with_markers(text, ("资料", "教材", "课程", "工具", "代码", "文档", "resource")))
        return self._dedupe(resources) or ["用户目标描述"]

    def _extract_priority(self, text: str) -> list[str]:
        priorities = self._sentences_with_markers(text, ("优先", "重点", "首先", "最重要", "priority", "first"))
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
        with_validation = deepcopy(plan)
        if not any("验证" in item["task"] or "检查" in item["task"] for item in with_validation):
            with_validation.append({"id": len(with_validation) + 1, "task": "验证计划是否满足目标、约束和资源条件"})
        candidates.append(with_validation)

        with_context = deepcopy(plan)
        if goal_json.get("constraints") or goal_json.get("resources"):
            with_context.insert(1, {"id": 2, "task": "整理可用资源和约束，形成执行清单"})
            with_context = self._renumber(with_context)
        candidates.append(with_context)
        return candidates

    def _score(self, goal_json: dict[str, Any], candidates: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        scored = []
        for plan in candidates:
            tasks = [item.get("task", "") for item in plan]
            score = min(len(tasks), 6)
            score += 2 if any("验证" in task or "检查" in task for task in tasks) else 0
            score += 1 if any("资源" in task or "约束" in task for task in tasks) else 0
            score += 1 if goal_json.get("goal") else 0
            score -= len(tasks) - len(set(tasks))
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
        return self._tot.optimize(payload, self._build_initial_plan(payload))

    def _build_initial_plan(self, goal_json: dict[str, Any]) -> list[dict[str, Any]]:
        goal = str(goal_json["goal"])
        if "学习路线" in goal:
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


class CriticPlugin(B4Plugin):
    name = "critic"

    def run(self, payload: dict[str, Any]) -> dict[str, int | str]:
        if not isinstance(payload, dict):
            raise ValueError("Critic input must contain plan and goal_json")
        plan = payload.get("plan")
        goal_json = payload.get("goal_json")
        if not isinstance(plan, list) or not isinstance(goal_json, dict):
            raise ValueError("Critic input must contain plan and goal_json")
        issues = self._find_issues(plan, goal_json)
        return {
            "score": max(0, 10 - len(issues) * 2),
            "reason": "Plan satisfies goal, constraints, order, and non-duplication checks." if not issues else "; ".join(issues),
        }

    def _find_issues(self, plan: list[dict[str, Any]], goal_json: dict[str, Any]) -> list[str]:
        tasks = [item.get("task") for item in plan if isinstance(item, dict)]
        issues: list[str] = []
        if not plan:
            issues.append("plan is empty")
        if len(tasks) != len(plan) or any(not isinstance(task, str) or not task.strip() for task in tasks):
            issues.append("plan contains invalid task")
        if len(tasks) != len(set(tasks)):
            issues.append("plan contains duplicate task")
        if not any("验证" in task or "检查" in task for task in tasks if isinstance(task, str)):
            issues.append("plan misses verification step")
        if goal_json.get("constraints") and not any("约束" in task or "限制" in task for task in tasks if isinstance(task, str)):
            issues.append("plan does not explicitly handle constraints")
        return issues


class SchedulerPlugin(B4Plugin):
    name = "scheduler"

    def run(self, payload: list[dict[str, Any]]) -> list[dict[str, int | str]]:
        if not isinstance(payload, list):
            raise ValueError("Scheduler input must be a plan array")
        schedule = []
        for order, item in enumerate(sorted(payload, key=lambda task: int(task.get("id", 0))), 1):
            task = item.get("task")
            if not isinstance(task, str) or not task.strip():
                raise ValueError("Scheduler received invalid task")
            schedule.append({"id": int(item.get("id", order)), "task": task, "order": order})
        return schedule


class ReflectionPlugin(B4Plugin):
    name = "reflection"

    def run(self, payload: dict[str, Any]) -> dict[str, bool | str]:
        execution_results = payload.get("execution_results") if isinstance(payload, dict) else None
        if not isinstance(execution_results, list):
            raise ValueError("Reflection input must contain execution_results")
        failed = [item for item in execution_results if not isinstance(item, dict) or item.get("status") != "success"]
        if failed:
            return {"success": False, "need_replan": True, "reason": f"{len(failed)} task(s) failed; return to Planner."}
        return {"success": True, "need_replan": False, "reason": "All scheduled tasks completed successfully."}


class B4CognitiveEngine:
    """Core scheduler for Goal Parser -> Planner/ToT -> Critic -> Scheduler -> Reflection."""

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
            reflection_result = self.plugins["reflection"].run({"execution_results": self._execute_schedule(schedule)})
            trace.append({"module": "Reflection", "round": round_index, "output": reflection_result})
            if not reflection_result["need_replan"]:
                break
            goal_json = self._add_replan_constraint(goal_json, str(reflection_result["reason"]))

        return {
            "goal_json": goal_json,
            "plan": plan,
            "critic": critic_result,
            "schedule": schedule,
            "reflection": reflection_result,
            "trace": trace,
        }

    def _execute_schedule(self, schedule: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"id": item["id"], "order": item["order"], "task": item["task"], "status": "success"} for item in schedule]

    def _add_replan_constraint(self, goal_json: dict[str, Any], reason: str) -> dict[str, Any]:
        updated = dict(goal_json)
        updated["constraints"] = list(updated.get("constraints") or []) + [f"Replan because: {reason}"]
        return updated


def _load_model_config(model_config: str | Path) -> tuple[Path, dict]:
    path = Path(model_config).resolve()
    config = read_yaml(path)
    if not isinstance(config, dict):
        raise ValueError("model.yaml must contain an object")
    return path, config


def _artifact_paths(artifact_dir: str | Path, stem: str | None) -> tuple[Path, Path, Path]:
    directory = Path(artifact_dir)
    prefix = f"{stem}_" if stem else ""
    return (
        directory / f"{prefix}raw_model_output.json",
        directory / f"{prefix}ai_message.json",
        directory / "llm_run_log.jsonl",
    )


def _extract_tool_result(message: dict) -> dict:
    try:
        result = json.loads(message["content"])
    except (KeyError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError("ToolMessage content is not a SkillResult JSON string") from exc
    if not isinstance(result, dict):
        raise ValueError("ToolMessage content must decode to an object")
    return result


def _three_points(text: str) -> list[str]:
    parts = [part.strip(" \t\r\n。") for part in re.split(r"\n+|(?<=[。！？!?])", text) if part.strip()]
    points = []
    for part in parts:
        if part not in points:
            points.append(part)
        if len(points) == 3:
            break
    while len(points) < 3:
        points.append("工具结果未提供更多可提取内容")
    return points


def _mock_generate(messages: list[dict]) -> dict:
    tool_messages = [message for message in messages if message.get("role") == "tool"]
    if not tool_messages:
        return make_ai_message(
            "",
            [
                {
                    "id": "call_001",
                    "name": "file_reader",
                    "args": {"path": "docs/agent_intro.txt", "max_chars": 2000},
                }
            ],
        )
    latest = tool_messages[-1]
    result = _extract_tool_result(latest)
    if latest.get("status") != "success" or result.get("status") != "success":
        error = result.get("error") or {}
        detail = error.get("message", "未知工具错误") if isinstance(error, dict) else str(error)
        return make_ai_message(f"工具调用失败，无法完成请求：{detail}", [])
    output = result.get("output") or {}
    content = output.get("content") if isinstance(output, dict) else None
    if not isinstance(content, str) or not content.strip():
        content = json.dumps(output, ensure_ascii=False)
    points = _three_points(content)
    answer = "三条中文要点如下：\n" + "\n".join(f"{index}. {point}" for index, point in enumerate(points, 1))
    return make_ai_message(answer, [])


def _parse_tool_calls_fragment(raw_text: str, original_error: json.JSONDecodeError) -> dict:
    markers = ['"tool_calls":[', '\\"tool_calls\\":[']
    marker_index = -1
    marker = ""
    for item in markers:
        marker_index = raw_text.find(item)
        if marker_index != -1:
            marker = item
            break
    if marker_index == -1:
        raise original_error
    array_start = marker_index + marker.index("[")
    array_end = raw_text.rfind("]")
    if array_end < array_start:
        raise ValueError("model output contains tool_calls marker but no closing array")
    array_text = raw_text[array_start : array_end + 1]
    try:
        tool_calls = json.loads(array_text)
    except json.JSONDecodeError:
        tool_calls = json.loads(array_text.replace('\\"', '"'))
    if not isinstance(tool_calls, list) or not tool_calls:
        raise original_error
    return {"content": "", "tool_calls": tool_calls}


def _parse_json_with_backtick_tail(raw_text: str, original_error: json.JSONDecodeError) -> dict:
    text = raw_text.strip()
    try:
        candidate, end_index = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        raise original_error
    trailing = text[end_index:].strip()
    if trailing and set(trailing) <= {"`"}:
        return candidate
    raise original_error


def _candidate_to_message(candidate: dict) -> tuple[dict, dict]:
    if not isinstance(candidate, dict):
        raise ValueError("model output JSON must be an object")
    expected_keys = {"content", "tool_calls"}
    unknown_keys = set(candidate) - expected_keys
    if unknown_keys:
        raise ValueError(f"model output JSON contains unknown keys: {', '.join(sorted(unknown_keys))}")
    message = {
        "role": "assistant",
        "content": candidate.get("content", ""),
        "tool_calls": candidate.get("tool_calls", []),
    }
    validate_ai_message(message)
    has_content = bool(message["content"].strip())
    has_tool_calls = bool(message["tool_calls"])
    if has_content == has_tool_calls:
        raise ValueError("model output must contain either final content or tool calls, but not both")
    parsed_candidate = {"content": message["content"], "tool_calls": message["tool_calls"]}
    return parsed_candidate, message


def _parse_model_output(raw_text: str) -> tuple[dict, dict]:
    try:
        candidate = json.loads(raw_text.strip())
    except json.JSONDecodeError as exc:
        try:
            candidate = _parse_json_with_backtick_tail(raw_text, exc)
        except json.JSONDecodeError:
            candidate = _parse_tool_calls_fragment(raw_text, exc)
    return _candidate_to_message(candidate)


def _dtype_value(torch_module: Any, configured: str) -> Any:
    if configured == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }
    if configured not in mapping:
        raise ValueError(f"unsupported torch_dtype: {configured}")
    return mapping[configured]


def _model_cache_key(
    model_path: Path,
    tokenizer_path: Path,
    local_only: bool,
    trust_remote_code: bool,
    dtype: Any,
    device_map: Any,
    max_memory: Any,
) -> tuple[str, ...]:
    try:
        device_map_key = json.dumps(device_map, sort_keys=True, separators=(",", ":"))
    except TypeError:
        device_map_key = repr(device_map)
    try:
        max_memory_key = json.dumps(max_memory, sort_keys=True, separators=(",", ":"))
    except TypeError:
        max_memory_key = repr(max_memory)
    return (
        str(model_path),
        str(tokenizer_path),
        str(local_only),
        str(trust_remote_code),
        str(dtype),
        device_map_key,
        max_memory_key,
    )


def _load_model_bundle(
    auto_model: Any,
    auto_tokenizer: Any,
    model_path: Path,
    tokenizer_path: Path,
    local_only: bool,
    trust_remote_code: bool,
    dtype: Any,
    device_map: Any,
    max_memory: Any,
) -> tuple[Any, Any]:
    cache_key = _model_cache_key(
        model_path,
        tokenizer_path,
        local_only,
        trust_remote_code,
        dtype,
        device_map,
        max_memory,
    )
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        print("model_cache=hit", file=sys.stderr, flush=True)
        return cached

    print("model_cache=miss", file=sys.stderr, flush=True)
    tokenizer = auto_tokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=local_only,
        trust_remote_code=trust_remote_code,
    )
    model = auto_model.from_pretrained(
        str(model_path),
        local_files_only=local_only,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        device_map=device_map,
        max_memory=max_memory,
    )
    _MODEL_CACHE[cache_key] = (tokenizer, model)
    return tokenizer, model


def _build_prompt_messages(messages: list[dict], tools_schema: list[dict]) -> list[dict]:
    prompt_messages = deepcopy(messages)
    format_instruction = (
        "IMPORTANT OUTPUT FORMAT:\n"
        "You must return exactly one valid JSON object.\n"
        "Do not output markdown.\n"
        "Do not output explanations.\n"
        "Do not output code fences or backticks.\n"
        'The first output character must be "{" and the last output character must be "}".\n\n'
        "Valid schema A:\n"
        '{"content":"final answer text","tool_calls":[]}\n\n'
        "Valid schema B:\n"
        '{"content":"","tool_calls":[{"id":"call_001","name":"file_reader",'
        '"args":{"path":"docs/agent_intro.txt","max_chars":2000}}]}\n\n'
        "The top-level keys must be exactly:\n"
        "- content: string\n"
        "- tool_calls: array\n\n"
        "Never put tool_calls inside content.\n"
        'Never output {"content":"tool_calls": ...}.'
    )
    envelope_reminder = (
        "IMPORTANT OUTPUT FORMAT: Output the JSON object now. "
        'Your first output character must be "{" and your last output character must be "}". '
        "Never output a backtick, Markdown, a code block, an explanation, or text outside the JSON. "
        'Use exactly the top-level keys "content" (string) and "tool_calls" (array). '
        "Choose exactly one schema: final content with an empty tool_calls array, or empty content with tool calls. "
        'Never put tool_calls inside content. Never output {"content":"tool_calls": ...}.'
    )
    system_instruction = (
        "\n\nAvailable tools JSON schema:\n"
        + json.dumps(tools_schema, ensure_ascii=False)
        + "\n"
        + format_instruction
    )
    if prompt_messages and prompt_messages[0].get("role") == "system":
        prompt_messages[0]["content"] += system_instruction
    else:
        prompt_messages.insert(0, {"role": "system", "content": system_instruction.strip()})

    for message in reversed(prompt_messages):
        if message.get("role") == "user":
            message["content"] += "\n\n" + envelope_reminder
            break
    if prompt_messages[-1].get("role") == "tool":
        prompt_messages.append(
            {
                "role": "user",
                "content": (
                    envelope_reminder
                    + " The latest ToolMessage already contains a tool result. If it provides the requested "
                    'information, answer with schema A now and set "tool_calls" to exactly []. Do not repeat the '
                    "completed tool call."
                ),
            }
        )
    return prompt_messages


def _prompt_json_generate(config_path: Path, config: dict, messages: list[dict], tools_schema: list[dict]) -> str:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("prompt_json mode requires requirements-llm.txt") from exc
    model_config = config.get("model", {})
    generation_config = config.get("generation", {})
    model_setting = model_config.get("model_name_or_path")
    tokenizer_setting = model_config.get("tokenizer_name_or_path", model_setting)
    if not isinstance(model_setting, str) or not isinstance(tokenizer_setting, str):
        raise ValueError("model_name_or_path and tokenizer_name_or_path are required")
    model_path = resolve_from_file(model_setting, config_path)
    tokenizer_path = resolve_from_file(tokenizer_setting, config_path)
    if not model_path.exists() or not tokenizer_path.exists():
        raise FileNotFoundError(f"local model path does not exist: {model_path}")
    local_only = bool(model_config.get("local_files_only", True))
    trust_remote_code = bool(model_config.get("trust_remote_code", False))
    dtype = _dtype_value(torch, str(model_config.get("torch_dtype", "auto")))
    tokenizer, model = _load_model_bundle(
        AutoModelForCausalLM,
        AutoTokenizer,
        model_path,
        tokenizer_path,
        local_only,
        trust_remote_code,
        dtype,
        model_config.get("device_map", "auto"),
        model_config.get("max_memory"),
    )
    prompt_messages = _build_prompt_messages(messages, tools_schema)
    inputs = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    )
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    input_length = inputs["input_ids"].shape[-1]
    options = {
        "max_new_tokens": int(generation_config.get("max_new_tokens", 1024)),
        "do_sample": bool(generation_config.get("do_sample", False)),
    }
    with torch.no_grad():
        generated = model.generate(**inputs, **options)
    new_tokens = generated[0][input_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_ai_message(
    model_config: str,
    messages: list[dict],
    tools_schema: list[dict],
    mode: str = "prompt_json",
    artifact_dir: str | None = None,
    artifact_stem: str | None = None,
) -> dict:
    config_path, config = _load_model_config(model_config)
    messages = validate_messages(deepcopy(messages))
    if not isinstance(tools_schema, list):
        raise ValueError("tools_schema must be an array")
    generated_at = now_iso()
    backend = "mock" if mode == "mock" else config.get("model", {}).get("backend", "transformers")
    if mode == "mock":
        ai_message = _mock_generate(messages)
        raw_text = json.dumps({"content": ai_message["content"], "tool_calls": ai_message["tool_calls"]}, ensure_ascii=False)
        parsed_candidate = {"content": ai_message["content"], "tool_calls": ai_message["tool_calls"]}
        status = "success"
        error = None
    elif mode == "prompt_json":
        raw_text = _prompt_json_generate(config_path, config, messages, tools_schema)
        try:
            parsed_candidate, ai_message = _parse_model_output(raw_text)
            status = "success"
            error = None
        except Exception as exc:
            parsed_candidate = None
            ai_message = make_ai_message(PARSE_ERROR_CONTENT, [])
            status = "error"
            error = {"type": type(exc).__name__, "message": str(exc)}
    else:
        raise ValueError("mode must be mock or prompt_json")
    raw_record = {
        "mode": mode,
        "backend": backend,
        "raw_text": raw_text,
        "parsed_candidate": parsed_candidate,
        "status": status,
        "error": error,
        "generated_at": generated_at,
    }
    if artifact_dir:
        raw_path, message_path, log_path = _artifact_paths(artifact_dir, artifact_stem)
        write_json(raw_record, raw_path)
        write_json(ai_message, message_path)
        append_jsonl(
            {
                "timestamp": generated_at,
                "mode": mode,
                "status": status,
                "raw_output_path": str(raw_path),
                "ai_message_path": str(message_path),
                "error": error,
            },
            log_path,
        )
    return {
        "ai_message": ai_message,
        "status": status,
        "error": error,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run B4 cognitive planning or generate one AIMessage.")
    parser.add_argument("--goal", help="Natural language goal for the B4 Cognitive Engine.")
    parser.add_argument("--max_rounds", type=int, default=2, help="Maximum reflection rounds for --goal mode.")
    parser.add_argument("--model_config")
    parser.add_argument("--messages")
    parser.add_argument("--tools_schema")
    parser.add_argument("--mode", choices=["mock", "prompt_json"])
    parser.add_argument("--outdir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.goal:
            from B4.core.engine import B4CognitiveEngine as PackageB4CognitiveEngine

            result = PackageB4CognitiveEngine(max_reflection_rounds=args.max_rounds).run(args.goal)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        missing = [
            name
            for name in ("model_config", "messages", "tools_schema", "mode", "outdir")
            if getattr(args, name) is None
        ]
        if missing:
            raise ValueError("--goal or all legacy LLM arguments are required: " + ", ".join(missing))

        outdir = resolve_cli_path(args.outdir)
        generate_ai_message(
            str(resolve_cli_path(args.model_config)),
            read_json(resolve_cli_path(args.messages)),
            read_json(resolve_cli_path(args.tools_schema)),
            args.mode,
            str(outdir),
        )
        print(outdir / "ai_message.json")
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
