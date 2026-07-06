# B4 与 LangGraph 对比分析

## 概述

| 维度 | B4 (`agent/b4_local_agent_llm.py`) | LangGraph (`langgraph`) |
|------|-----------------------------------|------------------------|
| **定位** | 单模块：LLM 调用层 | 全功能框架：构建有状态、多actor Agent |
| **架构** | 直接函数调用 | 基于图的消息流（Pregel 风格） |
| **依赖** | 纯 `transformers` | 构建在 LangChain `Runnable` 协议之上 |
| **适用场景** | 简单 Agent 循环的 LLM 调用 | 复杂工作流、多步骤协作、长时间运行任务 |

---

## 1. 设计哲学

### B4：简洁专用

B4 是一个**单一职责模块**，专注于 LLM 调用：

```
输入: messages + tools_schema → B4 → AIMessage (content 或 tool_calls)
```

核心功能：
- 加载本地模型（Qwen3.5-4B 等）
- 构建带格式约束的 Prompt
- 解析模型输出（JSON 解析 + 三种回退策略）
- Mock 模式用于调试

**特点**：轻量、无依赖图结构、直接面向任务。

### LangGraph：图灵完备的工作流框架

LangGraph 将 Agent 建模为**有向图**，节点是计算单元，边是消息流：

```
@entrypoint + @task 装饰器
    ↓
Pregel 图引擎执行
    ↓
支持条件分支、并行任务、循环
```

核心概念：
- **`entrypoint`**：入口函数，定义工作流
- **`task`**：可并行执行的任务单元
- **`StateGraph`**：状态图，支持复杂状态管理
- **Checkpoint**：断点续执，持久化中间状态
- **Channel**：节点间传递数据的通道

---

## 2. 核心机制对比

### 2.1 LLM 调用

**B4**：
```python
generate_ai_message(model_config, messages, tools_schema, mode)
```
- 直接调用 `model.generate()`
- 内置 prompt 模板，注入 `tools_schema`
- 输出格式：`{"content": "...", "tool_calls": [...]}`

**LangGraph**：
LangGraph 本身不绑定 LLM，通过 `task` 装饰器自由定义：

```python
@task
def llm_call(messages, config):
    # 可自由选择 langchain-openai / anthropic / 本地模型
    return llm.bind_tools(tools).invoke(messages)
```

**区别**：B4 将 LLM 调用和 tool schema 绑定在一起；LangGraph 将 LLM 视为可替换的组件。

### 2.2 工具调用流程

**B4**：
```
B4 生成 tool_calls → B3 执行工具 → B4 生成最终回答
```
- B4 只生成 `AIMessage`，不执行工具
- 工具执行由独立模块 B3 负责
- 模块间通过文件/函数调用传递消息

**LangGraph**：
```
Node (LLM) → tool_calls → Node (Tool Executor) → 结果回传
```
- 工具执行是图中的一个节点
- 工具节点执行后，结果自动流回 LLM 节点
- 无需手动管理消息传递

### 2.3 状态管理

**B4**：
- 依赖 B5 模块管理记忆
- 消息历史由调用方（B1）维护
- 状态是外部的，通过参数传入

**LangGraph**：
```python
@entrypoint(checkpointer=InMemorySaver())
def workflow(state: State, config):
    # state 是图管理的状态，可跨调用持久化
    return {"result": ...}
```
- 内置状态管理，支持跨线程/跨进程持久化
- Checkpoint 保存中间状态，支持中断恢复
- 支持 `previous` 参数访问上一次调用结果

### 2.4 并发模型

**B4**：
- 顺序执行：LLM 调用 → 工具执行 → LLM 调用
- 无内置并发支持

**LangGraph**：
```python
@entrypoint()
def parallel_workflow(items):
    futures = [process_task.item(n) for n in items]
    results = [f.result() for f in futures]  # 并行 + 收集
    return results
```
- `task` 返回 `SyncAsyncFuture`，支持并行执行
- 图引擎自动管理依赖和执行顺序
- 支持条件分支并行

---

## 3. 架构复杂度

| 维度 | B4 | LangGraph |
|------|-----|-----------|
| 代码规模 | ~450 行（单文件） | 数十万行（多模块 monorepo） |
| 依赖数量 | 少（torch + transformers） | 多（LangChain + 各类存储适配器） |
| 学习曲线 | 平缓 | 陡峭 |
| 定制化难度 | 容易（单文件可直接改） | 困难（需理解 Pregel 图引擎） |

---

## 4. 适用场景

### 选择 B4 当：

- ✅ 需要快速原型化一个简单的 Agent
- ✅ 项目不需要复杂的状态管理和断点续传
- ✅ 已有其他模块（B1/B3/B5）负责编排，只缺 LLM 调用层
- ✅ 有 GPU，想直接用本地模型推理

### 选择 LangGraph 当：

- ✅ 需要构建复杂的多步骤工作流
- ✅ 需要长时间运行、人机交互、中断恢复
- ✅ 需要并行执行多个任务
- ✅ 需要生产级部署、监控、回滚
- ✅ 需要与 LangChain 生态（Retriever、Vector Store 等）集成

---

## 5. 关键代码对照

### B4 的 AIMessage 格式契约

```python
# content 和 tool_calls 互斥
{"content": "...", "tool_calls": []}      # 最终回答
{"content": "", "tool_calls": [{...}]}    # 工具调用
```

### LangGraph 的状态流

```python
# entrypoint 返回值自动写入 StateGraph
@entrypoint()
def my_workflow(input: str) -> str:
    result = llm.invoke(input)
    return result

# 可嵌套 task
@task
def process(x): return x + 1
```

---

## 6. 总结

| 方面 | B4 | LangGraph |
|------|-----|-----------|
| **设计目标** | 单模块 LLM 调用 | 全功能 Agent 框架 |
| **核心抽象** | 函数 | 有向图（Pregel） |
| **状态管理** | 外部（B5） | 内置 Checkpoint |
| **并发** | 无 | Future-based |
| **工具执行** | 外部（B3） | 图节点 |
| **适用规模** | 教学/原型 | 生产级 |

**本质上**：B4 是"**写一个函数调用模型**"，LangGraph 是"**用一个图引擎管理 Agent 生命周期**"。

两者不是替代关系，而是**不同抽象层次**的解决方案：
- B4 解决"L4 层：如何让模型生成 tool_calls"
- LangGraph 解决"L1-L7 层：如何构建、运行、监控一个完整 Agent 系统"
