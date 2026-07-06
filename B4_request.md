请阅读文件《B4任务.md》，并严格按照其中定义重构B4模块代码。

本任务的目标是实现一个“结构化任务规划与执行系统（B4 Cognitive Engine）”。

---

# ■ 一、系统架构约束（必须遵守）

B4必须按照以下固定执行流实现：

Goal Parser
→ Planner
→ Tree of Thoughts (Planner内部优化器)
→ Critic
→ Scheduler
→ Reflection Loop

禁止改变顺序或删除模块。

---

# ■ 二、模块输入输出规范（必须严格遵守）

## 1. Goal Parser

输入：用户自然语言目标  
输出（必须为JSON）：

{
  "goal": "...",
  "constraints": [...],
  "resources": [...],
  "priority": [...]
}

---

## 2. Planner

输入：Goal JSON  
输出（必须为数组）：

[
  {
    "id": 1,
    "task": "..."
  },
  {
    "id": 2,
    "task": "..."
  }
]

要求：
- 每个task必须可执行
- 必须有顺序逻辑
- 不允许抽象描述

---

## 3. Tree of Thoughts（ToT）

说明：
ToT是Planner内部优化模块，不是独立流程。

功能：
对Planner生成的初始计划进行搜索优化：

流程：
Expand → Score → Prune → Expand

输出：
优化后的Plan（同Planner格式）

---

## 4. Critic

输入：Plan  
输出（必须JSON）：

{
  "score": 0-10,
  "reason": "..."
}

检查内容：
- 是否遗漏步骤
- 是否重复
- 是否违反constraints
- 是否满足goal

---

## 5. Scheduler

输入：Critic通过的Plan  
输出：

[
  {
    "id": 1,
    "task": "...",
    "order": 1
  }
]

功能：
- 排序任务
- 处理依赖关系
- 生成执行顺序

---

## 6. Reflection

输入：执行结果  
输出：

{
  "success": true/false,
  "need_replan": true/false,
  "reason": "..."
}

规则：
- 如果失败 → 回到Planner
- 如果成功 → 结束

---

# ■ 三、工程结构要求（必须实现插件化）

请将B4实现为插件架构：

B4/
 ├── core/        (调度器)
 ├── plugins/     (各模块插件)
 ├── interfaces/  (统一接口)
 ├── main.py      (入口)

要求：
- 每个模块必须独立插件
- plugin之间禁止直接调用
- 通过core调度通信

---

# ■ 四、代码要求

- 使用Python标准库即可
- 必须可运行
- 提供main.py测试入口
- 每个模块必须可独立测试

---

# ■ 五、输出要求

请输出：

1. 完整代码（按文件结构组织）
2. main.py运行示例
3. README.md（说明架构与执行流程）

---

# ■ 六、禁止事项

- 不允许改变B4流程结构
- 不允许省略模块
- 不允许ToT独立成系统
- 不允许无JSON输出