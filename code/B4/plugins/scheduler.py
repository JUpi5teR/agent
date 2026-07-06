from __future__ import annotations

from typing import Any

from B4.interfaces.plugin import B4Plugin


class SchedulerPlugin(B4Plugin):
    name = "scheduler"

    def run(self, payload: list[dict[str, Any]]) -> list[dict[str, int | str]]:
        if not isinstance(payload, list):
            raise ValueError("Scheduler input must be a plan array")

        sorted_plan = sorted(payload, key=lambda item: int(item.get("id", 0)))
        schedule = []
        for order, item in enumerate(sorted_plan, 1):
            task = item.get("task")
            if not isinstance(task, str) or not task.strip():
                raise ValueError("Scheduler received invalid task")
            schedule.append({"id": int(item.get("id", order)), "task": task, "order": order})
        return schedule

