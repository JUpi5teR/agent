from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class B4Plugin(ABC):
    """Unified interface implemented by every B4 plugin."""

    name: str

    @abstractmethod
    def run(self, payload: Any) -> Any:
        """Run the plugin with a structured payload and return structured data."""

