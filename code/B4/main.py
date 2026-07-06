from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from B4.core.engine import B4CognitiveEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the B4 Cognitive Engine.")
    parser.add_argument(
        "--goal",
        default="写一个Python学习路线，要求每一步可执行、可验证，优先安排基础语法和项目练习。",
        help="Natural language user goal.",
    )
    parser.add_argument("--max_rounds", type=int, default=2, help="Maximum reflection loop rounds.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = B4CognitiveEngine(max_reflection_rounds=args.max_rounds)
    result = engine.run(args.goal)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
