from __future__ import annotations

import sys

from scripts.ansi import ANSI


def success(message: str, /, prefix: str = "✓") -> None:
    """Print a success message."""
    print(f"{ANSI.success(prefix)} {message}")


def error(message: str, /, prefix: str = "✗") -> None:
    """Print an error message."""
    print(f"{ANSI.error(prefix)} {message}", file=sys.stderr)


def warning(message: str, /, prefix: str = "⚠") -> None:
    """Print a warning message."""
    print(f"{ANSI.warning(prefix)} {message}")


def info(message: str, /, prefix: str = "•") -> None:
    """Print an info message."""
    print(f"{ANSI.info(prefix)} {message}")


def header(text: str, /, width: int = 60) -> None:
    """Print a section header."""
    print()
    print(ANSI.highlight(text))
    print(ANSI.format("-" * min(len(text), width), ANSI.STYLE.DIM))


def step(message: str, /, step_num: int | None = None) -> None:
    """Print a step message in a process."""
    if step_num is not None:
        prefix = ANSI.format(f"[{step_num}]", ANSI.FG.CYAN, ANSI.STYLE.BOLD)
        print(f"{prefix} {message}")
    else:
        print(f"► {message}")


def progress(current: int, total: int, /, label: str = "", width: int = 40) -> None:
    """Print a progress bar."""
    percent = current / total if total > 0 else 0
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)

    parts = []
    if label:
        parts.append(label)
    parts.append(f"[{bar}]")
    parts.append(f"{percent * 100:5.1f}%")
    parts.append(f"({current}/{total})")

    print("\r" + " ".join(parts), end="", flush=True)

    if current >= total:
        print()
