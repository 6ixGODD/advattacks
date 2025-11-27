from __future__ import annotations

import pathlib
import typing as t

import loguru


class AttackStartLog(t.TypedDict):
    event: t.Literal["attack_start"]
    question_id: str
    scenario: str
    question: str


class AttackStepLog(t.TypedDict):
    event: t.Literal["attack_step"]
    question_id: str
    round: int
    model: str
    step: int
    linf_norm: float


class AttackCompleteLog(t.TypedDict):
    event: t.Literal["attack_complete"]
    question_id: str
    final_linf: float
    epsilon: float
    constraint_satisfied: bool
    duration_seconds: float


class GenerationLog(t.TypedDict):
    event: t.Literal["generation"]
    question_id: str
    model: str
    response: str
    is_adversarial: bool


class ErrorLog(t.TypedDict):
    event: t.Literal["error"]
    question_id: str | None
    error_type: str
    error_message: str


def setup_logger(log_file: pathlib.Path) -> loguru.Logger:
    """Setup loguru logger to write JSON lines to file.

    Args:
        log_file: Path to log file.

    Returns:
        Configured logger instance.
    """
    # Remove default handler
    loguru.logger.remove()

    # Add JSON file handler
    loguru.logger.add(log_file, format="{message}", serialize=True, level="INFO")

    return loguru.logger
