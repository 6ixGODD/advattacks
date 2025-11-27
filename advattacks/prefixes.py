"""Target prefix management for adversarial attacks."""

from __future__ import annotations

import typing as t

import torch

if t.TYPE_CHECKING:
    from advattacks.wrapper import Wrapper


# Default target prefixes for jailbreak attacks
DEFAULT_PREFIXES = [
    "Sure, here is how to",
    "Sure, here's how to",
    "Absolutely!  Here's",
    "Of course! Here is",
    "Certainly! Here's how",
]


def tokenize_prefixes(prefixes: t.Sequence[str], wrapper: Wrapper) -> list[torch.Tensor]:
    """Tokenize target prefixes using model's tokenizer.

    Args:
        prefixes: Sequence of target prefix strings.
        wrapper: Model wrapper containing processor/tokenizer.

    Returns:
        List of tokenized tensors, each of shape (seq_len,).
    """
    if not wrapper.is_loaded:
        raise RuntimeError("Wrapper must be loaded to access tokenizer.")

    tokenized = []

    for prefix in prefixes:
        # Use tokenizer
        tokens = wrapper.tokenizer(prefix, return_tensors="pt", add_special_tokens=False)

        # Extract token IDs and squeeze batch dimension
        token_ids = tokens["input_ids"].squeeze(0)
        tokenized.append(token_ids)

    return tokenized


def get_tokenized_prefixes(
    wrappers: t.Sequence[Wrapper],
    prefixes: t.Sequence[str] | None = None,
) -> dict[Wrapper, list[torch.Tensor]]:
    """Get tokenized prefixes for each wrapper.

    Args:
        wrappers: Sequence of model wrappers.
        prefixes: Sequence of target prefix strings. If None, uses
            DEFAULT_PREFIXES.

    Returns:
        Dictionary mapping each wrapper to its tokenized prefixes.
    """
    if prefixes is None:
        prefixes = DEFAULT_PREFIXES

    tokenized_map = {}

    for wrapper in wrappers:
        if not wrapper.is_loaded:
            wrapper.load()

        tokenized_map[wrapper] = tokenize_prefixes(prefixes, wrapper)

    return tokenized_map
