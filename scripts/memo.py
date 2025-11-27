from __future__ import annotations

import time

import torch


def print_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"CUDA Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def monitor_during_attack():
    while True:
        print_memory_usage()
        time.sleep(5)
