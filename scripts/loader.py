from __future__ import annotations

import json
import pathlib
import typing as t

import PIL.Image
import torch
import torchvision.transforms as transforms


class DataSample(t.TypedDict):
    question_id: str
    scenario: str
    question: str
    image_path: pathlib.Path
    image: torch.Tensor


def load_dataset(data_dir: pathlib.Path) -> list[DataSample]:
    """Load the competition dataset.

    Args:
        data_dir: Root directory containing 'questions' and 'imgs'
            subdirectories.

    Returns:
        List of data samples.
    """
    questions_dir = data_dir / "questions"
    images_dir = data_dir / "imgs"

    samples: list[DataSample] = []
    to_tensor = transforms.ToTensor()

    # Iterate through all JSON files in questions directory
    for json_file in sorted(questions_dir.glob("*. json")):
        scenario = json_file.stem  # e.g., "01-Illegal_Activity"

        # Load questions
        with json_file.open(encoding="utf-8") as f:
            questions = json.load(f)

        # Load corresponding images
        scenario_img_dir = images_dir / scenario

        for question_id, question_data in questions.items():
            image_path = scenario_img_dir / f"{question_id}.png"

            if not image_path.exists():
                continue

            # Load image
            img = PIL.Image.open(image_path).convert("RGB")
            image_tensor = to_tensor(img)

            samples.append({
                "question_id": question_id,
                "scenario": scenario,
                "question": question_data["Question"],
                "image_path": image_path,
                "image": image_tensor,
            })

    return samples


def load_prefixes(prefix_file: pathlib.Path) -> list[str]:
    """Load target prefixes from a text file.

    Args:
        prefix_file: Path to prefix file (one prefix per line).

    Returns:
        List of prefix strings.

    Raises:
        FileNotFoundError: If prefix file doesn't exist.
        ValueError: If file is empty or contains only whitespace.
    """
    if not prefix_file.exists():
        raise FileNotFoundError(f"Prefix file not found: {prefix_file}")

    with prefix_file.open(encoding="utf-8") as f:
        lines = f.readlines()

    # Filter out empty lines and strip whitespace
    prefixes = [line.strip() for line in lines if line.strip()]

    if not prefixes:
        raise ValueError(f"Prefix file is empty: {prefix_file}")

    return prefixes
