from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import time
import typing as t

from scripts import display
from scripts import evaluator
from scripts import loader
from scripts import logger
from scripts import visualizer
from scripts.evaluator import ComparisonMetrics
import torch
import torchvision.transforms as transforms

from advattacks.attack.pgd import PGD
from advattacks.wrapper.instructblip import InstructBlipWrapper
from advattacks.wrapper.llava import LlavaWrapper
from advattacks.wrapper.qwen import QwenWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run adversarial attacks on vision-language models"
    )

    parser.add_argument(
        "-i",
        "--input",
        type=pathlib.Path,
        default=pathlib.Path("data"),
        help="Input data directory (default: data)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("output"),
        help="Output directory (default: output)",
    )

    parser.add_argument(
        "--models-dir",
        type=pathlib.Path,
        default=pathlib.Path("models"),
        help="Directory containing model files (default: models)",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=32 / 255,
        help="Maximum L-infinity perturbation (default: 32/255)",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=2 / 255,
        help="PGD step size (default: 2/255)",
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=4,
        help="Number of attack rounds (default: 4)",
    )

    parser.add_argument(
        "--steps-per-model",
        type=int,
        default=5,
        help="PGD steps per model per round (default: 5)",
    )

    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip model generation phase",
    )

    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip visualization generation",
    )

    return parser.parse_args()


def setup_dirs(base_output: pathlib.Path) -> dict[str, pathlib.Path]:
    """Setup output directory structure.

    Args:
        base_output: Base output directory.

    Returns:
        Dictionary of output paths.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_output / timestamp

    dirs = {
        "run": run_dir,
        "advimages": run_dir / "advimages",
        "responses": run_dir / "responses",
        "visualizations": run_dir / "visualizations",
        "metrics": run_dir / "metrics",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def init_wrappers(models_dir: pathlib.Path) -> list[t.Any]:
    """Initialize model wrappers.

    Args:
        models_dir: Directory containing model files.

    Returns:
        List of initialized wrappers.
    """
    return [
        InstructBlipWrapper(models_dir / "Salesforce" / "instructblip-vicuna-7b"),
        LlavaWrapper(models_dir / "llava-hf" / "llava-1.5-7b-hf"),
        QwenWrapper(models_dir / "Qwen" / "Qwen2. 5-VL-7B-Instruct"),
    ]


def save_advimage(image: torch.Tensor, output_path: pathlib.Path) -> None:
    """Save adversarial image to disk.

    Args:
        image: Image tensor (C, H, W) in [0, 1] range.
        output_path: Path to save the image.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    to_pil = transforms.ToPILImage()
    pil_image = to_pil(image)
    pil_image.save(output_path)


def gen_response(
    wrappers: list[t.Any],
    image: torch.Tensor,
    text: str,
    question_id: str,
    log: logger.loguru.Logger,
    is_adversarial: bool = False,
) -> dict[str, str]:
    """Generate responses from all models.

    Args:
        wrappers: List of model wrappers.
        image: Image tensor.
        text: Text prompt.
        question_id: Question identifier.
        log: Logger instance.
        is_adversarial: Whether this is an adversarial image.

    Returns:
        Dictionary mapping model names to responses.
    """
    responses = {}
    wrapper_names = ["InstructBLIP", "LLaVA", "Qwen"]

    for wrapper, name in zip(wrappers, wrapper_names, strict=False):
        wrapper.load()
        response = wrapper.generate(image, text)
        wrapper.unload()

        responses[name] = response

        log.bind(
            **logger.GenerationLog(
                event="generation",
                question_id=question_id,
                model=name,
                response=response,
                is_adversarial=is_adversarial,
            )
        ).info("")

    return responses


def main() -> None:
    """Main execution function."""
    args = parse_args()

    # Setup output directories
    display.header("Adversarial Attack Pipeline")
    display.info(f"Input directory: {args.input}")
    display.info(f"Output directory: {args.output}")

    dirs = setup_dirs(args.output)
    display.success(f"Created output directory: {dirs['run']}")

    # Setup logger
    log = logger.setup_logger(dirs["run"] / "log.jsonl")
    display.success("Initialized logger")

    # Load dataset
    display.step("Loading dataset", 1)
    samples = loader.load_dataset(args.input)
    display.success(f"Loaded {len(samples)} samples")

    # Initialize wrappers
    display.step("Initializing model wrappers", 2)
    wrappers = init_wrappers(args.models_dir)
    display.success(f"Initialized {len(wrappers)} model wrappers")

    # Initialize attack
    display.step("Creating attack", 3)
    attack = PGD(
        wrappers=wrappers,
        epsilon=args.epsilon,
        alpha=args.alpha,
        num_rounds=args.rounds,
        steps_per_model=args.steps_per_model,
    )
    display.success("Initialized PGD attack")

    # Process each sample
    display.step("Running attacks", 4)
    print()

    all_metrics: list[ComparisonMetrics] = []

    for idx, sample in enumerate(samples, 1):
        question_id = sample["question_id"]
        scenario = sample["scenario"]
        question = sample["question"]
        original_image = sample["image"]

        display.header(f"Sample {idx}/{len(samples)}: {scenario}/{question_id}")
        display.info(f"Question: {question[:80]}...")

        # Log attack start
        log.bind(
            **logger.AttackStartLog(
                event="attack_start",
                question_id=question_id,
                scenario=scenario,
                question=question,
            )
        ).info("")

        # Run attack
        start_time = time.time()

        try:
            adversarial_image = attack(original_image, question, verbose=False)
            duration = time.time() - start_time

            final_linf = torch.norm(adversarial_image - original_image, p=float("inf")).item()
            constraint_satisfied = final_linf <= args.epsilon

            # Log attack completion
            log.bind(
                **logger.AttackCompleteLog(
                    event="attack_complete",
                    question_id=question_id,
                    final_linf=final_linf,
                    epsilon=args.epsilon,
                    constraint_satisfied=constraint_satisfied,
                    duration_seconds=duration,
                )
            ).info("")

            if constraint_satisfied:
                display.success(
                    f"Attack complete: L∞={final_linf:.6f} (≤ {args.epsilon:.6f}) in {duration:.1f}s"
                )
            else:
                display.warning(
                    f"Attack complete: L∞={final_linf:.6f} (> {args.epsilon:.6f}) in {duration:.1f}s"
                )

            # Save adversarial image
            adv_output_dir = dirs["advimages"] / scenario
            adv_output_path = adv_output_dir / f"{question_id}.png"
            save_advimage(adversarial_image, adv_output_path)
            display.success(f"Saved adversarial image: {adv_output_path}")

            # Generate responses
            if not args.skip_generation:
                display.info("Generating model responses...")

                # Original responses
                orig_responses = gen_response(
                    wrappers,
                    original_image,
                    question,
                    question_id,
                    log,
                    is_adversarial=False,
                )

                # Adversarial responses
                adv_responses = gen_response(
                    wrappers,
                    adversarial_image,
                    question,
                    question_id,
                    log,
                    is_adversarial=True,
                )

                # Save responses
                responses_data = {
                    "question_id": question_id,
                    "scenario": scenario,
                    "question": question,
                    "original_responses": orig_responses,
                    "adversarial_responses": adv_responses,
                }

                response_output_dir = dirs["responses"] / scenario
                response_output_dir.mkdir(parents=True, exist_ok=True)
                response_output_path = response_output_dir / f"{question_id}.json"

                with response_output_path.open("w", encoding="utf-8") as f:
                    json.dump(responses_data, f, indent=2, ensure_ascii=False)

                display.success(f"Saved responses: {response_output_path}")

                # Compute metrics (using first model's responses as representative)
                first_model = next(iter(orig_responses.keys()))
                metrics = evaluator.compute_metrics(
                    question_id,
                    scenario,
                    original_image,
                    adversarial_image,
                    orig_responses[first_model],
                    adv_responses[first_model],
                )
                all_metrics.append(metrics)

            # Create visualization
            if not args.skip_visualization:
                perturbation = adversarial_image - original_image
                vis_output_dir = dirs["visualizations"] / scenario
                vis_output_dir.mkdir(parents=True, exist_ok=True)
                vis_output_path = vis_output_dir / f"{question_id}.png"

                visualizer.visualize_comparison(
                    original_image,
                    adversarial_image,
                    perturbation,
                    question_id,
                    scenario,
                    vis_output_path,
                )
                display.success(f"Saved visualization: {vis_output_path}")

        except Exception as e:
            display.error(f"Failed to process sample: {e}")
            log.bind(
                **logger.ErrorLog(
                    event="error",
                    question_id=question_id,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
            ).error("")
            continue

        print()

    # Save metrics summary
    if all_metrics:
        display.step("Saving metrics summary", 5)
        metrics_path = dirs["metrics"] / "metrics. json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)
        display.success(f"Saved metrics: {metrics_path}")

        # Create summary visualization
        if not args.skip_visualization:
            summary_plot_path = dirs["metrics"] / "summary. png"
            visualizer.plot_metrics_summary(all_metrics, summary_plot_path)
            display.success(f"Saved summary plot: {summary_plot_path}")

    # Final summary
    display.header("Pipeline Complete")
    display.success(f"Processed {len(samples)} samples")
    display.info(f"Results saved to: {dirs['run']}")


if __name__ == "__main__":
    main()
