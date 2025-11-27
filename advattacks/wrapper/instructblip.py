from __future__ import annotations

import pathlib
import typing as t

import torch
import transformers as tfs

from advattacks.wrapper import Wrapper


class InstructBlipWrapper(Wrapper):
    """Wrapper for Salesforce InstructBLIP-Vicuna-7B model.

    InstructBLIP combines BLIP-2 with Vicuna language model for instruction-following
    visual question answering. It uses a Q-Former to bridge vision and language
    representations.

    Attributes:
        model: The InstructBLIP conditional generation model.
        processor: InstructBLIP processor for input preparation and decoding.
        tokenizer: GPT-2 based tokenizer used by InstructBLIP.
    """

    model: tfs.InstructBlipForConditionalGeneration | None
    processor: tfs.InstructBlipProcessor | None
    tokenizer: tfs.GPT2Tokenizer | None

    def __init__(self, model_path: str | pathlib.Path, device: torch.device | None = None):
        super().__init__(model_path, device)

    def load(self) -> None:
        """Load InstructBLIP model, processor, and tokenizer into
        memory."""
        if self._is_loaded:
            return

        self.processor = tfs.InstructBlipProcessor.from_pretrained(
            self.model_path,
            use_fast=True,
            padding_side="left",
        )

        self.tokenizer = tfs.AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=True,
            padding_side="left",
        )

        self.model = tfs.InstructBlipForConditionalGeneration.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self.model.eval()
        self._is_loaded = True

    def prepare_inputs(self, image: torch.Tensor, text: str) -> dict[str, torch.Tensor]:
        """Prepare inputs for InstructBLIP model.

        InstructBLIP uses a simple text prompt format without complex chat templates.
        The processor handles image preprocessing and text tokenization.

        Args:
            image: Image tensor of shape (C, H, W) in [0, 1] range.
            text: Text prompt/instruction.

        Returns:
            Dictionary containing 'input_ids', 'attention_mask', and 'pixel_values'.

        Raises:
            RuntimeError: If model is not loaded or processor is not initialized.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        if not self.processor:
            raise RuntimeError("Processor not initialized.")

        inputs = self.processor(
            images=image,
            text=text,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )

        return {k: v.to(self.device) for k, v in inputs.items()}

    def prepare_tf_inputs(
        self,
        image: torch.Tensor,
        question: str,
        target_prefix: str,
    ) -> dict[str, torch.Tensor]:
        """Prepare inputs for teacher-forcing loss computation.

        For InstructBLIP, we construct the input by concatenating the question
        with the target prefix. Since InstructBLIP doesn't use complex chat
        templates, we directly append the target to the question.

        Args:
            image: Image tensor of shape (C, H, W) in [0, 1] range.
            question: The input question/instruction.
            target_prefix: Target response prefix (e.g., "Sure, here is how to").

        Returns:
            Dictionary containing model inputs with proper labels for loss computation.

        Raises:
            RuntimeError: If model components are not loaded/initialized.
        """
        if not self._is_loaded or not self.processor or not self.tokenizer:
            raise RuntimeError("Model not loaded or components not initialized.")

        # 1. Tokenize target prefix to determine lengths
        prefix_tokens = self.tokenizer(target_prefix, add_special_tokens=False)["input_ids"]

        # 2. Construct full sequence: question + space + target prefix
        full_text = question + " " + target_prefix

        # 3. Process the full sequence with image
        inputs = self.processor(
            images=image,
            text=full_text,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )

        # 4.  Construct labels for teacher-forcing
        input_ids = inputs["input_ids"][0]  # Remove batch dimension
        labels = input_ids.clone()

        # 5. Mask question part with -100 (don't compute loss on input)
        # Only compute loss on the target prefix tokens at the end
        prefix_len = len(prefix_tokens)
        if prefix_len < len(labels):
            labels[:-prefix_len] = -100  # Mask all tokens except target prefix

        # 6. Add batch dimension back
        inputs["labels"] = labels.unsqueeze(0)

        return {k: v.to(self.device) for k, v in inputs.items()}

    def forward(
        self,
        image: torch.Tensor,
        text: str,
        target_ids: t.Optional[torch.Tensor] = None,  # noqa
    ) -> t.Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:  # noqa
        """Forward pass through InstructBLIP model.

        Args:
            image: Image tensor of shape (C, H, W) in [0, 1] range.
            text: Text prompt/instruction.
            target_ids: Optional target token IDs for loss computation (legacy interface).

        Returns:
            If target_ids is None: logits tensor of shape (batch_size, seq_len, vocab_size).
            If target_ids is provided: tuple of (loss, logits).

        Raises:
            RuntimeError: If model is not loaded or initialized.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded.  Call load() first.")
        if not self.model:
            raise RuntimeError("Model not initialized.")

        inputs = self.prepare_inputs(image, text)

        if target_ids is not None:
            inputs["labels"] = target_ids.to(self.device)

        outputs = self.model(**inputs)

        if target_ids is not None:
            return outputs.loss, outputs.logits
        return outputs.logits

    def compute_tfloss(
        self,
        image: torch.Tensor,
        question: str,
        target_prefix: str,
    ) -> torch.Tensor:
        """Compute teacher-forcing loss for InstructBLIP.

        This method constructs proper input-output pairs and computes the loss
        only on the target prefix tokens, not the entire sequence.

        Args:
            image: Image tensor of shape (C, H, W) in [0, 1] range.
            question: The input question/instruction.
            target_prefix: Target response prefix to optimize for.

        Returns:
            Scalar tensor representing the teacher-forcing loss.

        Raises:
            RuntimeError: If model is not loaded or initialized.
        """
        if not self._is_loaded or not self.model:
            raise RuntimeError("Model not loaded or not initialized.")

        inputs = self.prepare_tf_inputs(image, question, target_prefix)
        outputs = self.model(**inputs)

        return outputs.loss
