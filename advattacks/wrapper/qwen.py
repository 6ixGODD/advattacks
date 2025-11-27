from __future__ import annotations

import pathlib
import typing as t

import torch
import transformers as tfs

from advattacks.wrapper import Wrapper


class QwenWrapper(Wrapper):
    """Wrapper for Qwen2. 5-VL-7B-Instruct vision-language model.

    Qwen2.5-VL is Alibaba's latest vision-language model with strong capabilities
    in image understanding and text generation.  It uses a transformer architecture
    optimized for multimodal tasks.

    Attributes:
        model: The Qwen2.5-VL conditional generation model.
        processor: Qwen2.5-VL processor for multimodal input handling.
        tokenizer: Qwen2 tokenizer for text processing.
    """

    model: tfs.Qwen2_5_VLForConditionalGeneration | None
    processor: tfs.Qwen2_5_VLProcessor | None
    tokenizer: tfs.Qwen2Tokenizer | None

    def __init__(self, model_path: str | pathlib.Path, device: torch.device | None = None):
        super().__init__(model_path, device)

    def load(self) -> None:
        """Load Qwen2.5-VL model, processor, and tokenizer into
        memory."""
        if self._is_loaded:
            return

        self.processor = tfs.AutoProcessor.from_pretrained(
            self.model_path,
            use_fast=True,
            legacy=False,
            trust_remote_code=True,
        )

        self.tokenizer = tfs.AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=True,
            padding_side="left",
        )

        self.model = tfs.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self.model.eval()
        self._is_loaded = True

    def unload(self) -> None:
        """Unload model and free GPU memory."""
        if not self._is_loaded:
            return

        del self.model
        del self.processor
        del self.tokenizer
        self.model = None
        self.processor = None
        self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._is_loaded = False

    def prepare_inputs(self, image: torch.Tensor, text: str) -> dict[str, torch.Tensor]:
        """Prepare inputs for Qwen2.5-VL model.

        Qwen2. 5-VL uses a chat template format similar to other instruction-tuned
        models. The processor handles both image and text input formatting.

        Args:
            image: Image tensor of shape (C, H, W) in [0, 1] range.
            text: Text prompt/instruction.

        Returns:
            Dictionary containing tokenized inputs and processed image features.

        Raises:
            RuntimeError: If model is not loaded or processor is not initialized.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        if self.processor is None:
            raise RuntimeError("Processor not initialized.")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(
            images=image,
            text=prompt,
            do_rescale=False,
            return_tensors="pt",
        )

        return {k: v.to(self.device) for k, v in inputs.items()}

    def prepare_tf_inputs(
        self, image: torch.Tensor, question: str, target_prefix: str
    ) -> dict[str, torch.Tensor]:
        """Prepare inputs for teacher-forcing loss computation.

        For Qwen2.5-VL, we use the chat template format to create a conversation
        with both user message and assistant response. This ensures proper
        formatting for teacher-forcing loss computation.

        Args:
            image: Image tensor of shape (C, H, W) in [0, 1] range.
            question: The input question from user.
            target_prefix: Target assistant response prefix.

        Returns:
            Dictionary containing model inputs with labels for loss computation.

        Raises:
            RuntimeError: If model components are not loaded/initialized.
        """
        if not self._is_loaded or not self.processor or not self.tokenizer:
            raise RuntimeError("Model not loaded or components not initialized.")

        # 1. Tokenize target prefix to determine lengths
        prefix_tokens = self.tokenizer(target_prefix, add_special_tokens=False)["input_ids"]

        # 2. Construct conversation with both user and assistant parts
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                ],
            },
            {
                "role": "assistant",
                "content": target_prefix,  # Target response
            },
        ]

        # 3. Apply chat template without generation prompt (we have complete conversation)
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=False)

        # 4. Process inputs
        inputs = self.processor(
            images=image,
            text=prompt,
            do_rescale=False,
            return_tensors="pt",
        )

        # 5. Construct labels for teacher-forcing
        input_ids = inputs["input_ids"][0]  # Remove batch dimension
        labels = input_ids.clone()

        # 6. Mask non-target tokens with -100
        # For Qwen, we compute loss only on the assistant's response tokens
        prefix_len = len(prefix_tokens)
        if prefix_len < len(labels):
            labels[:-prefix_len] = -100  # Mask all except target tokens

        # 7. Add batch dimension back
        inputs["labels"] = labels.unsqueeze(0)

        return {k: v.to(self.device) for k, v in inputs.items()}

    def forward(
        self,
        image: torch.Tensor,
        text: str,
        target_ids: t.Optional[torch.Tensor] = None,  # noqa
    ) -> t.Union[torch.Tensor, t.Tuple[torch.Tensor, torch.Tensor]]:  # noqa
        """Forward pass through Qwen2.5-VL model.

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
            raise RuntimeError("Model not loaded. Call load() first.")
        if self.model is None:
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
        """Compute teacher-forcing loss for Qwen2.5-VL.

        Uses the chat template format to create proper conversation structure
        and computes loss only on the assistant's response tokens.

        Args:
            image: Image tensor of shape (C, H, W) in [0, 1] range.
            question: The input question from user.
            target_prefix: Target assistant response prefix to optimize for.

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
