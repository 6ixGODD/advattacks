from __future__ import annotations

import pathlib
import typing as t

import torch
import transformers as tfs

from advattacks.wrapper import Wrapper


class LlavaWrapper(Wrapper):
    model: tfs.LlavaForConditionalGeneration | None
    processor: tfs.LlavaProcessor | None
    tokenizer: tfs.LlamaTokenizer | None

    def __init__(self, model_path: str | pathlib.Path, device: torch.device | None = None):
        super().__init__(model_path, device)

    def load(self) -> None:
        if self._is_loaded:
            return

        self.processor = tfs.AutoProcessor.from_pretrained(
            self.model_path,
            use_fast=True,
            padding_side="left",
        )

        self.tokenizer = tfs.AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=True,
            padding_side="left",
        )

        self.model = tfs.LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self.model.eval()
        self._is_loaded = True

    def unload(self) -> None:
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
        """Prepare inputs for LLaVA."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        if not self.processor:
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
        """Prepare inputs for teacher-forcing with correct labels."""
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
        input_ids = inputs["input_ids"][0]  # 移除batch维度
        labels = input_ids.clone()

        # 6. Mask non-target tokens with -100
        prefix_len = len(prefix_tokens)
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
        """Forward pass through LLaVA."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
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
        """Compute teacher-forcing loss for LLaVA."""
        if not self._is_loaded or not self.model:
            raise RuntimeError("Model not loaded or not initialized.")

        inputs = self.prepare_tf_inputs(image, question, target_prefix)
        outputs = self.model(**inputs)

        return outputs.loss
