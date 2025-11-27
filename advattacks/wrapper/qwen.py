from __future__ import annotations

import pathlib
import typing as t

import torch
import transformers as tfs

from advattacks.wrapper import Wrapper


class QwenWrapper(Wrapper):
    """Wrapper for Qwen2.5-VL-7B-Instruct model."""

    def __init__(self, model_path: str | pathlib.Path, device: torch.device | None = None):
        super().__init__(model_path, device)
        self.model: tfs.Qwen2_5_VLForConditionalGeneration | None = None
        self.processor: tfs.Qwen2_5_VLProcessor | None = None
        self._is_loaded = False

    def load(self) -> None:
        if self._is_loaded:
            return

        self.processor = tfs.AutoProcessor.from_pretrained(
            self.model_path,
            use_fast=True,
            legacy=False,
            trust_remote_code=True,
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
        if not self._is_loaded:
            return

        del self.model
        del self.processor
        self.model = None
        self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._is_loaded = False

    def prepare_inputs(self, image: torch.Tensor, text: str) -> dict[str, torch.Tensor]:
        """Prepare inputs for Qwen.

        Args:
            image: Image tensor of shape (C, H, W) in [0, 1] range.
            text: Text prompt.

        Returns:
            Dictionary of model inputs.
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

    def forward(
        self,
        image: torch.Tensor,
        text: str,
        target_ids: t.Optional[torch.Tensor] = None,  # noqa
    ) -> t.Union[torch.Tensor, t.Tuple[torch.Tensor, torch.Tensor]]:  # noqa
        """Forward pass through Qwen.

        Args:
            image: Image tensor of shape (C, H, W) in [0, 1] range.
            text: Text prompt.
            target_ids: Target token IDs for loss computation.

        Returns:
            If target_ids is None: logits tensor.
            If target_ids is provided: tuple of (loss, logits).
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

    def compute_loss(
        self,
        image: torch.Tensor,
        text: str,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        loss, _ = self.forward(image, text, target_ids)
        return loss

    def compute_gradient(
        self,
        image: torch.Tensor,
        text: str,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        image_adv = image.clone().detach().requires_grad_(True)
        loss = self.compute_loss(image_adv, text, target_ids)
        return torch.autograd.grad(loss, image_adv)[0]

    def generate(
        self,
        image: torch.Tensor,
        text: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.prepare_inputs(image, text)

        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
            )

        return self.processor.decode(generate_ids[0], skip_special_tokens=True)
