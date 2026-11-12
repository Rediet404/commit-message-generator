"""CodeT5 generation for docstrings and commit messages with explicit input-length handling."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from commitgen.preprocessing.style import format_style


@dataclass(frozen=True)
class GenerationSettings:
    model_name: str = "Salesforce/codet5-small"
    max_input_tokens: int = 512
    max_output_tokens: int = 32
    num_beams: int = 4
    device: str = "cpu"
    input_prefix: str = "summarize: "
    style_mood: str | None = None  # "imperative" | "descriptive" | None
    style_length: str | None = None  # "short" | "detailed" | None


@dataclass
class Prediction:
    text: str
    input_tokens: int
    truncated: bool
    inference_time_seconds: float
    status: str
    error: str | None = None


class CodeT5Generator:
    """Lazy-loading CodeT5 wrapper supporting both CodeT5-small and codet5-base-multi-sum."""

    def __init__(self, settings: GenerationSettings | None = None, model: Any = None, tokenizer: Any = None):
        self.settings = settings or GenerationSettings()
        self._model = model
        self._tokenizer = tokenizer
        self._load_error: RuntimeError | None = None

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        if self._load_error is not None:
            raise self._load_error
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(self.settings.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.settings.model_name)
            self._model.to(torch.device(self.settings.device))
            self._model.eval()
        except Exception as exc:
            self._load_error = RuntimeError(
                f"Unable to load {self.settings.model_name}. Check transformers/torch installation, "
                f"network access, and the model cache. Original error: {exc}"
            )
            raise self._load_error from exc

    def prepare_input(self, text: str) -> tuple[Any, int, bool]:
        self._load()
        encoded = self._tokenizer(
            self.settings.input_prefix + text,
            return_tensors="pt",
            truncation=False,
        )
        input_tokens = int(encoded["input_ids"].shape[-1])
        truncated = input_tokens > self.settings.max_input_tokens
        if truncated:
            encoded = self._tokenizer(
                self.settings.input_prefix + text,
                return_tensors="pt",
                truncation=True,
                max_length=self.settings.max_input_tokens,
            )
        return encoded, input_tokens, truncated

    def generate(self, input_text: str) -> Prediction:
        """Generate a one-line summary from function code or a git diff."""
        started = time.perf_counter()
        try:
            self._load()
            encoded, input_tokens, truncated = self.prepare_input(input_text)
            generated = self._model.generate(
                **encoded,
                max_new_tokens=self.settings.max_output_tokens,
                num_beams=self.settings.num_beams,
                do_sample=False,
                early_stopping=True,
            )
            text = self._tokenizer.decode(generated[0], skip_special_tokens=True).strip()

            # Apply style transformations if configured
            if text and (self.settings.style_mood or self.settings.style_length):
                text = format_style(
                    text,
                    mood=self.settings.style_mood,
                    length=self.settings.style_length,
                )

            status = "ok" if text else "empty"
            return Prediction(text, input_tokens, truncated, time.perf_counter() - started, status)
        except Exception as exc:
            return Prediction(
                "",
                0,
                False,
                time.perf_counter() - started,
                "error",
                str(exc),
            )

    def generate_docstring(self, code: str) -> Prediction:
        """Generate a one-line docstring description from function code."""
        return self.generate(code)

    def generate_commit_message(self, diff: str) -> Prediction:
        """Generate a one-line commit subject from a git diff."""
        return self.generate(diff)
