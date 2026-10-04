"""Translate user prompts into English before they reach Text2CAD."""

from __future__ import annotations

import re
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]

TRANSLATION_MODELS = {
    "tr": REPO_ROOT / "Translarion" / "Models" / "TurkishToEnglishModel" / "final",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class Translator:
    def __init__(self, source_lang: str = "tr", model_dir: str | Path | None = None, device: str | None = None):
        if model_dir is None:
            if source_lang not in TRANSLATION_MODELS:
                raise ValueError(
                    f"No translation model for '{source_lang}'. Available: {sorted(TRANSLATION_MODELS)}"
                )
            model_dir = TRANSLATION_MODELS[source_lang]
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Translation model not found: {model_dir}")

        self.source_lang = source_lang
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(self.device).eval()

    @torch.no_grad()
    def generate(self, text: str, max_length: int = 128, num_beams: int = 4) -> str:
        """Translate `text` to English.

        The model was fine-tuned on short sentence pairs (max 128 tokens),
        so multi-sentence prompts are translated sentence by sentence.
        """
        sentences = [s for s in _SENTENCE_SPLIT.split(" ".join(text.split())) if s]
        if not sentences:
            return ""
        batch = self.tokenizer(
            sentences, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(self.device)
        output_ids = self.model.generate(**batch, max_length=max_length, num_beams=num_beams)
        return " ".join(self.tokenizer.batch_decode(output_ids, skip_special_tokens=True))
