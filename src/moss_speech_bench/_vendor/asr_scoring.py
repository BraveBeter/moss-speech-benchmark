# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import functools
import string
from jiwer import process_words
from types import SimpleNamespace as SampleOutput

@functools.lru_cache(maxsize=1)
def _get_en_normalizer():
    """Lazy-load the required English WER normalizer from openai-whisper."""
    try:
        from whisper.normalizers import EnglishTextNormalizer
    except ImportError as exc:
        raise RuntimeError(
            "English WER requires openai-whisper "
            "(whisper.normalizers.EnglishTextNormalizer). "
            "Install pinned deps with uv pip install -e ."
        ) from exc

    return EnglishTextNormalizer()

def normalize_text(text: str, lang: str) -> str:
    if lang == "zh":
        try:
            from zhon.hanzi import punctuation as zh_punct
        except ImportError as exc:
            raise RuntimeError(
                "Chinese WER requires zhon (zhon.hanzi.punctuation). "
                "Install pinned deps with uv pip install -e ."
            ) from exc

        all_punct = zh_punct + string.punctuation
        for ch in all_punct:
            if ch == "'":
                continue
            text = text.replace(ch, "")
        text = text.replace(" ", "").replace("\u3000", "").strip()
        text = " ".join(list(text))
        return text

    normalizer = _get_en_normalizer()
    return normalizer(text)

def apply_wer(output: SampleOutput, hyp_text: str, lang: str) -> SampleOutput:
    """Fill output with normalized texts and per-sample WER fields."""
    output.whisper_text = hyp_text
    output.ref_norm = normalize_text(output.target_text, lang)
    output.hyp_norm = normalize_text(hyp_text, lang)
    if not output.ref_norm:
        output.error = "Empty reference after normalization"
        return output
    measures = process_words(output.ref_norm, output.hyp_norm)
    output.wer = measures.wer
    output.substitutions = measures.substitutions
    output.deletions = measures.deletions
    output.insertions = measures.insertions
    output.hits = measures.hits
    output.is_success = True
    return output
