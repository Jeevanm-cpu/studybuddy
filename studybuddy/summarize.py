"""Notes summarization.
With AI answers off (or unavailable): a no-download extractive summary —
picks the sentences most central to the document using TF-IDF, entirely
offline and instant.
With AI answers on: asks the local Phi-3 model for a proper written summary.
"""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def summarize(chunks: list[str], use_llm: bool = False, num_sentences: int = 5) -> str:
    if use_llm:
        try:
            return _llm_summary(chunks)
        except Exception as e:
            return _extractive_summary(chunks, num_sentences) + f"\n\n(AI summary unavailable: {e})"
    return _extractive_summary(chunks, num_sentences)


def _llm_summary(chunks: list[str]) -> str:
    from studybuddy.llm import generate

    text = "\n\n".join(chunks)[:6000]  # keep the prompt a manageable size
    prompt = (
        "<|user|>\nSummarize the following notes in 4 to 6 concise bullet points, "
        f"covering only the most important ideas.\n\nNotes:\n{text}<|end|>\n<|assistant|>"
    )
    return generate(prompt, max_new_tokens=300)


def _extractive_summary(chunks: list[str], num_sentences: int = 5) -> str:
    text = " ".join(chunks)
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if len(s.strip()) > 20]
    if not sentences:
        return "Not enough text to summarize."
    if len(sentences) <= num_sentences:
        return "\n".join(f"- {s}" for s in sentences)

    matrix = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    centroid = matrix.mean(axis=0)
    scores = np.asarray(matrix.dot(centroid.T)).ravel()

    top_idx = sorted(np.argsort(scores)[::-1][:num_sentences])  # restore reading order
    return "\n".join(f"- {sentences[i]}" for i in top_idx)
