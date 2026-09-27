"""Flashcard generation.
With AI answers off (or unavailable): cloze-style cards built with no
model — blanks out a key term in a sentence, entirely offline and instant.
With AI answers on: asks the local Phi-3 model to write real Q&A flashcards.
"""
import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z][A-Za-z\-]{5,}")
_QA_PAIR = re.compile(r"Q:\s*(.+?)\s*A:\s*(.+?)(?=Q:|$)", flags=re.S)


def flashcards(chunks: list[str], use_llm: bool = False, num_cards: int = 5) -> list[dict]:
    if use_llm:
        try:
            cards = _llm_flashcards(chunks, num_cards)
            if cards:
                return cards
        except Exception:
            pass  # fall through to the heuristic version below
    return _heuristic_flashcards(chunks, num_cards)


def _llm_flashcards(chunks: list[str], num_cards: int) -> list[dict]:
    from studybuddy.llm import generate

    text = "\n\n".join(chunks)[:6000]
    prompt = (
        "<|user|>\n"
        f"Create exactly {num_cards} study flashcards from the notes below. "
        "Use only this exact format, one pair per card and nothing else:\n"
        "Q: <question>\nA: <answer>\n\n"
        f"Notes:\n{text}<|end|>\n<|assistant|>"
    )
    raw = generate(prompt, max_new_tokens=500)
    return _parse_flashcards(raw)


def _parse_flashcards(raw: str) -> list[dict]:
    cards = []
    for q, a in _QA_PAIR.findall(raw):
        q, a = q.strip(), a.strip()
        if q and a:
            cards.append({"question": q, "answer": a})
    return cards


def _heuristic_flashcards(chunks: list[str], num_cards: int = 5) -> list[dict]:
    text = " ".join(chunks)
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if len(s.strip()) > 30]

    cards = []
    for s in sentences:
        words = _WORD.findall(s)
        if not words:
            continue
        term = max(words, key=len)  # longest word = likely key term
        question = re.sub(rf"\b{re.escape(term)}\b", "____", s, count=1)
        if question == s:
            continue
        cards.append({"question": question, "answer": term})
        if len(cards) >= num_cards:
            break
    return cards
