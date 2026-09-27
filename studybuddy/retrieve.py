from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    """Keyword search (TF-IDF). Simple, no download, works everywhere.
    Used as the Day 1 baseline and as a fallback if the neural model
    (EmbeddingRetriever) can't be loaded."""

    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(chunks)

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        top = scores.argsort()[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


class EmbeddingRetriever:
    """Neural semantic search via ONNX Runtime embeddings (see embed.py).
    Understands meaning, not just shared keywords, and runs on the NPU
    on Snapdragon hardware with onnxruntime-qnn installed."""

    def __init__(self, chunks: list[str], embed_fn=None):
        if embed_fn is None:
            from studybuddy.embed import embed as embed_fn
        self._embed = embed_fn
        self.chunks = chunks
        self.vectors = embed_fn(chunks)

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        query_vec = self._embed([query])[0]
        scores = self.vectors @ query_vec
        top = scores.argsort()[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]
