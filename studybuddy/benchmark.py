"""How fast is embedding search on this machine? Reports which ONNX Runtime
execution providers are available and times the embedding step. On a
Snapdragon laptop with onnxruntime-qnn installed, this shows the NPU
(QNNExecutionProvider) actually being used; elsewhere it reports CPU."""
import time

import onnxruntime as ort


def available_providers() -> list[str]:
    return ort.get_available_providers()


def benchmark_embeddings(sample_texts: list[str], repeats: int = 3) -> dict:
    from studybuddy import embed as embed_module

    embed_module._cache.clear()  # force a fresh session so provider choice is re-checked

    start = time.perf_counter()
    embed_module.embed(sample_texts)  # warm-up: includes model load / first-run cost
    warmup = time.perf_counter() - start

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        embed_module.embed(sample_texts)
        times.append(time.perf_counter() - t0)

    return {
        "providers": available_providers(),
        "warmup_seconds": round(warmup, 3),
        "avg_seconds": round(sum(times) / len(times), 3) if times else None,
        "runs_seconds": [round(t, 3) for t in times],
        "num_texts": len(sample_texts),
    }
