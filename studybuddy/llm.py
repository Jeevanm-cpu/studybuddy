"""Local AI-generated answers via Phi-3-mini, running through ONNX Runtime
GenAI. Downloads the CPU/NPU-ready model (~2.7GB) from Hugging Face on first
use, then runs fully offline. Falls back to raw passages if the model isn't
installed, hasn't downloaded yet, or generation fails for any reason, so the
app never crashes even without internet or enough disk space.
"""
import os

MODEL_REPO = "microsoft/Phi-3-mini-4k-instruct-onnx"
SUBFOLDER = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
LOCAL_ROOT = os.path.join(os.path.expanduser("~"), ".studybuddy_models", "phi3-mini-cpu-int4")

_cache: dict = {}


def answer(question: str, passages: list[str], use_llm: bool = False) -> str:
    """Return an answer for `question` grounded in `passages`.
    With use_llm=False (or if the model can't be loaded), returns the raw
    matching passages instead of a generated answer."""
    context = "\n\n".join(passages)

    if not use_llm:
        return "(Showing best matching passages. Turn on AI answers above for a written summary.)\n\n" + context

    try:
        return _generate_answer(question, context)
    except Exception as e:
        return (
            "(Couldn't use the local AI model, showing best matching passages instead. "
            f"Reason: {e})\n\n" + context
        )


def _load():
    if "model" in _cache:
        return _cache["model"], _cache["tokenizer"], _cache["stream"]

    import onnxruntime_genai as og
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_REPO,
        allow_patterns=[f"{SUBFOLDER}/*"],
        local_dir=LOCAL_ROOT,
    )
    model_dir = os.path.join(LOCAL_ROOT, SUBFOLDER)

    model = og.Model(model_dir)
    tokenizer = og.Tokenizer(model)
    stream = tokenizer.create_stream()

    _cache.update(model=model, tokenizer=tokenizer, stream=stream)
    return model, tokenizer, stream


def _generate_answer(question: str, context: str) -> str:
    prompt = (
        "<|user|>\n"
        "Answer the question using only the context below. "
        "If the answer isn't in the context, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}<|end|>\n<|assistant|>"
    )
    return generate(prompt, max_new_tokens=400)


def generate(prompt: str, max_new_tokens: int = 400) -> str:
    """Run the local model on any prompt and return the generated text.
    Raises on failure (model missing, download failed, etc.) — callers
    that want a graceful fallback should catch exceptions themselves,
    the way `answer()` above does."""
    import onnxruntime_genai as og

    model, tokenizer, stream = _load()
    tokens = tokenizer.encode(prompt)

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(tokens) + max_new_tokens)

    # The generate() API is a preview API and its token-feeding method has
    # changed across onnxruntime-genai releases. Support both.
    if hasattr(og.Generator, "append_tokens"):
        generator = og.Generator(model, params)
        generator.append_tokens(tokens)

        def step():
            generator.generate_next_token()
    else:
        params.input_ids = tokens
        generator = og.Generator(model, params)

        def step():
            generator.compute_logits()
            generator.generate_next_token()

    out = []
    while not generator.is_done():
        step()
        out.append(stream.decode(generator.get_next_tokens()[0]))
    return "".join(out).strip()
