import streamlit as st

from studybuddy.benchmark import benchmark_embeddings
from studybuddy.flashcards import flashcards
from studybuddy.ingest import chunk_text, read_pdf
from studybuddy.llm import answer
from studybuddy.retrieve import EmbeddingRetriever, Retriever
from studybuddy.summarize import summarize

st.set_page_config(page_title="StudyBuddy", page_icon="📚", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 760px;}

    /* Hero header */
    .hero {
        display: flex; align-items: center; gap: 16px;
        padding: 18px 22px; margin-bottom: 20px; border-radius: 16px;
        background: linear-gradient(135deg, #0F9D7414, #0F9D7404);
        border: 1px solid #0F9D7428;
    }
    .hero-icon {
        font-size: 28px; min-width: 54px; height: 54px; border-radius: 14px;
        background: #0F9D74; display: flex; align-items: center; justify-content: center;
        box-shadow: 0 4px 10px #0F9D7440;
    }
    .hero-text h1 { margin: 0; font-size: 25px; font-weight: 700; color: #14322A; }
    .hero-text p { margin: 2px 0 0; font-size: 14px; color: #5B6B65; }

    /* Status badge */
    .badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 5px 14px; border-radius: 999px;
        font-size: 13px; font-weight: 600; margin-bottom: 14px;
    }
    .badge-ok   {background: #0F9D7418; color: #0B7D5D;}
    .badge-warn {background: #C7770018; color: #8A5300;}

    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 600; font-size: 14.5px; padding: 10px 18px !important;
        border-radius: 10px 10px 0 0 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: #0F9D7412; color: #0B7D5D !important;
    }

    /* Cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        box-shadow: 0 1px 6px rgba(0,0,0,0.07);
    }
    div[data-testid="stExpander"] {
        border: 1px solid #E6E6E6 !important; border-radius: 12px !important;
        overflow: hidden;
    }
    section[data-testid="stFileUploaderDropzone"] { border-radius: 12px !important; }
    .stButton button { border-radius: 10px !important; font-weight: 600; }
    </style>

    <div class="hero">
      <div class="hero-icon">📚</div>
      <div class="hero-text">
        <h1>StudyBuddy</h1>
        <p>An offline study assistant &mdash; your notes never leave your device.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Settings")
    use_llm = st.checkbox(
        "🧠 Use local AI (Phi-3-mini via ONNX Runtime)",
        help="Powers written answers, AI summaries, and AI flashcards. First "
             "use downloads a ~2.7GB model once, then works fully offline. "
             "Generation can take 10-30+ seconds per request on a regular CPU.",
    )
    st.divider()
    st.caption("Search runs on Snapdragon's NPU automatically when available "
               "(onnxruntime-qnn installed), and falls back to CPU otherwise.")

files = st.file_uploader("Upload your notes (PDF)", type="pdf", accept_multiple_files=True)

if not files:
    st.stop()

key = tuple((f.name, f.size) for f in files)
if st.session_state.get("key") != key:
    chunks = [c for f in files for c in chunk_text(read_pdf(f))]
    if not chunks:
        st.error("No text found. Scanned PDFs need OCR (a good add-on for later).")
        st.stop()

    with st.spinner("Loading smarter search (first run downloads a small model, ~90MB)..."):
        try:
            st.session_state.retriever = EmbeddingRetriever(chunks)
            st.session_state.mode = "neural search"
        except Exception as e:
            st.session_state.retriever = Retriever(chunks)
            st.session_state.mode = "keyword search (fallback)"
            st.session_state.mode_note = str(e)
    st.session_state.chunks = chunks
    st.session_state.key = key

is_fallback = st.session_state.mode.endswith("(fallback)")
badge_class = "badge-warn" if is_fallback else "badge-ok"
badge_icon = "⚠️" if is_fallback else "✅"
st.markdown(
    f'<span class="badge {badge_class}">{badge_icon} {st.session_state.mode} '
    f'· {len(st.session_state.chunks)} chunks loaded</span>',
    unsafe_allow_html=True,
)
if is_fallback:
    st.warning(f"Neural search unavailable, using keyword search instead. ({st.session_state.mode_note})")

tab_ask, tab_summary, tab_cards, tab_bench = st.tabs(
    ["🔍 Ask", "📝 Summary", "🗂️ Flashcards", "🏎️ Benchmark"]
)

with tab_ask:
    question = st.text_input("Ask a question about your notes")
    if question:
        hits = st.session_state.retriever.search(question)
        spinner_msg = (
            "Generating an answer (first time downloads the AI model, ~2.7GB, can take a while)..."
            if use_llm else "Searching..."
        )
        with st.spinner(spinner_msg):
            result = answer(question, [text for text, _ in hits], use_llm=use_llm)
        with st.container(border=True):
            st.markdown("**Answer**")
            st.write(result)
        with st.expander("Sources"):
            for text, score in hits:
                st.progress(min(max(score, 0.0), 1.0), text=f"match score {score:.2f}")
                st.write(text)
                st.divider()

with tab_summary:
    if st.button("Generate summary", type="primary"):
        spinner_msg = "Writing an AI summary..." if use_llm else "Summarizing..."
        with st.spinner(spinner_msg):
            st.session_state.summary = summarize(st.session_state.chunks, use_llm=use_llm)
    if "summary" in st.session_state:
        with st.container(border=True):
            st.markdown("**Summary**")
            st.write(st.session_state.summary)

with tab_cards:
    num_cards = st.slider("Number of flashcards", 3, 10, 5)
    if st.button("Generate flashcards", type="primary"):
        spinner_msg = "Writing AI flashcards..." if use_llm else "Building flashcards..."
        with st.spinner(spinner_msg):
            st.session_state.cards = flashcards(st.session_state.chunks, use_llm=use_llm, num_cards=num_cards)
    dots = ["🟢", "🔵", "🟣", "🟠", "🟡", "🔴", "🟤", "⚪", "⚫", "🟩"]
    for i, card in enumerate(st.session_state.get("cards", []), start=1):
        dot = dots[(i - 1) % len(dots)]
        with st.expander(f"{dot} Card {i}: {card['question']}"):
            st.write(card["answer"])

with tab_bench:
    st.caption("Times the embedding-search step on this machine and shows which "
               "ONNX Runtime execution providers are available (CPU, or NPU/QNN on Snapdragon).")
    if st.button("Run benchmark", type="primary"):
        with st.spinner("Benchmarking..."):
            result = benchmark_embeddings(st.session_state.chunks[:20])
        with st.container(border=True):
            st.write(f"**Providers available:** {', '.join(result['providers'])}")
            col1, col2 = st.columns(2)
            col1.metric("Average time per search", f"{result['avg_seconds']}s")
            col2.metric("Texts embedded", result["num_texts"])
            st.caption(f"Warm-up: {result['warmup_seconds']}s · runs: {result['runs_seconds']}")
