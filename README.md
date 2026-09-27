# StudyBuddy: offline study assistant

Upload PDF notes and get search, AI-written answers, summaries, and
flashcards — all running locally. Built for the Snapdragon® AI Lab Build &
Present Challenge: search runs on Snapdragon's NPU automatically when
available, and the AI features use Phi-3-mini, an open-source model, via
ONNX Runtime.

## Setup (Windows)
1. Install Python 3.11+ (on a Snapdragon laptop, get the **ARM64** build).
2. In this folder:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   python check_env.py
   streamlit run app.py
   ```
3. Upload a PDF. Try the **Ask**, **Summary**, and **Flashcards** tabs.

## Features
- **Ask** — semantic search over your notes (ONNX Runtime embeddings), with
  raw matching passages by default or an AI-written answer when "Use local
  AI" is checked in the sidebar.
- **Summary** — an instant extractive summary (TF-IDF, no download), or an
  AI-written summary with "Use local AI" checked.
- **Flashcards** — instant cloze-style cards (blank-the-key-term, no
  download), or real AI-written Q&A flashcards with "Use local AI" checked.
- **Benchmark** — times the embedding step and shows which ONNX Runtime
  execution providers are available on this machine (CPU, or NPU/QNN on
  Snapdragon).

## The "Use local AI" checkbox
Turning it on downloads **Phi-3-mini** (~2.7GB) from Hugging Face the first
time you use Ask, Summary, or Flashcards with it checked — do this on good
wifi, it can take several minutes. After that it runs fully offline.
Generation takes roughly 10-30 seconds per request on a regular CPU. If the
model can't load for any reason, every feature falls back automatically to
its no-AI version instead of crashing.

## Snapdragon notes
- Swap `onnxruntime` for `onnxruntime-qnn` in requirements.txt. `check_env.py`
  should then show `QNNExecutionProvider`, and the Benchmark tab will show it
  in use.
- No Snapdragon device yet? Everything here runs on CPU. Use Qualcomm AI
  Hub's hosted devices to profile models later.

## Project layout
```
app.py                  Streamlit UI (all 4 tabs)
check_env.py             Prints Python/ONNX Runtime info + available providers
requirements.txt
studybuddy/
  ingest.py               PDF -> text chunks
  retrieve.py             TF-IDF search + neural (ONNX) semantic search
  embed.py                Sentence embeddings via ONNX Runtime
  llm.py                  Phi-3-mini text generation via ONNX Runtime GenAI
  summarize.py            Extractive + AI summaries
  flashcards.py           Heuristic + AI flashcards
  benchmark.py            Timing benchmark for the embedding step
```

## Remaining: submission day
- Write up the README/approach, record a short demo video, and submit before
  30 Sep 2026, 11:59 PM IST.
