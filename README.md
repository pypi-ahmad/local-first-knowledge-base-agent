# Local-First Knowledge Base Agent

A personal, privacy-first knowledge base agent that indexes your notes, code, PDFs, images, audio, and browser history, then answers questions about them — with citations, temporal reasoning, and a local knowledge graph — using LangGraph and Streamlit.

![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)
![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-1c3c3c)
![Local-first](https://img.shields.io/badge/storage-local--first-green)
![License: MIT](https://img.shields.io/badge/license-MIT-yellow)

**Repository:** https://github.com/pypi-ahmad/local-first-knowledge-base-agent

## Contents

- [Features](#features)
- [Demo / Screenshots](#demo--screenshots)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Installation & setup](#installation--setup)
- [Environment variables](#environment-variables)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Configuration options](#configuration-options)
- [Examples](#examples)
- [Future improvements](#future-improvements)
- [Documentation](#documentation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

**Indexing**
- Notes (Markdown, plain text), code (Python gets AST-aware chunking by function/class; other languages use generic chunking), PDFs (layout-aware: headings, tables, and OCR fallback for scanned pages), images (OCR + captioning via a local Ollama vision model), audio (local transcription via `faster-whisper`), and Chrome/Edge browser history.
- Incremental: only new/changed files are reprocessed, tracked by mtime + content hash in SQLite.
- Deep extraction (optional, on by default): a local LLM pass per chunk pulls entities, relations, decisions, action items, and open questions into a lightweight knowledge graph.

**Retrieval**
- Hybrid search: vector similarity (Chroma) + BM25 keyword search, combined with reciprocal rank fusion.
- Graph-augmented retrieval (GraphRAG-style): entities mentioned in a query are traversed in the knowledge graph to pull in related documents that wouldn't otherwise match lexically or semantically.
- Query expansion (LLM-generated paraphrases) and LLM-based re-ranking of candidates — both always run on a local Ollama model, regardless of which provider is selected for the final answer.
- Temporal filtering: relative dates ("last month", "yesterday", "recently"), explicit months/quarters ("in January", "Q1 2026"), and entity-anchored ranges ("before Project Atlas started").

**Conversational agent**
- A 6-node LangGraph workflow: query understanding → retrieval → re-rank → generation → citation → reflection.
- Cross-session conversation memory via a SQLite-backed LangGraph checkpointer (multiple threads, persists across restarts).
- Inline `[n]` citations resolved to source file, snippet, type, and date.
- A reflection step flags answers that look incomplete or unsupported (informational only — it doesn't loop/retry).

**Multi-provider model routing**
- **Ollama** (local): models are listed dynamically from whatever you've pulled.
- **OpenAI-compatible**: `gpt-5.6-luna` / `gpt-5.6-terra` (medium reasoning effort).
- **Agnes AI**: `agnes-2.5-flash`.
- **Google Gemini**: `gemini-3.5-flash-lite` / `gemini-3.7-flash`.
- Embeddings always run locally via Ollama. A **local-only mode** toggle disables all remote providers at runtime.

**Proactive intelligence**
- On-demand daily/weekly digests summarizing recent activity.
- "You mentioned X N times this month" suggestions from the knowledge graph.
- Forgotten open questions and conflicting-decision detection.
- Per-topic timeline view.

**Privacy & export**
- A dashboard listing every indexed file, with one-click purge and permanent folder/file exclusion.
- Export any answer, or a generated topic "knowledge report", as Markdown or PDF.

## Demo / Screenshots

_Not yet included — run the app locally (see below) to try it against your own files._

## Tech stack

| Layer | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) (+ `langgraph-checkpoint-sqlite` for memory) |
| LLM/embedding clients | `langchain-core`, `langchain-ollama`, `langchain-openai`, `langchain-google-genai` |
| Vector store | [Chroma](https://www.trychroma.com/) (`langchain-chroma`), persisted locally |
| Keyword search | `rank-bm25` |
| Knowledge graph | `networkx` over entities/relations stored in SQLite |
| PDF parsing/export | `pymupdf` (layout, tables, scanned-page OCR, and Markdown→PDF export) |
| Audio transcription | `faster-whisper` |
| Temporal parsing | `dateparser` |
| Metadata / memory store | SQLite (stdlib `sqlite3`) |
| Package management | [uv](https://docs.astral.sh/uv/) |

## Project structure

```
.
├── app.py                  # Streamlit UI (entry point)
├── graph.py                # LangGraph workflow wiring
├── state.py                # Shared LangGraph state schema (TypedDicts)
├── config.py                # Env vars, paths, model catalog, tunables
├── export.py                # Markdown/PDF export (no LLM calls)
├── utils.py                  # Hashing, snippets, date helpers
├── agents/
│   ├── models.py            # Provider/model factory (Ollama/OpenAI-compat/Agnes/Gemini)
│   ├── nodes.py              # LangGraph node implementations
│   ├── extraction.py         # LLM-based entity/relation/decision extraction
│   └── proactive.py          # Digests, suggestions, conflict detection, reports
├── indexer/
│   ├── loaders.py             # File discovery + text/PDF extraction + chunking
│   ├── code_parser.py         # AST-aware Python chunking
│   ├── image_loader.py        # Ollama vision OCR/captioning
│   ├── audio_loader.py        # faster-whisper transcription
│   ├── browser_history.py     # Chrome/Edge history reader
│   └── pipeline.py            # Incremental indexing orchestration
├── retriever/
│   ├── store.py               # Chroma vector store wrapper
│   ├── keyword.py             # BM25 + reciprocal rank fusion (hybrid search)
│   ├── rerank.py               # LLM-based re-ranking
│   ├── temporal.py             # Relative/quarter/entity-anchored date parsing
│   └── graph_rag.py            # Graph-augmented retrieval
├── db/
│   ├── metadata.py             # Indexed-file tracking, excluded paths
│   ├── graph_store.py          # Entities/relations + networkx traversal
│   ├── annotations.py          # Decisions/action items/open questions, pinned answers
│   └── checkpointer.py         # SQLite-backed LangGraph conversation memory
├── .env.example
└── run.cmd                    # Double-click setup + launch (Windows)
```

## Installation & setup

**Requirements:** Windows, [uv](https://docs.astral.sh/uv/) (installed automatically by `run.cmd` if missing), and [Ollama](https://ollama.com) running locally with at least one chat model and one embedding model pulled (e.g. `ollama pull qwen3.5:0.8b && ollama pull embeddinggemma:300m`).

### Option 1 — one-click (recommended)

Double-click **`run.cmd`**. It will:
1. Install `uv` if it isn't already on your PATH.
2. Run `uv sync` to create `.venv` and install all dependencies.
3. Create `.env` from `.env.example` on first run (edit it to add any API keys you use).
4. Launch the app at **http://localhost:8943**.

### Option 2 — manual

```bash
uv sync
copy .env.example .env    # then edit .env with your keys
uv run streamlit run app.py --server.port 8943
```

## Environment variables

All keys are read from environment variables (never hardcoded). If a variable is already set at the OS level, it takes precedence over `.env`.

| Variable | Required for | Default |
|---|---|---|
| `OLLAMA_BASE_URL` | Local models (embeddings, retrieval helpers, and generation when Ollama is selected) | `http://localhost:11434` |
| `OLLAMA_VISION_MODEL` | Image OCR/captioning during indexing | `qwen3-vl:4b` |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | OpenAI-compatible provider (`gpt-5.6-luna`, `gpt-5.6-terra`) | — |
| `AGNES_API_KEY` | Agnes AI provider (`agnes-2.5-flash`, base URL `https://apihub.agnes-ai.com/v1`) | — |
| `GOOGLE_API_KEY` | Google Gemini provider (`gemini-3.5-flash-lite`, `gemini-3.7-flash`) | — |
| `WHISPER_MODEL_SIZE` | Audio transcription model size | `small` |
| `WHISPER_DEVICE` | `cpu` or `cuda` | `cpu` |
| `WHISPER_COMPUTE_TYPE` | faster-whisper compute type | `int8` |
| `LOCAL_ONLY_MODE` | Default state of the local-only toggle (`true`/`false`) | `false` |

See `.env.example` for a ready-to-copy template.

## Usage

1. Launch the app (see above) and open http://localhost:8943.
2. In the sidebar, add one or more folders to index, optionally include browser history, and click **Re-index now**. Progress is shown live.
3. Pick a provider/model and an embedding model.
4. Ask a question in the **Chat** tab — e.g. *"What did we decide about the vector store last month?"* Answers cite sources inline; expand **Sources** to see snippets.
5. Use **Timeline & digest** for daily/weekly digests, repeated-mention suggestions, and conflict checks; **Knowledge report** to generate and export a topic summary; **Privacy & sources** to audit or purge what's indexed.

## How it works

```
User question
     │
     ▼
query_understanding  →  parses relative/explicit dates, detects source-type hints
     ▼
retrieval            →  query expansion (local LLM) + hybrid (vector+BM25) search
     │                  + graph-augmented lookup for entity-connected documents
     ▼
rerank               →  local LLM scores and re-orders candidates
     ▼
generation           →  selected provider/model answers using only retrieved context
     ▼
citation             →  resolves [n] markers to source/snippet/date
     ▼
reflection           →  local LLM flags answers that look incomplete (no retry loop)
```

Indexing runs the same file through: extraction (text/PDF/image/audio) → chunking (AST-aware for Python) → optional LLM entity/relation/decision extraction → embedding → Chroma upsert, with a SQLite record so unchanged files are skipped on the next pass.

## Configuration options

Tunables live in `config.py`:

- `CHUNK_SIZE` / `CHUNK_OVERLAP` — chunking granularity for notes/code/PDFs.
- `RETRIEVAL_TOP_K` / `RERANK_TOP_K` — candidates retrieved vs. kept after re-ranking.
- `GRAPH_HOP_DEPTH` — how many hops the graph-augmented retrieval traverses.
- `QUERY_EXPANSION_COUNT` — number of paraphrases generated per query.
- `HYBRID_VECTOR_WEIGHT` — vector vs. BM25 weighting in the fusion score.
- `PRICING_USD_PER_1M` — reference pricing shown next to the model picker (approximate, not billing-accurate).

## Examples

- "What did we decide about the vector store last month?"
- "Show me open questions about Project Atlas."
- "What have I been reading about lately?" (browser history)
- "Summarize the `Greeter` class." (code, structure-aware)
- "Give me a weekly digest."

## Future improvements

These are deliberate scope decisions, not oversights — noted here as upgrade paths:

- Swap the LLM-based re-ranker for a real cross-encoder if precision needs it.
- Add tree-sitter for structure-aware chunking of non-Python languages.
- Replace on-demand digests with a real scheduled job (Windows Task Scheduler / cron).
- Persist the BM25 index instead of rebuilding it per query, if the corpus grows large.

## Documentation

| Document | Purpose |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Cited technical deep-dive: tech stack, subsystems, data flow, and inferred design decisions |
| [USAGE.md](USAGE.md) | Step-by-step walkthrough of every tab and feature, plus a troubleshooting table |
| [LICENSE](LICENSE) | MIT license terms |

## License

[MIT](LICENSE)

## Acknowledgements

Built on [LangGraph](https://github.com/langchain-ai/langgraph), [Streamlit](https://streamlit.io), [Chroma](https://www.trychroma.com/), and [Ollama](https://ollama.com).

<p align="center">Made with ❤️ by Ahmad Mujtaba</p>
