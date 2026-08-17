# Usage Guide

A step-by-step walkthrough of the Streamlit app, grounded in the actual UI
code in `app.py`. For what the app is and how it's built, see
[ARCHITECTURE.md](ARCHITECTURE.md).

## 1. First-time setup

```powershell
git clone https://github.com/pypi-ahmad/local-first-knowledge-base-agent.git
cd local-first-knowledge-base-agent
run.cmd
```

`run.cmd` installs `uv` if it's missing, runs `uv sync` (creates `.venv` in
the project root), creates `.env` from `.env.example` on first run, and
launches the app at `http://localhost:8943`.

Manual setup is the same three steps run yourself: `uv sync`, copy
`.env.example` to `.env`, then `uv run streamlit run app.py --server.port 8943`.

You need a running [Ollama](https://ollama.com) instance for embeddings and
for retrieval-quality helpers (query expansion, re-ranking, reflection) —
these always run locally regardless of which provider you pick for chat
generation. Pull at least one model and one embedding model, e.g.:

```powershell
ollama pull llama3.1
ollama pull embeddinggemma:300m
```

## 2. Pick a provider and model (sidebar)

Open the app and look at the sidebar, top to bottom:

1. **Local-only mode** toggle — when on, only Ollama models are usable;
   picking any other provider raises an error rather than silently calling
   out to the network.
2. **Provider** — one of Ollama (local), OpenAI-compatible, Agnes AI, or
   Google Gemini.
3. **Model** — populated from whichever provider you picked.
4. **Embedding model** — defaults to `embeddinggemma:300m` if it's already
   pulled in Ollama; otherwise pick any embedding-capable Ollama model.
   Embeddings always run locally, even in non-local-only mode.
5. **Pricing reference** expander — shows per-1M-token cost estimates for
   the currently selected model, sourced from `config.PRICING_USD_PER_1M`.

You must pick both a model and an embedding model before you can chat.

## 3. Add sources and index them

In the sidebar's **Sources** section:

1. Enter a folder path and click **Add** — it's saved to
   `sources.json` in the app's data directory and persists across restarts.
2. Optionally check **Include browser history** to index your Chrome/Edge
   history (disabled if no supported browser is detected).
3. **Deep extraction** (on by default) additionally extracts entities,
   relationships, and decisions/action items/open questions per chunk using
   your local Ollama model — turn it off to index faster if you only need
   plain search, not graph-augmented retrieval or the Timeline/proactive
   features.
4. Click **Re-index now**. A progress bar tracks per-file progress; added,
   updated, skipped, removed, and error counts are shown when it finishes.

Indexing is incremental — re-running it only touches files that changed
(by modification time and size) since the last run, and files deleted from
a watched folder are automatically removed from the index.

Supported inputs: plain text, Markdown, PDF, Word/Office documents, Python
source (parsed per-function/class rather than as flat text), images
(captioned via Ollama vision), and audio (transcribed via `faster-whisper`).

## 4. Ask questions (Chat tab)

1. Use the **conversation** selector at the top of the Chat tab, or click
   **New chat** to start a fresh thread. Each thread's history persists
   across app restarts (backed by a SQLite checkpoint database).
2. Optionally set **Search filters** in the sidebar (file types, folders,
   date range) before asking — they narrow retrieval to matching sources.
3. Type your question in the chat box. The agent expands your query,
   retrieves from hybrid vector+keyword search plus graph-connected
   documents, re-ranks, generates an answer with numbered citations, and
   flags itself if the answer might be incomplete.
4. Under each answer:
   - **Sources** expander shows every cited chunk with its file name, type,
     date, and a snippet.
   - **Pin answer** saves the Q&A pair for later reference (see Chat tab's
     "Pinned answers" section below the chat box).
   - **Save as note** writes the answer to `saved_notes/` and re-indexes it
     immediately, folding it back into your knowledge base.
   - **Export Markdown** / **Export PDF** download the answer with its
     citations.

## 5. Timeline & digest tab

- **Daily digest** / **Weekly digest** — an LLM-written summary of recent
  annotated activity (decisions, action items, open questions) and the most
  frequently mentioned entities. Requires at least one pulled Ollama model.
- **Suggestions** — surfaces entities you've mentioned 3+ times in the last
  30 days.
- **Forgotten open questions** — open questions older than 60 days that are
  still unresolved.
- **Check for conflicting decisions** — compares each entity's two most
  recent recorded decisions and flags contradictions via the local LLM.
- **Timeline for a topic** — a date-sorted table of every annotation tied
  to an entity or topic you type in.

All of these require **Deep extraction** to have been enabled during
indexing — without it, no entities/annotations exist to report on.

## 6. Knowledge report tab

Type a topic and click **Generate report** (needs an embedding model and at
least one pulled Ollama model). It runs a hybrid search over the topic,
then asks the local model to write a structured report (Overview, Key
Decisions, Timeline, Open Questions, Sources) from only the retrieved
context. Download the result as Markdown or PDF.

## 7. Privacy & sources tab

- **Indexed files** — a table of every indexed source with its type, chunk
  count, and last-indexed time. Select one and click **Purge selected
  source** to remove it (and its graph relations/annotations) from the
  index entirely.
- **Excluded paths** — permanently exclude a path from future indexing runs
  (or un-exclude it later), independent of whether it's currently indexed.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Chat input says to pick a model first | No provider model or embedding model selected in the sidebar | Select both dropdowns before typing a question |
| "Local-only mode" blocks a provider | `local_only` toggle is on and you picked a non-Ollama provider | Turn the toggle off, or switch to Ollama |
| Embedding model dropdown is empty | No embedding-capable model pulled in Ollama | `ollama pull embeddinggemma:300m` (or any embedding model), then refresh |
| "Include browser history" checkbox is disabled | No supported browser (Chrome/Edge) history file found on this machine | Only relevant on Windows with a supported browser installed; safe to ignore otherwise |
| Digest/report/conflict-check buttons are disabled | No Ollama model is pulled at all | Pull any chat-capable Ollama model |
| Timeline tab shows nothing for a topic | Deep extraction was off during indexing | Re-index the relevant folder with **Deep extraction** enabled |
| Re-index reports files under "errors" | A specific file failed to load/parse (corrupt PDF, unsupported encoding, etc.) | Indexing continues past the bad file by design; check the file itself |
| Pricing reference shows models you never selected (e.g. `sonnet-5`, `grok-4.6`) | Known stale entries in `config.py`'s pricing table with no matching selectable provider/model | Cosmetic only — does not affect cost estimates for models you can actually pick; see [ARCHITECTURE.md](ARCHITECTURE.md#eol--dead-dependency-scan) |

## Uninstalling / resetting

All state lives under the app's data directory (Chroma vector store,
`metadata.sqlite3`, `checkpoints.sqlite3`, `sources.json`, `saved_notes/`).
Deleting that directory resets the app to a clean state; deleting `.venv`
and re-running `run.cmd` resets the Python environment.
