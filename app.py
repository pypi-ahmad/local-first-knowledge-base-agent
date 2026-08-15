"""Streamlit UI for the local-first knowledge-base agent."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

import config
from agents import models, proactive
from agents.nodes import format_context
from db import annotations as annotations_db
from db import metadata
from export import build_export_markdown, markdown_to_pdf_bytes
from graph import get_graph
from indexer.pipeline import clear_index, index_browser_history, index_folder, purge_source
from indexer.browser_history import available_browsers
from retriever.keyword import hybrid_search
from retriever.store import build_where_filter, get_vectorstore

st.set_page_config(page_title="Local-First Knowledge Base Agent", page_icon=":material/psychology:", layout="wide")

SOURCES_FILE = config.DATA_DIR / "sources.json"


@st.cache_data(show_spinner=False)
def _cached_pdf(markdown_text: str) -> bytes:
    return markdown_to_pdf_bytes(markdown_text)


def _load_sources() -> list[str]:
    if SOURCES_FILE.is_file():
        return json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
    return []


def _save_sources(sources: list[str]) -> None:
    SOURCES_FILE.write_text(json.dumps(sources), encoding="utf-8")


def _init_state() -> None:
    st.session_state.setdefault("sources", _load_sources())
    st.session_state.setdefault("thread_id", str(uuid.uuid4()))
    st.session_state.setdefault("threads", [st.session_state["thread_id"]])
    st.session_state.setdefault("local_only", config.LOCAL_ONLY_MODE_DEFAULT)


_init_state()


# --- Sidebar: model selection -------------------------------------------------

with st.sidebar:
    st.header("Model")

    local_only = st.toggle(
        "Local-only mode", value=st.session_state["local_only"],
        help="When on, only Ollama models are used; remote API calls are refused.",
    )
    st.session_state["local_only"] = local_only

    provider_labels = {
        "ollama": "Ollama (local)",
        "openai_compatible": "OpenAI-compatible",
        "agnes": "Agnes AI",
        "gemini": "Google Gemini",
    }
    available_providers = ["ollama"] if local_only else config.PROVIDERS
    provider = st.selectbox(
        "Provider", available_providers, format_func=lambda p: provider_labels.get(p, p),
    )

    model_options = models.models_for_provider(provider)
    if not model_options:
        st.warning("No models available for this provider (check it's running / configured).")
        model = None
    else:
        model = st.selectbox("Model", model_options)

    ollama_models = models.list_ollama_models()
    embedding_model = st.selectbox(
        "Embedding model (Ollama, local)", ollama_models,
        index=ollama_models.index("embeddinggemma:300m") if "embeddinggemma:300m" in ollama_models else 0,
        help="Embeddings always run locally, regardless of the provider chosen above.",
    ) if ollama_models else None

    if model and model in config.PRICING_USD_PER_1M:
        rates = config.PRICING_USD_PER_1M[model]
        st.caption(f"Est. price: ${rates['input']:.2f} in / ${rates['output']:.2f} out per 1M tokens")

    with st.expander("Pricing reference"):
        st.dataframe(
            [{"model": m, **r} for m, r in config.PRICING_USD_PER_1M.items()],
            hide_index=True, width="stretch",
        )

    st.divider()
    st.header("Sources")

    new_folder = st.text_input("Folder to index", placeholder=r"C:\Users\you\Documents\Notes")
    if st.button("Add folder", width="stretch") and new_folder:
        if new_folder not in st.session_state["sources"]:
            st.session_state["sources"].append(new_folder)
            _save_sources(st.session_state["sources"])
        st.rerun()

    for folder in list(st.session_state["sources"]):
        with st.container(horizontal=True):
            st.text(folder)
            if st.button("Remove", key=f"remove_{folder}"):
                st.session_state["sources"].remove(folder)
                _save_sources(st.session_state["sources"])
                st.rerun()

    include_history = st.checkbox("Include browser history", value=False, disabled=not available_browsers())
    if not available_browsers():
        st.caption("No Chrome/Edge history file found on this machine.")
    deep_extraction = st.checkbox(
        "Deep extraction (entities/decisions)", value=True,
        help="Runs an extra local-LLM pass per chunk to build the knowledge graph and detect decisions/action items. Slower.",
    )

    reindex_col, clear_col = st.columns(2)
    if reindex_col.button("Re-index now", width="stretch", disabled=not embedding_model):
        embeddings = models.build_embeddings(embedding_model)
        extraction_llm = models.build_chat_model("ollama", model if provider == "ollama" else (ollama_models[0] if ollama_models else None)) if deep_extraction and ollama_models else None
        progress_bar = st.progress(0.0, text="Indexing...")

        def _progress(update: dict) -> None:
            total = max(update["total"], 1)
            progress_bar.progress(update["current"] / total, text=f"{update['current']}/{update['total']}: {Path(update['file']).name}")

        total_stats = {"added": 0, "updated": 0, "skipped": 0, "removed": 0, "errors": []}
        for folder in st.session_state["sources"]:
            folder_path = Path(folder)
            if not folder_path.is_dir():
                total_stats["errors"].append(f"{folder}: not a directory")
                continue
            stats = index_folder(folder_path, embeddings, llm=extraction_llm, deep_extraction=deep_extraction, progress_cb=_progress)
            for key in ("added", "updated", "skipped", "removed"):
                total_stats[key] += stats[key]
            total_stats["errors"].extend(stats["errors"])
        if include_history:
            history_stats = index_browser_history(embeddings, progress_cb=_progress)
            total_stats["added"] += history_stats["added"]
            total_stats["skipped"] += history_stats["skipped"]

        progress_bar.empty()
        st.success(f"Indexed: {total_stats['added']} added, {total_stats['updated']} updated, "
                   f"{total_stats['skipped']} unchanged, {total_stats['removed']} removed.")
        if total_stats["errors"]:
            st.error("\n".join(total_stats["errors"][:5]))

    if clear_col.button("Clear index", width="stretch"):
        if embedding_model:
            clear_index(models.build_embeddings(embedding_model))
            st.success("Index cleared.")

    stats = metadata.get_stats()
    st.metric("Indexed files", stats["total_files"])
    st.metric("Chunks", stats["total_chunks"])
    if stats["by_type"]:
        st.caption(", ".join(f"{k}: {v}" for k, v in stats["by_type"].items()))

    st.divider()
    st.header("Search filters")
    file_types = st.multiselect("File type", ["note", "code", "pdf", "image", "audio", "history"])
    date_col1, date_col2 = st.columns(2)
    date_start = date_col1.date_input("From", value=None)
    date_end = date_col2.date_input("To", value=None)


filters = {}
if file_types:
    filters["file_types"] = file_types
if date_start:
    filters["date_start"] = date_start.isoformat()
if date_end:
    filters["date_end"] = date_end.isoformat()


# --- Main area -----------------------------------------------------------------

chat_tab, timeline_tab, report_tab, privacy_tab = st.tabs(
    ["Chat", "Timeline & digest", "Knowledge report", "Privacy & sources"]
)

with chat_tab:
    thread_col, new_col = st.columns([4, 1])
    thread_col.selectbox("Conversation", st.session_state["threads"], key="thread_id", label_visibility="collapsed")
    if new_col.button("New chat", width="stretch"):
        new_id = str(uuid.uuid4())
        st.session_state["threads"].append(new_id)
        st.session_state["thread_id"] = new_id
        st.rerun()

    graph = get_graph()
    graph_config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
    snapshot = graph.get_state(graph_config)
    history = snapshot.values.get("messages", []) if snapshot.values else []

    for msg in history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    user_input = st.chat_input("Ask about your notes, code, PDFs, or browsing history...")
    if user_input:
        if not model or not embedding_model:
            st.error("Pick a model and an embedding model in the sidebar first.")
        else:
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = graph.invoke(
                        {
                            "messages": [HumanMessage(content=user_input)],
                            "provider": provider,
                            "model": model,
                            "embedding_model": embedding_model,
                            "local_only": local_only,
                            "filters": filters,
                        },
                        config=graph_config,
                    )
                st.markdown(result["answer"])
                if result.get("needs_retry"):
                    st.caption(f":material/warning: Possibly incomplete: {result.get('reflection_notes', '')}")

                if result["citations"]:
                    with st.expander(f"Sources ({len(result['citations'])})"):
                        for c in result["citations"]:
                            st.markdown(f"**{Path(c['source']).name}** · {c['source_type']} · {c.get('date') or 'undated'}")
                            st.caption(c["snippet"])

                action_cols = st.columns(4)
                if action_cols[0].button("Pin answer", key=f"pin_{len(history)}"):
                    annotations_db.pin_answer(user_input, result["answer"], result["citations"])
                    st.toast("Pinned.")
                if action_cols[1].button("Save as note", key=f"save_{len(history)}"):
                    notes_dir = config.DATA_DIR / "saved_notes"
                    notes_dir.mkdir(exist_ok=True)
                    note_path = notes_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                    note_path.write_text(f"# {user_input}\n\n{result['answer']}\n", encoding="utf-8")
                    if embedding_model:
                        index_folder(notes_dir, models.build_embeddings(embedding_model), deep_extraction=False)
                    st.toast("Saved to knowledge base.")
                export_md = build_export_markdown("Q&A export", f"**Q:** {user_input}\n\n{result['answer']}", result["citations"])
                action_cols[2].download_button("Export Markdown", export_md, file_name="answer.md", key=f"export_md_{len(history)}")
                action_cols[3].download_button(
                    "Export PDF", _cached_pdf(export_md), file_name="answer.pdf", mime="application/pdf",
                    key=f"export_pdf_{len(history)}",
                )

    if st.session_state.get("thread_id"):
        pinned = annotations_db.list_pinned()
        if pinned:
            with st.expander(f"Pinned answers ({len(pinned)})"):
                for p in pinned:
                    st.markdown(f"**{p['question']}**")
                    st.caption(p["answer"][:300])
                    if st.button("Unpin", key=f"unpin_{p['id']}"):
                        annotations_db.unpin(p["id"])
                        st.rerun()


with timeline_tab:
    st.subheader("Digest")
    digest_cols = st.columns(2)
    local_model_for_digest = ollama_models[0] if ollama_models else None
    if digest_cols[0].button("Daily digest", disabled=not local_model_for_digest):
        llm = models.build_chat_model("ollama", local_model_for_digest)
        st.info(proactive.generate_digest(llm, "daily"))
    if digest_cols[1].button("Weekly digest", disabled=not local_model_for_digest):
        llm = models.build_chat_model("ollama", local_model_for_digest)
        st.info(proactive.generate_digest(llm, "weekly"))

    st.subheader("Suggestions")
    for suggestion in proactive.repeated_mention_suggestions():
        st.write(f":material/lightbulb: {suggestion}")

    forgotten = proactive.forgotten_open_questions()
    if forgotten:
        st.subheader("Forgotten open questions")
        for q in forgotten[:5]:
            st.write(f"- {q['text']} (from {Path(q['source']).name}, {q['date']})")

    if local_model_for_digest and st.button("Check for conflicting decisions"):
        conflicts = proactive.detect_conflicts(models.build_chat_model("ollama", local_model_for_digest))
        if not conflicts:
            st.success("No conflicting decisions found.")
        for c in conflicts:
            st.warning(f"**{c['entity']}**: \"{c['older']['text']}\" vs \"{c['newer']['text']}\" — {c['reason']}")

    st.divider()
    st.subheader("Timeline for a topic")
    topic = st.text_input("Entity or topic", key="timeline_topic")
    if topic:
        items = annotations_db.list_annotations(entity=topic)
        if items:
            st.dataframe(
                [{"date": i["date"], "kind": i["kind"], "text": i["text"], "source": Path(i["source"]).name} for i in items],
                hide_index=True, width="stretch",
            )
        else:
            st.caption("No annotated decisions/action items/open questions for this entity yet.")


with report_tab:
    st.subheader("Knowledge report")
    report_topic = st.text_input("Topic", key="report_topic")
    if st.button("Generate report", disabled=not (report_topic and embedding_model and ollama_models)):
        embeddings = models.build_embeddings(embedding_model)
        vs = get_vectorstore(embeddings)
        where = build_where_filter(filters)
        docs = hybrid_search(vs, report_topic, k=config.RETRIEVAL_TOP_K, where=where)
        context = format_context(docs)
        llm = models.build_chat_model("ollama", ollama_models[0])
        report_text = proactive.generate_knowledge_report(llm, report_topic, context)
        st.session_state["last_report"] = build_export_markdown(f"Knowledge report: {report_topic}", report_text)

    if st.session_state.get("last_report"):
        st.markdown(st.session_state["last_report"])
        dl_cols = st.columns(2)
        dl_cols[0].download_button("Download Markdown", st.session_state["last_report"], file_name="report.md")
        dl_cols[1].download_button(
            "Download PDF", _cached_pdf(st.session_state["last_report"]),
            file_name="report.pdf", mime="application/pdf",
        )


with privacy_tab:
    st.subheader("Indexed files")
    files = metadata.list_files()
    if files:
        st.dataframe(
            [{"path": f.path, "type": f.source_type, "chunks": f.chunk_count, "indexed_at": f.indexed_at} for f in files],
            hide_index=True, width="stretch",
        )
        purge_target = st.selectbox("Purge a specific source", [f.path for f in files])
        if st.button("Purge selected source") and embedding_model:
            purge_source(models.build_embeddings(embedding_model), purge_target)
            st.success(f"Removed {purge_target} from the index.")
            st.rerun()
    else:
        st.caption("Nothing indexed yet.")

    st.subheader("Excluded paths")
    exclude_input = st.text_input("Path to exclude permanently", key="exclude_input")
    if st.button("Exclude") and exclude_input:
        metadata.exclude_path(exclude_input)
        st.rerun()
    for p in metadata.list_excluded_paths():
        with st.container(horizontal=True):
            st.text(p)
            if st.button("Un-exclude", key=f"unexclude_{p}"):
                metadata.unexclude_path(p)
                st.rerun()
