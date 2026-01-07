from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import os
import pandas as pd
import sys
import streamlit as st
from dotenv import load_dotenv

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(PKG_DIR, ".."))

# Streamlit Cloud import fix: parent'ı sys.path'e ekle
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from podkg_hybrid.shared.types import GraphSettings, VectorSettings
from podkg_hybrid.vector.engine import (
    BM25_DROP_RATIO_SEARCH_DEFAULT,
    DENSE_NPROBE_DEFAULT,
    ENABLE_BM25,
    EMBED_MODEL,
    GROUP_SIZE_FIXED,
    HYBRID_RANKER_DEFAULT,
    HYBRID_WEIGHT_BM25_DEFAULT,
    HYBRID_WEIGHT_DENSE_DEFAULT,
    JUDGE_MODEL,
    CHAT_MODEL,
    get_collection_field_names,
    judge_answer,
    linkify_citations,
    openrouter_chat,
    rag_answer,
    vector_retrieve,
)
from podkg_hybrid.vector.render import render_vector_debug_and_results
from podkg_hybrid.graph.engine import (
    NEO4J_DATABASE,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    build_openrouter_client,
    get_driver,
    graph_close_driver,
    graph_execute,
    graph_pick_entities,
    graph_prepare,
    graph_safe_text2cypher,
    llm_answer,
    normalize_str,
    pack_micro_evidence_for_answer,
    set_debug_print_to_console,
    try_fetch_schema,
)
from podkg_hybrid.graph.render import render_graph_results

load_dotenv()

st.set_page_config(page_title="PodKG Hybrid Retrieval", layout="wide")
st.title("PodKG Hybrid Retrieval (Vector + Graph)")

MODE_VECTOR_ONLY = "Only VectorDB Retrieval"
MODE_GRAPH_ONLY = "Only GraphDB Retrieval"
MODE_HYBRID = "Hybrid (Vector + Graph) Retrieval"


# ----------------------------
# URL / citation utilities
# ----------------------------
def _safe_http_url(u: Any) -> str:
    s = (u or "").strip()
    if not s:
        return ""
    if s.startswith("http://") or s.startswith("https://"):
        return s
    return ""


def _parse_video_id_from_doc_id(doc_id: Any) -> str:
    """
    Common pattern in your data: doc_id like 'yt_f9f9271a06' => video_id = 'f9f9271a06'
    """
    s = (doc_id or "").strip()
    if s.startswith("yt_") and len(s) >= 14:
        vid = s[3:14]
        if len(vid) == 11:
            return vid
    return ""


def _get_first_float(d: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in d and d[k] is not None and d[k] != "":
            try:
                return float(d[k])
            except Exception:
                pass
    return None


def _build_youtube_ts_url(video_id: str, start_s: Optional[float]) -> str:
    if not video_id:
        return ""
    t = 0
    if start_s is not None:
        try:
            t = max(0, int(round(float(start_s))))
        except Exception:
            t = 0
    return f"https://www.youtube.com/watch?v={video_id}&t={t}s"


def _ensure_ts_url_from_source(src: Dict[str, Any]) -> str:
    """
    Tries hard to produce a clickable URL for a source/evidence item.
    Priority:
      1) ts_url
      2) url
      3) build youtube URL from video_id/doc_id + start
    """
    u = _safe_http_url(src.get("ts_url")) or _safe_http_url(src.get("url"))
    if u:
        return u

    video_id = (src.get("video_id") or "").strip()
    if not video_id:
        video_id = _parse_video_id_from_doc_id(src.get("doc_id"))

    start_s = _get_first_float(src, ["start", "start_s", "ts_start", "t_start", "begin", "offset_s"])
    return _safe_http_url(_build_youtube_ts_url(video_id, start_s))


def build_hybrid_graph_evidence(
    graph_result_action: str,
    graph_df,
    max_items: int,
    max_micros_each: int,
) -> List[Dict[str, Any]]:
    items = pack_micro_evidence_for_answer(
        graph_df,
        graph_result_action,
        max_items=max_items,
        max_micros_each=max_micros_each,
    )
    evidence: List[Dict[str, Any]] = []
    gid = 1

    for item in items:
        itype = item.get("type")

        if itype == "fact":
            ctx = "fact: {sub} {pred} {obj}".format(
                sub=item.get("subject") or "",
                pred=item.get("predicate") or "",
                obj=item.get("object") or "",
            ).strip()

            for ev in item.get("evidence", []):
                row = {
                    "id": f"G{gid}",
                    "context": ctx,
                    "micro_id": ev.get("micro_id"),
                    "start": ev.get("start"),
                    "end": ev.get("end"),
                    "speaker": ev.get("speaker"),
                    "snippet": ev.get("snippet"),
                    # keep both url + ts_url possibility
                    "url": ev.get("url"),
                    "ts_url": ev.get("ts_url"),
                    "video_id": ev.get("video_id"),
                    "doc_id": ev.get("doc_id"),
                    "doc_title": ev.get("doc_title"),
                    "doc_year": ev.get("doc_year"),
                }
                # Guarantee: try to build a clickable url if missing
                row["url"] = _ensure_ts_url_from_source(row) or _safe_http_url(row.get("url"))
                evidence.append(row)
                gid += 1

        elif itype == "micro":
            row = {
                "id": f"G{gid}",
                "context": "micro evidence",
                "micro_id": item.get("micro_id"),
                "start": item.get("start"),
                "end": item.get("end"),
                "speaker": item.get("speaker"),
                "snippet": item.get("snippet"),
                "url": item.get("url"),
                "ts_url": item.get("ts_url"),
                "video_id": item.get("video_id"),
                "doc_id": item.get("doc_id"),
                "doc_title": item.get("doc_title"),
                "doc_year": item.get("doc_year"),
            }
            row["url"] = _ensure_ts_url_from_source(row) or _safe_http_url(row.get("url"))
            evidence.append(row)
            gid += 1

        elif itype == "keyword_mentions":
            ctx = f"keyword mention: {item.get('entity_name') or ''}"
            for ev in item.get("evidence", []):
                row = {
                    "id": f"G{gid}",
                    "context": ctx,
                    "micro_id": ev.get("micro_id"),
                    "start": ev.get("start"),
                    "end": ev.get("end"),
                    "speaker": ev.get("speaker"),
                    "snippet": ev.get("snippet"),
                    "url": ev.get("url"),
                    "ts_url": ev.get("ts_url"),
                    "video_id": ev.get("video_id"),
                    "doc_id": ev.get("doc_id"),
                    "doc_title": ev.get("doc_title"),
                    "doc_year": ev.get("doc_year"),
                }
                row["url"] = _ensure_ts_url_from_source(row) or _safe_http_url(row.get("url"))
                evidence.append(row)
                gid += 1

    return evidence


def build_hybrid_prompt(
    question: str,
    vector_sources: List[Dict[str, Any]],
    graph_evidence: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    system = (
        "You are an evidence-first assistant.\n"
        "\n"
        "CRITICAL RULES:\n"
        "1) Use ONLY the provided evidence. Do NOT use outside knowledge.\n"
        "2) Every factual claim must include citations.\n"
        "3) Cite Vector evidence ONLY as [V1], [V2], ... and Graph evidence ONLY as [G1], [G2], ...\n"
        "   - Use exact bracket form like [V3] or [G2].\n"
        "   - If multiple citations: write like [V1][G2] (no commas inside one bracket).\n"
        "4) If evidence is missing/insufficient, say so explicitly.\n"
        "5) Do NOT output raw URLs; only output citations.\n"
        "\n"
        "OUTPUT FORMAT (Markdown):\n"
        "### Answer\n"
        "<1-6 sentences, concise, fully cited>\n"
        "\n"
        "### Evidence\n"
        "- Bullet points quoting/pointing to the specific snippets you relied on (each bullet cited)\n"
        "\n"
        "### Notes\n"
        "- Optional: contradictions, uncertainty, what is missing (cited if referencing evidence)\n"
    )

    payload = {
        "question": question,
        "vector_sources": vector_sources,
        "graph_evidence": graph_evidence,
    }
    user = json.dumps(payload, ensure_ascii=False, indent=2)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def linkify_hybrid_citations(
    answer_md: str,
    vector_sources: List[Dict[str, Any]],
    graph_evidence: List[Dict[str, Any]],
) -> str:
    if not answer_md:
        return answer_md or ""

    import re

    id2url: Dict[str, str] = {}

    # Vector: prefer ts_url; else try to build one (guarantee best-effort)
    for s in vector_sources:
        sid = (s.get("id") or "").strip()
        if not sid:
            continue
        u = _ensure_ts_url_from_source(s)
        if u:
            id2url[sid] = u

    # Graph: prefer url/ts_url; else build
    for g in graph_evidence:
        gid = (g.get("id") or "").strip()
        if not gid:
            continue
        u = _ensure_ts_url_from_source(g)
        if u:
            id2url[gid] = u

    def _link_one(token: str) -> str:
        url = id2url.get(token, "")
        if url:
            # Wrap URL in <...> to avoid markdown edge cases (parentheses, etc.)
            return f"[{token}](<{url}>)"
        return f"[{token}]"

    # 1) Replace bracket groups like [V1, V2, G3] -> [V1](...)[V2](...)[G3](...)
    def repl_multi(m: re.Match) -> str:
        inner = m.group(1) or ""
        toks = re.findall(r"(V\d+|G\d+)", inner)
        if not toks:
            return m.group(0)
        return "".join(_link_one(t) for t in toks)

    answer_md = re.sub(r"\[((?:\s*(?:V\d+|G\d+)\s*(?:,\s*)?)+)\]", repl_multi, answer_md)

    # 2) Replace remaining singles [V12] / [G4]
    def repl_single(m: re.Match) -> str:
        return _link_one(m.group(1))

    answer_md = re.sub(r"\[(V\d+|G\d+)\]", repl_single, answer_md)
    return answer_md


def run_hybrid_answer(
    question: str,
    vector_sources: List[Dict[str, Any]],
    graph_evidence: List[Dict[str, Any]],
) -> Optional[str]:
    if not OPENROUTER_API_KEY.strip() or not OPENROUTER_BASE_URL.strip():
        return None
    messages = build_hybrid_prompt(question, vector_sources, graph_evidence)
    out = openrouter_chat(messages, model=CHAT_MODEL, temperature=0.2)
    return (out or "").strip() or None


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Retrieval mode", [MODE_VECTOR_ONLY, MODE_GRAPH_ONLY, MODE_HYBRID], index=2)

    st.divider()
    st.header("Vector Settings")

    st.subheader("Grouping Search")
    grouping_enabled = st.checkbox(
        "Enable Grouping Search (server-side)",
        value=True,
        help=(
            "Server-side grouping by a scalar field. When enabled:\n"
            "- Retrieval limit = number of groups\n"
            "- group_size fixed to 4\n"
            "This helps avoid many chunks from the same doc dominating."
        ),
    )

    schema_fields_now = sorted(list(get_collection_field_names()))
    safe_group_fields = [f for f in ["doc_id", "video_id", "show", "primary_speaker", "year"] if f in schema_fields_now]
    if not safe_group_fields:
        safe_group_fields = ["doc_id"]

    if grouping_enabled:
        group_by_field = st.selectbox("group_by_field", safe_group_fields, index=0)
        total_chunks_budget = st.slider(
            "Total chunks budget (group_size=4)",
            4,
            100,
            16,
            4,
            help="Total chunks budget. groups = budget/4.",
        )
        num_groups = max(1, int(total_chunks_budget) // GROUP_SIZE_FIXED)
        strict_group_size = st.checkbox(
            "strict_group_size (try exact 4 per group)",
            value=False,
            help="True: tries to return exactly 4 per group. False: prioritizes group count.",
        )
        retrieval_limit = int(num_groups)
        approx_total_chunks = retrieval_limit * GROUP_SIZE_FIXED
        st.caption(
            f"Grouping ON -> budget={int(total_chunks_budget)} chunks -> groups={retrieval_limit} "
            f"-> group_size={GROUP_SIZE_FIXED} (approx {approx_total_chunks} chunks)"
        )
    else:
        group_by_field = None
        strict_group_size = False
        top_k_chunks = st.slider("Top-K (chunks)", 3, 50, 12, 1)
        retrieval_limit = int(top_k_chunks)
        approx_total_chunks = retrieval_limit
        st.caption(f"Grouping OFF -> top_k(chunks)={retrieval_limit}")

    st.subheader("Vector Retrieval")
    hybrid_enabled_ui_default = True if ENABLE_BM25 else False
    vector_hybrid_enabled = st.checkbox(
        "Hybrid Search (Dense + BM25)",
        value=hybrid_enabled_ui_default,
        help="If enabled: Dense (embedding) + BM25 (text sparse) together. ENABLE_BM25=false will fallback to dense.",
    )

    if vector_hybrid_enabled:
        ranker_kind = st.selectbox(
            "Hybrid ranker",
            ["rrf", "weighted"],
            index=0 if HYBRID_RANKER_DEFAULT != "weighted" else 1,
        )

        default_dense_cand = min(200, max(50, int(approx_total_chunks * 5)))
        default_bm25_cand = min(200, max(50, int(approx_total_chunks * 5)))

        dense_limit = st.slider("Hybrid dense candidate limit", 10, 200, int(default_dense_cand), 5)
        bm25_limit = st.slider("Hybrid BM25 candidate limit", 10, 200, int(default_bm25_cand), 5)

        nprobe = st.slider("Dense nprobe", 1, 64, int(DENSE_NPROBE_DEFAULT), 1)
        bm25_drop_ratio_search = st.slider(
            "BM25 drop_ratio_search", 0.0, 0.95, float(BM25_DROP_RATIO_SEARCH_DEFAULT), 0.05
        )

        if ranker_kind == "weighted":
            w_dense = st.slider("Weight (dense)", 0.0, 1.0, float(HYBRID_WEIGHT_DENSE_DEFAULT), 0.05)
            w_bm25 = st.slider("Weight (BM25)", 0.0, 1.0, float(HYBRID_WEIGHT_BM25_DEFAULT), 0.05)
        else:
            w_dense = float(HYBRID_WEIGHT_DENSE_DEFAULT)
            w_bm25 = float(HYBRID_WEIGHT_BM25_DEFAULT)
    else:
        ranker_kind = HYBRID_RANKER_DEFAULT
        dense_limit = 50
        bm25_limit = 50
        nprobe = int(DENSE_NPROBE_DEFAULT)
        bm25_drop_ratio_search = float(BM25_DROP_RATIO_SEARCH_DEFAULT)
        w_dense = float(HYBRID_WEIGHT_DENSE_DEFAULT)
        w_bm25 = float(HYBRID_WEIGHT_BM25_DEFAULT)

    st.subheader("Vector Filters")
    video_id = st.text_input("video_id (11 chars) filter", value="")
    doc_id = st.text_input("doc_id filter (e.g. yt_f9f9271a06)", value="")
    year = st.number_input("year filter", min_value=0, max_value=2100, value=0, step=1)
    show = st.text_input("show filter (exact match)", value="")

    st.subheader("Vector Debug")
    show_sanity = st.checkbox("Show collection sanity info", value=True)
    show_raw_hit = st.checkbox("Show raw hit debug (very verbose)", value=False)

    vector_mode = st.radio("Vector Mode", ["RAG Answer", "Retrieval Only", "RAG + Judge"], index=0)

    st.divider()
    st.header("Graph Settings")
    planner_enabled = st.toggle("Use LLM planner (JSON)", value=True)
    llm_row_selector = st.toggle("LLM select most relevant rows (Facts/Micros)", value=True)
    generate_answer = st.toggle("Generate narrative answer from evidence", value=True)
    enable_safe_text2cypher = st.toggle(
        "Enable SAFE Text2Cypher fallback (LLM->Cypher->readonly->execute)",
        value=True,
    )

    st.caption("Row selector affects: entity_facts + (keyword_micros, speaker_micros). No predicate regex filtering is used.")
    st.caption("Entity anchors are always LLM-selected.")

    st.divider()
    st.subheader("Macro context (prev/next micros)")
    show_macro_context = st.toggle("Show macro context around a micro", value=True)
    macro_window = st.number_input("Context window (each side)", min_value=0, max_value=10, value=1, step=1)
    macro_context_snippet_len = st.number_input("Context snippet length", min_value=80, max_value=900, value=360, step=20)

    st.divider()
    st.subheader("Selector Limits")
    selector_keep_facts = st.number_input("Max Facts to keep (fallback)", min_value=5, max_value=250, value=40, step=5)
    selector_keep_micros = st.number_input("Max Micros to keep (fallback)", min_value=5, max_value=250, value=35, step=5)

    st.divider()
    st.subheader("Schema")
    use_manual_schema = st.toggle("Paste schema manually", value=False)
    if "graph_schema_text" not in st.session_state:
        st.session_state["graph_schema_text"] = ""
    schema_text = st.text_area("Schema (auto or manual)", value=st.session_state["graph_schema_text"], height=200)
    if st.button("Fetch schema from Neo4j"):
        if not (OPENROUTER_API_KEY and OPENROUTER_BASE_URL):
            st.warning("OPENROUTER env vars missing; schema fetch still attempts Neo4j.")
        try:
            d = get_driver(os.getenv("NEO4J_URI", ""), os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", ""))
            fetched = try_fetch_schema(d, os.getenv("NEO4J_DATABASE", "") or None)
            st.session_state["graph_schema_text"] = fetched
            st.success("Schema fetched.")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    st.divider()
    st.subheader("Limits")
    top_candidates = st.number_input("Top candidates (entity/speaker)", min_value=1, max_value=25, value=10, step=1)
    limit_rows = st.number_input("Rows limit", min_value=10, max_value=1200, value=250, step=10)
    evidence_k = st.number_input("Evidence micros per item", min_value=1, max_value=10, value=3, step=1)
    snippet_len = st.number_input("Snippet length", min_value=80, max_value=900, value=240, step=20)
    max_anchors = st.number_input("Max anchors (multi-entity)", min_value=1, max_value=6, value=3, step=1)

    st.divider()
    st.subheader("Graph Debug")
    show_plan = st.toggle("Show plan JSON", value=True)
    show_executed_cyphers = st.toggle("Show executed Cyphers", value=True)
    debug_show_schema = st.toggle("Show schema", value=True)
    debug_print_to_console = st.toggle("Print debug to console", value=False)

    if st.button("Clear debug log"):
        st.session_state["graph_debug_lines"] = []

    st.divider()
    st.header("Hybrid Answer")
    hybrid_judge = st.toggle("Enable hybrid judge (optional)", value=False)


# ----------------------------
# Main
# ----------------------------
st.subheader("Query")
question = st.text_input("Ask in natural language (English)", value="", placeholder="e.g., What books did David Senra recommend?")
run = st.button("Run", type="primary", disabled=not question.strip())

if run:
    t0 = time.perf_counter()

    graph_debug_lines = st.session_state.get("graph_debug_lines", [])
    set_debug_print_to_console(debug_print_to_console)

    vector_settings = VectorSettings(
        grouping_enabled=bool(grouping_enabled),
        group_by_field=group_by_field,
        total_chunks_budget=int(total_chunks_budget) if grouping_enabled else 0,
        strict_group_size=bool(strict_group_size),
        top_k_chunks=int(top_k_chunks) if not grouping_enabled else 0,
        hybrid_enabled=bool(vector_hybrid_enabled),
        ranker_kind=str(ranker_kind),
        dense_limit=int(dense_limit),
        bm25_limit=int(bm25_limit),
        nprobe=int(nprobe),
        bm25_drop_ratio_search=float(bm25_drop_ratio_search),
        w_dense=float(w_dense),
        w_bm25=float(w_bm25),
        video_id=video_id,
        doc_id=doc_id,
        year=int(year),
        show=show,
        show_sanity=bool(show_sanity),
        show_raw_hit=bool(show_raw_hit),
        mode=str(vector_mode),
    )

    graph_settings = GraphSettings(
        planner_enabled=bool(planner_enabled),
        llm_row_selector=bool(llm_row_selector),
        generate_answer=bool(generate_answer),
        enable_safe_text2cypher=bool(enable_safe_text2cypher),
        show_macro_context=bool(show_macro_context),
        macro_window=int(macro_window),
        macro_context_snippet_len=int(macro_context_snippet_len),
        selector_keep_facts=int(selector_keep_facts),
        selector_keep_micros=int(selector_keep_micros),
        use_manual_schema=bool(use_manual_schema),
        schema_text=str(schema_text or ""),
        top_candidates=int(top_candidates),
        limit_rows=int(limit_rows),
        evidence_k=int(evidence_k),
        snippet_len=int(snippet_len),
        max_anchors=int(max_anchors),
        show_plan=bool(show_plan),
        show_executed_cyphers=bool(show_executed_cyphers),
        debug_show_schema=bool(debug_show_schema),
        debug_print_to_console=bool(debug_print_to_console),
        uat_mode=True,
    )

    vector_future = None
    vector_result = None

    # vector in a worker thread so Graph can run in parallel
    executor: Optional[ThreadPoolExecutor] = None
    if mode in (MODE_VECTOR_ONLY, MODE_HYBRID):
        executor = ThreadPoolExecutor(max_workers=1)
        vector_future = executor.submit(vector_retrieve, question, vector_settings)

    graph_prep = None
    graph_result = None
    graph_error = None
    picked_speaker_id = None
    picked_entity_ids: List[str] = []
    graph_driver = None

    if mode in (MODE_GRAPH_ONLY, MODE_HYBRID):
        try:
            graph_prep = graph_prepare(question, graph_settings, graph_debug_lines)
            graph_driver = graph_prep.driver
            st.session_state["graph_schema_text"] = graph_prep.schema_text

            with st.expander("Anchor resolution"):
                st.markdown("**Speaker candidates**")
                if not graph_prep.speaker_candidates.empty:
                    st.dataframe(graph_prep.speaker_candidates, use_container_width=True)
                else:
                    st.info("No Speaker candidates.")

                st.markdown("**Entity candidates (multi-mention)**")
                if graph_prep.plan_entity_queries:
                    st.caption(
                        f"Planner entity_queries: {graph_prep.plan_entity_queries} | "
                        f"entity_mode={normalize_str(graph_prep.plan.get('entity_mode'))}"
                    )
                if not graph_prep.entity_candidates.empty:
                    display_df = graph_prep.entity_candidates.drop(columns=["aliases_json"], errors="ignore")
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.info("No Entity candidates.")

                if not graph_prep.speaker_candidates.empty:
                    opts = [
                        f"{r['speaker_name']} | {r['speaker_id']}"
                        for _, r in graph_prep.speaker_candidates.iterrows()
                    ]
                    picked = st.selectbox("Pick speaker (optional)", ["(auto)"] + opts, index=0)
                    if picked != "(auto)":
                        picked_speaker_id = picked.split("|")[1].strip()

            if picked_speaker_id is None and isinstance(graph_prep.speaker_candidates, pd.DataFrame) and (not graph_prep.speaker_candidates.empty):
                picked_speaker_id = normalize_str(graph_prep.speaker_candidates.iloc[0].get("speaker_id")) or None

            client = build_openrouter_client(OPENROUTER_API_KEY, OPENROUTER_BASE_URL)
            if client is not None and not graph_prep.entity_candidates.empty:
                llm_ids = graph_pick_entities(
                    client=client,
                    question=question,
                    candidates_df=graph_prep.entity_candidates,
                    max_anchors=graph_settings.max_anchors,
                    debug_lines=graph_debug_lines,
                )
                if llm_ids:
                    picked_entity_ids = llm_ids

            if not picked_entity_ids:
                for q in graph_prep.plan_entity_queries[: int(graph_settings.max_anchors)]:
                    eid = graph_prep.per_query_top.get(q)
                    if eid and eid not in picked_entity_ids:
                        picked_entity_ids.append(eid)
                    if len(picked_entity_ids) >= int(graph_settings.max_anchors):
                        break

                if not picked_entity_ids and not graph_prep.entity_candidates.empty:
                    eid0 = normalize_str(graph_prep.entity_candidates.iloc[0].get("entity_id"))
                    if eid0:
                        picked_entity_ids = [eid0]

            graph_result = graph_execute(
                question=question,
                settings=graph_settings,
                prep=graph_prep,
                picked_speaker_id=picked_speaker_id,
                picked_entity_ids=picked_entity_ids,
                debug_lines=graph_debug_lines,
            )
        except Exception as e:
            graph_error = str(e)

    if vector_future is not None:
        try:
            vector_result = vector_future.result()
        except Exception as e:
            vector_result = None
            st.error(f"Vector retrieval failed: {type(e).__name__}: {e}")
        finally:
            if executor is not None:
                executor.shutdown(wait=False)

    if mode in (MODE_VECTOR_ONLY, MODE_HYBRID):
        st.divider()
        st.subheader("Vector Retrieval Output")
        if vector_result is None:
            st.warning("Vector retrieval did not return a result.")
        else:
            render_vector_debug_and_results(vector_result, vector_settings)

    if mode in (MODE_GRAPH_ONLY, MODE_HYBRID):
        st.divider()
        st.subheader("Graph Retrieval Output")
        if graph_error:
            st.error(graph_error)
        elif graph_result is None:
            st.warning("Graph retrieval did not return a result.")
        else:
            if graph_result.fallback_reason:
                st.warning(f"Falling back to SAFE Text2Cypher: {graph_result.fallback_reason}")
            render_graph_results(graph_result, graph_settings, graph_driver, NEO4J_DATABASE or None)

            if graph_result.result_action == "generic_text2cypher":
                st.subheader("SAFE Text2Cypher fallback (readonly enforced BEFORE execution)")
                if not graph_settings.enable_safe_text2cypher:
                    st.warning("SAFE Text2Cypher is disabled in sidebar.")
                else:
                    try:
                        cypher, df_fb = graph_safe_text2cypher(
                            question=question,
                            settings=graph_settings,
                            schema_text=graph_result.schema_text,
                            entity_ft_indexes=graph_result.entity_ft_indexes,
                            micro_ft_indexes=graph_result.micro_ft_indexes,
                            debug_lines=graph_debug_lines,
                        )
                        st.subheader("Generated Cypher (read-only)")
                        st.code(cypher, language="cypher")

                        if df_fb.empty:
                            st.info("No rows returned.")
                        else:
                            st.subheader("Rows")
                            st.dataframe(df_fb.head(200), use_container_width=True)
                            if graph_settings.uat_mode:
                                cols = {c.lower() for c in df_fb.columns}
                                if "micro_id" not in cols and "snippet" not in cols:
                                    st.warning(
                                        "UAT strict is ON, but the generated query did not return micro evidence fields "
                                        "(micro_id/snippet). Ask for evidence snippets explicitly."
                                    )
                    except Exception as e:
                        st.error(f"SAFE Text2Cypher failed: {e}")

    if mode == MODE_VECTOR_ONLY and vector_result is not None and not vector_result.error:
        if vector_settings.mode != "Retrieval Only":
            st.subheader("Answer")
            with st.spinner("Generating answer (RAG)..."):
                answer = rag_answer(question.strip(), vector_result.sources_text, CHAT_MODEL, term_scan_block=vector_result.term_scan_block)
            answer = linkify_citations(answer, vector_result.sources)
            st.markdown(answer)

            if vector_settings.mode == "RAG + Judge":
                st.subheader("Judge / Evaluation")
                with st.spinner("Judging faithfulness..."):
                    verdict = judge_answer(question.strip(), answer, vector_result.sources_text, JUDGE_MODEL, term_scan_block=vector_result.term_scan_block)
                st.json(verdict)

    if mode == MODE_GRAPH_ONLY and graph_result is not None and not graph_error:
        if graph_settings.generate_answer and graph_settings.planner_enabled:
            evidence_items = pack_micro_evidence_for_answer(
                graph_result.result_df,
                graph_result.result_action,
                max_items=12,
                max_micros_each=int(graph_settings.evidence_k),
            )
            if evidence_items:
                st.subheader("Answer (evidence-only)")
                client = build_openrouter_client(OPENROUTER_API_KEY, OPENROUTER_BASE_URL)
                ans = llm_answer(client, OPENROUTER_MODEL, question, evidence_items, graph_debug_lines)
                if ans:
                    st.markdown(ans)
                else:
                    st.info("Answer generation failed (LLM).")
            else:
                st.info("Not enough evidence items to generate an answer.")

    if mode == MODE_HYBRID:
        st.divider()
        st.subheader("Hybrid Answer")

        vector_sources_for_hybrid: List[Dict[str, Any]] = []
        if vector_result is not None and not vector_result.error:
            for i, s in enumerate(vector_result.sources[:12], start=1):
                row = {
                    "id": f"V{i}",
                    "text": s.get("text", ""),
                    "title": s.get("title"),
                    "show": s.get("show"),
                    "doc_id": s.get("doc_id"),
                    "video_id": s.get("video_id"),
                    "year": s.get("year"),
                    "start": s.get("start"),
                    "end": s.get("end"),
                    "ts_url": s.get("ts_url"),
                    "url": s.get("url"),
                }
                # Guarantee: force a usable ts_url if possible
                row["ts_url"] = _ensure_ts_url_from_source(row) or _safe_http_url(row.get("ts_url"))
                vector_sources_for_hybrid.append(row)

        graph_evidence_for_hybrid: List[Dict[str, Any]] = []
        if graph_result is not None and graph_result.result_action != "generic_text2cypher":
            graph_evidence_for_hybrid = build_hybrid_graph_evidence(
                graph_result.result_action,
                graph_result.result_df,
                max_items=8,
                max_micros_each=int(graph_settings.evidence_k),
            )

        if not vector_sources_for_hybrid and not graph_evidence_for_hybrid:
            st.info("No evidence available to generate a hybrid answer.")
        else:
            with st.spinner("Generating hybrid answer..."):
                ans = run_hybrid_answer(question.strip(), vector_sources_for_hybrid, graph_evidence_for_hybrid)

            if ans:
                ans = linkify_hybrid_citations(ans, vector_sources_for_hybrid, graph_evidence_for_hybrid)
                st.markdown(ans)
            else:
                st.warning("Hybrid answer generation failed (LLM).")

            if hybrid_judge and vector_sources_for_hybrid:
                st.subheader("Hybrid Judge / Evaluation (vector sources only)")
                # Build judge context with the SAME [V#] ids to keep it consistent
                sources_text = ""
                for v in vector_sources_for_hybrid:
                    sources_text += f"[{v['id']}]\n{v.get('text','')}\n\n"
                with st.spinner("Judging faithfulness (vector sources only)..."):
                    verdict = judge_answer(
                        question.strip(),
                        ans or "",
                        sources_text,
                        JUDGE_MODEL,
                        term_scan_block=vector_result.term_scan_block if vector_result else "",
                    )
                st.json(verdict)

    st.write({"total_s": round(time.perf_counter() - t0, 3), "embed_model": EMBED_MODEL})

    if graph_driver is not None:
        graph_close_driver(graph_driver)
