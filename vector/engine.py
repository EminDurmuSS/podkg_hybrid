from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import streamlit as st

import requests
from pymilvus import MilvusClient

try:
    from pymilvus import AnnSearchRequest  # type: ignore
except Exception:
    AnnSearchRequest = None  # type: ignore

try:
    from pymilvus import RRFRanker, WeightedRanker  # type: ignore
except Exception:
    RRFRanker = None  # type: ignore
    WeightedRanker = None  # type: ignore

from podkg_hybrid.shared.types import VectorResult, VectorSettings


import streamlit as st

# ----------------------------
# Milvus / Zilliz
# ----------------------------
MILVUS_URI = str(st.secrets.get("MILVUS_URI", "")).strip()
MILVUS_TOKEN = str(st.secrets.get("MILVUS_TOKEN", "")).strip() or None
COLLECTION_NAME = str(st.secrets.get("COLLECTION_NAME", "youtube_transcripts")).strip()

# Field names (allow override)
DENSE_VECTOR_FIELD = str(st.secrets.get("DENSE_VECTOR_FIELD", "embedding")).strip()
TEXT_FIELD = str(st.secrets.get("TEXT_FIELD", "text")).strip()

# ----------------------------
# OpenRouter
# ----------------------------
OPENROUTER_API_KEY = str(st.secrets.get("OPENROUTER_API_KEY", "")).strip()
OPENROUTER_BASE_URL = str(
    st.secrets.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
).rstrip("/")

# Models (no UI controls; secrets can override)
EMBED_MODEL = str(st.secrets.get("OPENROUTER_EMBED_MODEL", "qwen/qwen3-embedding-8b")).strip()
CHAT_MODEL = str(st.secrets.get("OPENROUTER_CHAT_MODEL", "google/gemini-3-flash-preview")).strip()
JUDGE_MODEL = str(st.secrets.get("OPENROUTER_JUDGE_MODEL", "google/gemini-3-flash-preview")).strip()

# ----------------------------
# BM25 / Hybrid
# ----------------------------
ENABLE_BM25 = str(st.secrets.get("ENABLE_BM25", "true")).lower() in ("1", "true", "yes", "y")
BM25_SPARSE_FIELD = str(st.secrets.get("BM25_SPARSE_FIELD", "text_sparse")).strip()

DENSE_METRIC_TYPE = str(st.secrets.get("DENSE_METRIC_TYPE", "COSINE")).strip().upper()
DENSE_NPROBE_DEFAULT = int(str(st.secrets.get("DENSE_NPROBE", "16")))
DENSE_SEARCH_LEVEL = str(st.secrets.get("DENSE_SEARCH_LEVEL", "")).strip()  # AUTOINDEX level 1..5

HYBRID_RANKER_DEFAULT = str(st.secrets.get("HYBRID_RANKER", "rrf")).lower().strip()  # rrf | weighted
HYBRID_WEIGHT_DENSE_DEFAULT = float(str(st.secrets.get("HYBRID_WEIGHT_DENSE", "0.6")))
HYBRID_WEIGHT_BM25_DEFAULT = float(str(st.secrets.get("HYBRID_WEIGHT_BM25", "0.4")))
BM25_DROP_RATIO_SEARCH_DEFAULT = float(str(st.secrets.get("BM25_DROP_RATIO_SEARCH", "0.0")))


# Grouping Search
GROUP_SIZE_FIXED = 4


# ----------------------------
# Small helpers
# ----------------------------
def _require_env():
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is missing.")
    if not MILVUS_URI:
        raise RuntimeError("MILVUS_URI is missing.")
    if "zilliz" in MILVUS_URI.lower() and not MILVUS_TOKEN:
        raise RuntimeError("Zilliz Cloud detected but MILVUS_TOKEN is missing.")


def _or_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "podkg-streamlit-rag",
    }


def get_http_session() -> requests.Session:
    return requests.Session()


def get_milvus_client() -> MilvusClient:
    if MILVUS_TOKEN:
        return MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    return MilvusClient(MILVUS_URI)


def openrouter_embed(texts: List[str], model: str) -> List[List[float]]:
    sess = get_http_session()
    url = f"{OPENROUTER_BASE_URL}/embeddings"
    payload: Dict[str, Any] = {"model": model, "input": texts}
    r = sess.post(url, headers=_or_headers(), json=payload, timeout=60)
    r.raise_for_status()
    j = r.json()
    data = j.get("data") or []
    vecs = []
    for row in data:
        vecs.append(row["embedding"])
    if len(vecs) != len(texts):
        raise RuntimeError(f"Embedding mismatch: {len(vecs)} != {len(texts)}")
    return vecs


def openrouter_chat(messages: List[Dict[str, str]], model: str, temperature: float = 0.2) -> str:
    sess = get_http_session()
    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    payload = {"model": model, "messages": messages, "temperature": temperature}
    r = sess.post(url, headers=_or_headers(), json=payload, timeout=120)
    r.raise_for_status()
    j = r.json()
    return (j.get("choices") or [{}])[0].get("message", {}).get("content", "")


def _escape_str(v: str) -> str:
    return (v or "").replace("\\", "\\\\").replace("'", "\\'")


def build_filter_expr(video_id: str, doc_id: str, year: int, show: str) -> Optional[str]:
    parts = []
    if video_id:
        parts.append(f"video_id == '{_escape_str(video_id)}'")
    if doc_id:
        parts.append(f"doc_id == '{_escape_str(doc_id)}'")
    if year and year > 0:
        parts.append(f"year == {int(year)}")
    if show:
        parts.append(f"show == '{_escape_str(show)}'")
    if not parts:
        return None
    return " and ".join(parts)


def _dense_search_params(metric_type: str, nprobe: int) -> Dict[str, Any]:
    sp: Dict[str, Any] = {"metric_type": metric_type}
    if DENSE_SEARCH_LEVEL:
        try:
            sp["level"] = int(DENSE_SEARCH_LEVEL)
            return sp
        except Exception:
            pass
    sp["params"] = {"nprobe": int(nprobe)}
    return sp


def get_collection_field_names() -> List[str]:
    c = get_milvus_client()
    try:
        desc = c.describe_collection(COLLECTION_NAME)
        fields = desc.get("fields") or []
        names = set()
        for f in fields:
            name = f.get("name")
            if name:
                names.add(name)
        return sorted(list(names))
    except Exception:
        return []


def resolve_output_fields(requested: List[str]) -> Tuple[List[str], List[str]]:
    schema_fields = set(get_collection_field_names())
    if not schema_fields:
        return requested, []
    effective = [f for f in requested if f in schema_fields]
    missing = [f for f in requested if f not in schema_fields]
    return effective, missing


def build_ts_url(video_id: str, start_sec: Optional[float]) -> str:
    if not video_id:
        return ""
    try:
        t = int(float(start_sec or 0))
    except Exception:
        t = 0
    return f"https://www.youtube.com/watch?v={video_id}&t={t}s"


def _hit_to_dict(h: Any) -> Dict[str, Any]:
    if isinstance(h, dict):
        ent = h.get("entity", {})
        if not isinstance(ent, dict):
            ent = {}
        return {
            "id": h.get("id", None),
            "distance": h.get("distance", None),
            "score": h.get("score", None),
            **ent,
        }

    if hasattr(h, "to_dict"):
        try:
            hd = h.to_dict()  # type: ignore
            ent = hd.get("entity", None) or hd.get("fields", None)
            if isinstance(ent, dict):
                ent_dict = ent
            else:
                ent_dict = {k: v for k, v in hd.items() if k not in ("entity", "fields")}
            return {
                "id": hd.get("id", getattr(h, "id", None)),
                "distance": hd.get("distance", getattr(h, "distance", None)),
                "score": hd.get("score", getattr(h, "score", None)),
                **ent_dict,
            }
        except Exception:
            pass

    try:
        ent = getattr(h, "entity", None)
        if ent is None:
            ent = getattr(h, "fields", None)

        ent_dict: Dict[str, Any] = {}
        if isinstance(ent, dict):
            ent_dict = ent
        else:
            try:
                ent_dict = dict(ent) if ent is not None else {}
            except Exception:
                ent_dict = {}

        return {
            "id": getattr(h, "id", None),
            "distance": getattr(h, "distance", None),
            "score": getattr(h, "score", None),
            **ent_dict,
        }
    except Exception:
        return {}


def _normalize_hits(res0: Any) -> List[Dict[str, Any]]:
    hits = res0 or []
    out: List[Dict[str, Any]] = []
    for h in hits:
        d = _hit_to_dict(h)
        if d:
            out.append(d)
    return out


def _grouping_kwargs(group_by_field: Optional[str], group_size: int, strict_group_size: bool) -> Dict[str, Any]:
    kw: Dict[str, Any] = {}
    if group_by_field:
        kw["group_by_field"] = str(group_by_field)
        if int(group_size) > 0:
            kw["group_size"] = int(group_size)
            kw["strict_group_size"] = bool(strict_group_size)
    return kw


def milvus_dense_search(
    query_vec: List[float],
    top_k: int,
    expr: Optional[str],
    output_fields: List[str],
    *,
    nprobe: int = DENSE_NPROBE_DEFAULT,
    metric_type: str = DENSE_METRIC_TYPE,
    group_by_field: Optional[str] = None,
    group_size: int = 0,
    strict_group_size: bool = False,
) -> Tuple[List[Dict[str, Any]], bool]:
    client = get_milvus_client()

    base_kwargs: Dict[str, Any] = dict(
        collection_name=COLLECTION_NAME,
        data=[query_vec],
        anns_field=DENSE_VECTOR_FIELD,
        search_params=_dense_search_params(metric_type, nprobe),
        limit=int(top_k),
        filter=expr or "",
        output_fields=output_fields,
    )

    gkw = _grouping_kwargs(group_by_field, group_size, strict_group_size)
    used_grouping = bool(gkw)

    try:
        res = client.search(**base_kwargs, **gkw)
        res0 = res[0] if res else []
        return _normalize_hits(res0), used_grouping
    except TypeError:
        res = client.search(**base_kwargs)
        res0 = res[0] if res else []
        return _normalize_hits(res0), False


def _make_hybrid_ranker(ranker_kind: str, w_dense: float, w_bm25: float):
    kind = (ranker_kind or "rrf").lower().strip()
    if kind == "weighted":
        if WeightedRanker is not None:
            return WeightedRanker(weights=[float(w_dense), float(w_bm25)])
        if RRFRanker is not None:
            return RRFRanker()
        return None
    if RRFRanker is not None:
        return RRFRanker()
    return None


def milvus_hybrid_search(
    query_text: str,
    query_vec: List[float],
    top_k: int,
    expr: Optional[str],
    output_fields: List[str],
    *,
    dense_limit: int,
    bm25_limit: int,
    ranker_kind: str,
    w_dense: float,
    w_bm25: float,
    nprobe: int,
    metric_type: str,
    bm25_drop_ratio_search: float,
    group_by_field: Optional[str] = None,
    group_size: int = 0,
    strict_group_size: bool = False,
) -> Tuple[List[Dict[str, Any]], bool]:
    if not ENABLE_BM25:
        raise RuntimeError("ENABLE_BM25=false; hybrid search disabled by env.")
    if AnnSearchRequest is None:
        raise RuntimeError("AnnSearchRequest import failed. Upgrade pymilvus.")
    client = get_milvus_client()
    if not hasattr(client, "hybrid_search"):
        raise RuntimeError("This pymilvus/MilvusClient version lacks hybrid_search.")

    dense_req = AnnSearchRequest(
        data=[query_vec],
        anns_field=DENSE_VECTOR_FIELD,
        param=_dense_search_params(metric_type, nprobe),
        limit=int(dense_limit),
        expr=expr or "",
    )

    bm25_param: Dict[str, Any] = {"metric_type": "BM25", "params": {}}
    if bm25_drop_ratio_search and bm25_drop_ratio_search > 0:
        bm25_param["params"]["drop_ratio_search"] = float(bm25_drop_ratio_search)

    bm25_req = AnnSearchRequest(
        data=[query_text],
        anns_field=BM25_SPARSE_FIELD,
        param=bm25_param,
        limit=int(bm25_limit),
        expr=expr or "",
    )

    ranker = _make_hybrid_ranker(ranker_kind, w_dense, w_bm25)

    base_kwargs: Dict[str, Any] = dict(
        collection_name=COLLECTION_NAME,
        reqs=[dense_req, bm25_req],
        ranker=ranker,
        limit=int(top_k),
        output_fields=output_fields,
    )

    gkw = _grouping_kwargs(group_by_field, group_size, strict_group_size)
    used_grouping = bool(gkw)

    try:
        res = client.hybrid_search(**base_kwargs, **gkw)
        res0 = res[0] if res else []
        return _normalize_hits(res0), used_grouping
    except TypeError:
        res = client.hybrid_search(**base_kwargs)
        res0 = res[0] if res else []
        return _normalize_hits(res0), False


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_multiword_proper_nouns(q: str) -> List[str]:
    terms = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", q or "")
    seen = set()
    out: List[str] = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def extract_quoted_claim(q: str) -> Optional[str]:
    m = re.search(r"['\"]([^'\"]{6,240})['\"]", q or "")
    return m.group(1).strip() if m else None


def term_scan(sources: List[Dict[str, Any]], terms: List[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for term in terms:
        t = _norm(term)
        hits: List[str] = []
        for s in sources:
            if t and t in _norm(s.get("text", "")):
                hits.append(s["sid"])
        result[term] = {"hits": hits}
    return result


def build_term_scan_block(scan: Dict[str, Any]) -> str:
    lines: List[str] = []
    for term, obj in scan.items():
        hits = obj.get("hits") or []
        if hits:
            lines.append(f"- {term}: FOUND in {', '.join(hits)}")
        else:
            lines.append(f"- {term}: NOT FOUND")
    return "\n".join(lines)


def _get_text_from_hit(h: Dict[str, Any]) -> str:
    for k in (TEXT_FIELD, "text", "content", "chunk_text"):
        v = h.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def format_sources(hits: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    sources: List[Dict[str, Any]] = []
    blocks: List[str] = []

    for i, h in enumerate(hits, start=1):
        sid = f"S{i}"
        full_txt = _get_text_from_hit(h)

        video_id = h.get("video_id", "") or ""
        start_sec = h.get("start_sec", None)
        ts_url = h.get("ts_url", "") or ""
        if not ts_url:
            ts_url = build_ts_url(video_id, start_sec)

        src = {
            "sid": sid,
            "title": h.get("title", "") or h.get("video_title", "") or "",
            "show": h.get("show", "") or "",
            "video_id": video_id,
            "doc_id": h.get("doc_id", "") or "",
            "year": h.get("year", None),
            "start_sec": start_sec,
            "end_sec": h.get("end_sec", None),
            "ts_url": ts_url,
            "chunk_id": h.get("chunk_id", "") or "",
            "chunk_idx": h.get("chunk_idx", None),
            "source_file": h.get("source_file", "") or "",
            "micro_count": h.get("micro_count", None),
            "approx_tokens": h.get("approx_tokens", None),
            "duration_s": h.get("duration_s", None),
            "reason": h.get("reason", "") or "",
            "primary_speaker": h.get("primary_speaker", "") or "",
            "distance": h.get("distance", None),
            "score": h.get("score", None),
            "id": h.get("id", None),
            "text": full_txt,
            "text_full": full_txt,
            "_raw": h,
        }
        sources.append(src)

        blocks.append(
            f"[{sid}]\n"
            f"title: {src['title']}\n"
            f"show: {src['show']}\n"
            f"video_id: {src['video_id']}\n"
            f"doc_id: {src['doc_id']}\n"
            f"chunk_id: {src['chunk_id']}  chunk_idx: {src['chunk_idx']}\n"
            f"start_sec: {src['start_sec']}  url: {src['ts_url']}\n"
            f"primary_speaker: {src['primary_speaker']}\n"
            f"text:\n{src['text']}\n"
        )

    return "\n\n".join(blocks), sources


_CIT_RE = re.compile(r"\[(S\d+)\]")


def linkify_citations(answer_md: str, sources: List[Dict[str, Any]]) -> str:
    if not answer_md:
        return answer_md or ""

    sid2url = {s.get("sid"): (s.get("ts_url") or "") for s in (sources or [])}

    def repl(m: re.Match) -> str:
        sid = m.group(1)
        url = sid2url.get(sid, "")
        if url:
            return f"[{sid}]({url})"
        return f"[{sid}]"

    return _CIT_RE.sub(repl, answer_md)


def rag_answer(question: str, sources_text: str, model: str, term_scan_block: str = "") -> str:
    system = (
        "You are a careful analyst. Answer the user's question using ONLY the provided SOURCES.\n"
        "Rules:\n"
        "- If sources are insufficient, say you could not find evidence in the dataset.\n"
        "- Do NOT invent quotes, names, years, or claims.\n"
        "- Every claim must be backed by citations like [S1], [S2].\n"
        "- Provide clickable timestamps by citing the source id (the UI will render the link).\n"
        "- Keep it professional and structured.\n"
        "- IMPORTANT: If TERM_SCAN says a term is FOUND, do NOT claim it is not mentioned.\n"
    )
    user = f"QUESTION:\n{question}\n\nTERM_SCAN:\n{term_scan_block}\n\nSOURCES:\n{sources_text}"
    return openrouter_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model,
        temperature=0.2,
    )


def judge_answer(question: str, answer: str, sources_text: str, model: str, term_scan_block: str = "") -> Dict[str, Any]:
    system = (
        "You are a strict evaluator for a RAG system.\n"
        "You must judge if the ANSWER is fully supported by SOURCES.\n"
        "Return ONLY valid JSON.\n"
        "IMPORTANT: If TERM_SCAN says a term is FOUND, penalize any claim that it is not mentioned.\n"
    )
    user = (
        f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nTERM_SCAN:\n{term_scan_block}\n\n"
        f"SOURCES:\n{sources_text}\n\n"
        "Return JSON with keys:\n"
        "{"
        "\"groundedness\": 0-1, "
        "\"has_hallucination\": true/false, "
        "\"citation_quality\": 0-1, "
        "\"coverage\": 0-1, "
        "\"notes\": \"short\""
        "}"
    )
    raw = openrouter_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model,
        temperature=0.0,
    )
    try:
        return json.loads(raw)
    except Exception:
        return {"error": "judge_json_parse_failed", "raw": raw}


def collect_sanity_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    c = get_milvus_client()
    info["collection"] = COLLECTION_NAME
    try:
        info["has_collection"] = c.has_collection(COLLECTION_NAME)
    except Exception as e:
        info["has_collection_error"] = f"{type(e).__name__}: {e}"
    try:
        c.load_collection(COLLECTION_NAME)
        info["load_collection"] = "OK"
    except Exception as e:
        info["load_collection_error"] = f"{type(e).__name__}: {e}"
    try:
        info["collection_stats"] = c.get_collection_stats(COLLECTION_NAME)
    except Exception as e:
        info["collection_stats_error"] = f"{type(e).__name__}: {e}"
    try:
        desc = c.describe_collection(COLLECTION_NAME)
        info["schema"] = desc
        info["schema_fields"] = get_collection_field_names()
    except Exception as e:
        info["schema_error"] = f"{type(e).__name__}: {e}"
    try:
        if hasattr(c, "list_indexes"):
            info["indexes"] = c.list_indexes(COLLECTION_NAME)  # type: ignore
    except Exception as e:
        info["indexes_error"] = f"{type(e).__name__}: {e}"

    info["config"] = {
        "embed_model": EMBED_MODEL,
        "answer_model": CHAT_MODEL,
        "judge_model": JUDGE_MODEL,
        "dense_field": DENSE_VECTOR_FIELD,
        "metric": DENSE_METRIC_TYPE,
        "nprobe_default": int(DENSE_NPROBE_DEFAULT),
        "level": DENSE_SEARCH_LEVEL or "(unset)",
        "bm25_enabled": bool(ENABLE_BM25),
        "bm25_field": BM25_SPARSE_FIELD,
    }
    return info


def vector_retrieve(question: str, settings: VectorSettings) -> VectorResult:
    result = VectorResult()
    try:
        _require_env()
    except Exception as e:
        result.error = str(e)
        return result

    q = (question or "").strip()
    if not q:
        result.error = "Question is empty."
        return result

    if settings.show_sanity:
        result.debug_sanity = collect_sanity_info()

    expr = build_filter_expr(
        settings.video_id.strip(),
        settings.doc_id.strip(),
        int(settings.year),
        settings.show.strip(),
    )
    result.filter_expr = expr

    t0 = time.perf_counter()
    qv = openrouter_embed([q], EMBED_MODEL)[0]
    t_embed = time.perf_counter()

    requested_output_fields = [
        "chunk_id",
        "video_id",
        "title",
        "show",
        "year",
        "doc_id",
        "chunk_idx",
        "start_sec",
        "end_sec",
        "ts_url",
        TEXT_FIELD,
        "text",
        "source_file",
        "micro_count",
        "approx_tokens",
        "duration_s",
        "reason",
        "primary_speaker",
    ]

    if settings.grouping_enabled and settings.group_by_field and settings.group_by_field not in requested_output_fields:
        requested_output_fields.append(str(settings.group_by_field))

    output_fields, missing_fields = resolve_output_fields(requested_output_fields)
    result.output_fields = output_fields
    result.missing_fields = missing_fields

    used_hybrid = False
    used_grouping = False
    effective_limit = int(settings.total_chunks_budget // GROUP_SIZE_FIXED) if settings.grouping_enabled else int(settings.top_k_chunks)

    try:
        if settings.hybrid_enabled and ENABLE_BM25:
            hits, used_grouping = milvus_hybrid_search(
                query_text=q,
                query_vec=qv,
                top_k=effective_limit,
                expr=expr,
                output_fields=output_fields,
                dense_limit=int(settings.dense_limit),
                bm25_limit=int(settings.bm25_limit),
                ranker_kind=settings.ranker_kind,
                w_dense=float(settings.w_dense),
                w_bm25=float(settings.w_bm25),
                nprobe=int(settings.nprobe),
                metric_type=DENSE_METRIC_TYPE,
                bm25_drop_ratio_search=float(settings.bm25_drop_ratio_search),
                group_by_field=str(settings.group_by_field) if settings.grouping_enabled else None,
                group_size=GROUP_SIZE_FIXED if settings.grouping_enabled else 0,
                strict_group_size=bool(settings.strict_group_size) if settings.grouping_enabled else False,
            )
            used_hybrid = True
        else:
            hits, used_grouping = milvus_dense_search(
                qv,
                effective_limit,
                expr,
                output_fields=output_fields,
                nprobe=int(settings.nprobe),
                group_by_field=str(settings.group_by_field) if settings.grouping_enabled else None,
                group_size=GROUP_SIZE_FIXED if settings.grouping_enabled else 0,
                strict_group_size=bool(settings.strict_group_size) if settings.grouping_enabled else False,
            )
    except Exception as e:
        result.error = f"Search failed: {type(e).__name__}: {e}"
        return result

    t_search = time.perf_counter()

    result.used_hybrid = used_hybrid
    result.used_grouping = used_grouping
    result.retrieval_limit = effective_limit
    result.approx_total_chunks = effective_limit * GROUP_SIZE_FIXED if settings.grouping_enabled else effective_limit
    result.group_by_field = settings.group_by_field if settings.grouping_enabled else None

    if not hits:
        result.error = "No results found (after filters)."
        return result

    sources_text, sources = format_sources(hits)

    terms = extract_multiword_proper_nouns(q)
    claim = extract_quoted_claim(q)
    scan_terms: List[str] = []
    if claim:
        scan_terms.append(claim)
    scan_terms += terms
    scan = term_scan(sources, scan_terms)
    scan_block = build_term_scan_block(scan)

    result.hits = hits
    result.sources = sources
    result.sources_text = sources_text
    result.term_scan = scan
    result.term_scan_block = scan_block
    result.timings = {
        "embed_s": round(t_embed - t0, 3),
        "search_s": round(t_search - t_embed, 3),
        "total_s": round(t_search - t0, 3),
    }
    return result
