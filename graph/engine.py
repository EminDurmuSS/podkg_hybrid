from __future__ import annotations

# Graph engine implementation. Content appended in sections.

import json
import os
import re
import streamlit as st
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from neo4j import GraphDatabase
import neo4j

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from podkg_hybrid.shared.types import GraphPrep, GraphResult, GraphSettings


# ----------------------------
# ENV
# ----------------------------
NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_USERNAME = st.secrets.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
NEO4J_DATABASE = st.secrets.get("NEO4J_DATABASE", "")

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
OPENROUTER_BASE_URL = st.secrets.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", "google/gemini-3-flash-preview")


# ----------------------------
# Safety (read-only)
# ----------------------------
FORBIDDEN_CYPHER = re.compile(
    r"(?is)\b("
    r"CREATE|MERGE|DELETE|DETACH\s+DELETE|SET|DROP|LOAD\s+CSV|"
    r"ALTER|GRANT|REVOKE|"
    r")\b"
)


# ----------------------------
# Small utils
# ----------------------------
def normalize_str(x: Any) -> str:
    return (str(x) if x is not None else "").strip()


def to_lower(x: Any) -> str:
    return normalize_str(x).lower()


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def json_dumps_safe(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


KEYWORD_SPLIT_RE = re.compile(r"\s*(?:,|;|\||/|\band\b|\&)\s*", flags=re.I)
QUOTED_TERM_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')


def split_keyword_terms(raw: str, max_terms: int = 6) -> List[str]:
    s = normalize_str(raw)
    if not s:
        return []

    terms: List[str] = []

    def _repl(m: re.Match) -> str:
        val = normalize_str(m.group(1) or m.group(2))
        if val:
            terms.append(val)
        return " "

    s = QUOTED_TERM_RE.sub(_repl, s)
    for part in KEYWORD_SPLIT_RE.split(s):
        t = normalize_str(part)
        if t:
            terms.append(t)

    seen = set()
    out: List[str] = []
    for t in terms:
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= max_terms:
            break
    return out


DEBUG_PRINT_TO_CONSOLE = False


def set_debug_print_to_console(enabled: bool) -> None:
    global DEBUG_PRINT_TO_CONSOLE
    DEBUG_PRINT_TO_CONSOLE = bool(enabled)


def debug_log(debug_lines: List[str], msg: str, print_to_console: bool = False):
    debug_lines.append(msg)
    if print_to_console or DEBUG_PRINT_TO_CONSOLE:
        print(msg)


def get_driver(uri: str, user: str, pwd: str) -> neo4j.Driver:
    return GraphDatabase.driver(uri, auth=(user, pwd))


def run_read_records(
    driver: neo4j.Driver,
    database: Optional[str],
    cypher: str,
    params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    res = driver.execute_query(
        cypher,
        parameters_=params or {},
        database_=database or None,
        routing_=neo4j.RoutingControl.READ,
    ).records
    return [r.data() for r in res]


def run_read_df(
    driver: neo4j.Driver,
    database: Optional[str],
    cypher: str,
    params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    rows = run_read_records(driver, database, cypher, params)
    return pd.DataFrame(rows)


def try_read_df(
    driver: neo4j.Driver,
    database: Optional[str],
    cypher: str,
    params: Optional[Dict[str, Any]] = None,
    debug_lines: Optional[List[str]] = None,
) -> pd.DataFrame:
    try:
        return run_read_df(driver, database, cypher, params)
    except Exception as e:
        if debug_lines is not None:
            debug_log(debug_lines, f"[try_read_df] failed: {e} | cypher={cypher.strip()[:220]}")
        return pd.DataFrame()


def build_openrouter_client(api_key: str, base_url: str):
    if OpenAI is None:
        return None
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={"HTTP-Referer": "http://localhost:8501", "X-Title": "Fact-first NL2Neo4j"},
    )


def try_fetch_schema(driver: neo4j.Driver, database: Optional[str]) -> str:
    node_props = run_read_records(
        driver,
        database,
        "CALL db.schema.nodeTypeProperties() "
        "YIELD nodeType, propertyName, propertyTypes "
        "RETURN nodeType, collect({name: propertyName, types: propertyTypes}) AS props",
    )
    rel_props = run_read_records(
        driver,
        database,
        "CALL db.schema.relTypeProperties() "
        "YIELD relType, propertyName, propertyTypes "
        "RETURN relType, collect({name: propertyName, types: propertyTypes}) AS props",
    )

    lines = ["Node properties:"]
    for r in node_props:
        label = str(r["nodeType"]).lstrip(":") or str(r["nodeType"])
        props = r.get("props") or []
        pretty = []
        for p in props:
            t = p.get("types")
            t_str = t[0] if isinstance(t, list) and t else (str(t) if t else "ANY")
            pretty.append(f"{p.get('name')}: {t_str}")
        lines.append(f"{label} " + "{" + ", ".join(pretty) + "}")

    lines.append("Relationship properties:")
    for r in rel_props:
        rel_type = r.get("relType")
        props = r.get("props") or []
        pretty = []
        for p in props:
            t = p.get("types")
            t_str = t[0] if isinstance(t, list) and t else (str(t) if t else "ANY")
            pretty.append(f"{p.get('name')}: {t_str}")
        lines.append(f"{rel_type} " + "{" + ", ".join(pretty) + "}")

    return "\n".join(lines)


def compact_schema(schema_text: str, max_lines: int = 80) -> str:
    if not schema_text:
        return ""
    lines = [ln.strip() for ln in schema_text.splitlines() if ln.strip()]
    return "\n".join(lines[:max_lines])


SHOW_INDEXES_CYPHER = """
SHOW INDEXES
YIELD name, type, entityType, labelsOrTypes, properties
RETURN name, type, entityType, labelsOrTypes, properties
ORDER BY name
"""

SHOW_FULLTEXT_INDEXES_CYPHER = """
SHOW FULLTEXT INDEXES
YIELD name, entityType, labelsOrTypes, properties
RETURN name, "FULLTEXT" AS type, entityType, labelsOrTypes, properties
ORDER BY name
"""

CALL_DB_INDEXES_FALLBACK = """
CALL db.indexes()
YIELD name, type, entityType, labelsOrTypes, properties
RETURN name, type, entityType, labelsOrTypes, properties
ORDER BY name
"""


def detect_indexes(driver: neo4j.Driver, database: Optional[str], debug_lines: List[str]) -> pd.DataFrame:
    df = try_read_df(driver, database, SHOW_FULLTEXT_INDEXES_CYPHER, debug_lines=debug_lines)
    if not df.empty:
        return df
    df = try_read_df(driver, database, SHOW_INDEXES_CYPHER, debug_lines=debug_lines)
    if not df.empty:
        return df
    df = try_read_df(driver, database, CALL_DB_INDEXES_FALLBACK, debug_lines=debug_lines)
    return df if df is not None else pd.DataFrame()


def pick_fulltext_index(df_indexes: pd.DataFrame, label: str, prefer_props: List[str]) -> List[str]:
    if df_indexes is None or df_indexes.empty:
        return []
    d = df_indexes.copy()
    d["type_up"] = d["type"].astype(str).str.upper()
    d = d[d["type_up"].str.contains("FULLTEXT", na=False)].copy()
    if d.empty:
        return []

    label_l = label.lower()

    def score_row(row) -> int:
        labels = row.get("labelsOrTypes") or []
        props = row.get("properties") or []
        labels_s = " ".join([str(x).lower() for x in labels]) if isinstance(labels, list) else str(labels).lower()
        props_s = " ".join([str(x).lower() for x in props]) if isinstance(props, list) else str(props).lower()
        score = 0
        if label_l in labels_s:
            score += 100
        for i, p in enumerate(prefer_props):
            if p.lower() in props_s:
                score += 30 - i
        return score

    d["pref_score"] = d.apply(score_row, axis=1)
    d = d.sort_values("pref_score", ascending=False)
    return [str(x) for x in d["name"].tolist() if str(x)]


def get_capabilities(driver: neo4j.Driver, db: Optional[str]) -> Dict[str, Any]:
    caps: Dict[str, Any] = {}
    try:
        labels = run_read_records(driver, db, "CALL db.labels() YIELD label RETURN collect(label) AS labels")
        caps["labels"] = labels[0]["labels"] if labels else []
    except Exception:
        caps["labels"] = []
    try:
        rels = run_read_records(
            driver,
            db,
            "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS rels",
        )
        caps["relationship_types"] = rels[0]["rels"] if rels else []
    except Exception:
        caps["relationship_types"] = []
    try:
        props = run_read_records(driver, db, "CALL db.propertyKeys() YIELD propertyKey RETURN collect(propertyKey) AS props")
        caps["property_keys"] = props[0]["props"] if props else []
    except Exception:
        caps["property_keys"] = []
    return caps


TOOL_SPEAKER_CANDIDATES = """
MATCH (s:Speaker)
WHERE toLower(s.name) CONTAINS toLower($frag)
RETURN s.id AS speaker_id, s.name AS speaker_name
ORDER BY speaker_name
LIMIT $k
"""

TOOL_SPEAKER_RESOLVES_TO_ENTITY = """
MATCH (s:Speaker)
WHERE toLower(s.name) CONTAINS toLower($frag)
OPTIONAL MATCH (s)-[:RESOLVES_TO]->(e:Entity)
RETURN
  s.name AS speaker_name,
  s.id AS speaker_id,
  e.id AS entity_id,
  e.canonical AS entity_name,
  e.canonical_key AS canonical_key,
  e.type_label AS type_label,
  e.type AS type,
  e.aliases_json AS aliases_json
LIMIT $k
"""

TOOL_SPEAKER_TO_ENTITY_BY_ID = """
MATCH (s:Speaker {id:$speaker_id})
OPTIONAL MATCH (s)-[:RESOLVES_TO]->(e:Entity)
RETURN e.id AS entity_id, e.canonical AS entity_name, e.canonical_key AS canonical_key, e.type_label AS type_label, e.type AS type
LIMIT 1
"""

TOOL_ENTITY_TO_SPEAKER = """
MATCH (s:Speaker)-[:RESOLVES_TO]->(e:Entity {id:$entity_id})
RETURN s.id AS speaker_id, s.name AS speaker_name
LIMIT 5
"""

TOOL_ENTITY_FULLTEXT = """
CALL db.index.fulltext.queryNodes($index, $q) YIELD node, score
WHERE node:Entity
RETURN
  node.id AS entity_id,
  node.canonical AS entity_name,
  node.canonical_key AS canonical_key,
  node.type_label AS type_label,
  node.type AS type,
  node.aliases_json AS aliases_json,
  score AS score
ORDER BY score DESC
LIMIT $k
"""

TOOL_ENTITY_CONTAINS = """
MATCH (e:Entity)
WHERE
  toLower(e.canonical) CONTAINS toLower($q)
  OR toLower(e.canonical_key) CONTAINS toLower($q)
  OR (e.aliases_json IS NOT NULL AND toLower(e.aliases_json) CONTAINS toLower($q))
RETURN
  e.id AS entity_id,
  e.canonical AS entity_name,
  e.canonical_key AS canonical_key,
  e.type_label AS type_label,
  e.type AS type,
  e.aliases_json AS aliases_json,
  0.0 AS score
ORDER BY entity_name
LIMIT $k
"""

TOOL_ENTITY_FACTS_EVIDENCE = """
MATCH (anchor:Entity {id:$entity_id})
MATCH (f:Fact)-[:HAS_SUBJECT|HAS_OBJECT]->(anchor)
OPTIONAL MATCH (f)-[:HAS_SUBJECT]->(sub:Entity)
OPTIONAL MATCH (f)-[:HAS_OBJECT]->(obj:Entity)
WITH anchor, f, sub, obj,
     CASE
       WHEN (f)-[:HAS_SUBJECT]->(anchor) THEN "ANCHOR_IS_SUBJECT"
       WHEN (f)-[:HAS_OBJECT]->(anchor) THEN "ANCHOR_IS_OBJECT"
       ELSE "UNKNOWN"
     END AS anchor_role,
     CASE
       WHEN (f)-[:HAS_SUBJECT]->(anchor) THEN obj
       WHEN (f)-[:HAS_OBJECT]->(anchor) THEN sub
       ELSE obj
     END AS other
CALL {
  WITH f
  OPTIONAL MATCH (f)-[:EVIDENCED_BY]->(m:Micro)
  OPTIONAL MATCH (m)-[:IN_MACRO]->(ma:Macro)
  WITH m, ma
  WHERE m IS NOT NULL
  ORDER BY m.start ASC
  RETURN collect({
    micro_id: m.id,
    start: m.start,
    end: coalesce(m.end, m.start),
    speaker: m.speaker,
    snippet: substring(coalesce(m.text,""), 0, $snippet_len),
    url: m.url,
    doc_id: m.doc_id,
    macro_id: ma.id,
    doc_title: ma.doc_title,
    doc_year: ma.doc_year,
    doc_url: ma.doc_url,
    youtube_id: ma.youtube_id
  })[0..$evidence_k] AS evidence_rel
}
CALL {
  WITH f
  UNWIND coalesce(f.evidence_micro_ids, []) AS mid
  OPTIONAL MATCH (m2:Micro {id: mid})
  OPTIONAL MATCH (m2)-[:IN_MACRO]->(ma2:Macro)
  WITH m2, ma2
  WHERE m2 IS NOT NULL
  ORDER BY m2.start ASC
  RETURN collect({
    micro_id: m2.id,
    start: m2.start,
    end: coalesce(m2.end, m2.start),
    speaker: m2.speaker,
    snippet: substring(coalesce(m2.text,""), 0, $snippet_len),
    url: m2.url,
    doc_id: m2.doc_id,
    macro_id: ma2.id,
    doc_title: ma2.doc_title,
    doc_year: ma2.doc_year,
    doc_url: ma2.doc_url,
    youtube_id: ma2.youtube_id
  })[0..$evidence_k] AS evidence_prop
}
RETURN
  anchor.id AS anchor_entity_id,
  anchor.canonical AS anchor_entity_name,
  f.id AS fact_id,
  coalesce(f.pred_norm, f.predicate) AS predicate,
  f.support_count AS support_count,
  anchor_role,
  sub.id AS subject_id,
  sub.canonical AS subject,
  obj.id AS object_id,
  obj.canonical AS object,
  obj.type_label AS object_type_label,
  obj.type AS object_type,
  other.id AS other_entity_id,
  other.canonical AS other_entity,
  other.type_label AS other_type_label,
  other.type AS other_type,
  evidence_rel,
  evidence_prop
ORDER BY coalesce(f.support_count, 0) DESC, f.id DESC
LIMIT $limit
"""

TOOL_MULTI_ENTITY_FACTS_EVIDENCE = """
UNWIND $entity_ids AS anchor_id
CALL {
  WITH anchor_id
  MATCH (anchor:Entity {id: anchor_id})
  MATCH (f:Fact)-[:HAS_SUBJECT|HAS_OBJECT]->(anchor)
  OPTIONAL MATCH (f)-[:HAS_SUBJECT]->(sub:Entity)
  OPTIONAL MATCH (f)-[:HAS_OBJECT]->(obj:Entity)
  WITH anchor, f, sub, obj,
       CASE
         WHEN (f)-[:HAS_SUBJECT]->(anchor) THEN "ANCHOR_IS_SUBJECT"
         WHEN (f)-[:HAS_OBJECT]->(anchor) THEN "ANCHOR_IS_OBJECT"
         ELSE "UNKNOWN"
       END AS anchor_role,
       CASE
         WHEN (f)-[:HAS_SUBJECT]->(anchor) THEN obj
         WHEN (f)-[:HAS_OBJECT]->(anchor) THEN sub
         ELSE obj
       END AS other
  CALL {
    WITH f
    OPTIONAL MATCH (f)-[:EVIDENCED_BY]->(m:Micro)
    OPTIONAL MATCH (m)-[:IN_MACRO]->(ma:Macro)
    WITH m, ma
    WHERE m IS NOT NULL
    ORDER BY m.start ASC
    RETURN collect({
      micro_id: m.id, start: m.start, end: coalesce(m.end, m.start),
      speaker: m.speaker, snippet: substring(coalesce(m.text,""), 0, $snippet_len),
      url: m.url, doc_id: m.doc_id,
      macro_id: ma.id, doc_title: ma.doc_title, doc_year: ma.doc_year,
      doc_url: ma.doc_url, youtube_id: ma.youtube_id
    })[0..$evidence_k] AS evidence_rel
  }
  CALL {
    WITH f
    UNWIND coalesce(f.evidence_micro_ids, []) AS mid
    OPTIONAL MATCH (m2:Micro {id: mid})
    OPTIONAL MATCH (m2)-[:IN_MACRO]->(ma2:Macro)
    WITH m2, ma2
    WHERE m2 IS NOT NULL
    ORDER BY m2.start ASC
    RETURN collect({
      micro_id: m2.id, start: m2.start, end: coalesce(m2.end, m2.start),
      speaker: m2.speaker, snippet: substring(coalesce(m2.text,""), 0, $snippet_len),
      url: m2.url, doc_id: m2.doc_id,
      macro_id: ma2.id, doc_title: ma2.doc_title, doc_year: ma2.doc_year,
      doc_url: ma2.doc_url, youtube_id: ma2.youtube_id
    })[0..$evidence_k] AS evidence_prop
  }
  RETURN
    anchor.id AS anchor_entity_id,
    anchor.canonical AS anchor_entity_name,
    f.id AS fact_id,
    coalesce(f.pred_norm, f.predicate) AS predicate,
    f.support_count AS support_count,
    anchor_role,
    sub.id AS subject_id,
    sub.canonical AS subject,
    obj.id AS object_id,
    obj.canonical AS object,
    obj.type_label AS object_type_label,
    obj.type AS object_type,
    other.id AS other_entity_id,
    other.canonical AS other_entity,
    other.type_label AS other_type_label,
    other.type AS other_type,
    evidence_rel,
    evidence_prop
  ORDER BY coalesce(f.support_count, 0) DESC, f.id DESC
  LIMIT $per_anchor_limit
}
RETURN *
"""

TOOL_ENTITY_RELATION_FACTS_EVIDENCE = """
WITH $entity_ids AS ids
MATCH (f:Fact)-[:HAS_SUBJECT]->(sub:Entity)
MATCH (f)-[:HAS_OBJECT]->(obj:Entity)
WHERE sub.id IN ids AND obj.id IN ids AND sub.id <> obj.id
WITH f, sub, obj
CALL {
  WITH f
  OPTIONAL MATCH (f)-[:EVIDENCED_BY]->(m:Micro)
  OPTIONAL MATCH (m)-[:IN_MACRO]->(ma:Macro)
  WITH m, ma
  WHERE m IS NOT NULL
  ORDER BY m.start ASC
  RETURN collect({
    micro_id: m.id, start: m.start, end: coalesce(m.end, m.start),
    speaker: m.speaker, snippet: substring(coalesce(m.text,""), 0, $snippet_len),
    url: m.url, doc_id: m.doc_id,
    macro_id: ma.id, doc_title: ma.doc_title, doc_year: ma.doc_year,
    doc_url: ma.doc_url, youtube_id: ma.youtube_id
  })[0..$evidence_k] AS evidence_rel
}
RETURN
  sub.id AS subject_id,
  sub.canonical AS subject,
  obj.id AS object_id,
  obj.canonical AS object,
  f.id AS fact_id,
  coalesce(f.pred_norm, f.predicate) AS predicate,
  f.support_count AS support_count,
  evidence_rel
ORDER BY coalesce(f.support_count,0) DESC, f.id DESC
LIMIT $limit
"""

TOOL_DATASET_YEAR_BOUNDS = """
MATCH (ma:Macro)
WHERE ma.doc_year IS NOT NULL
RETURN min(ma.doc_year) AS min_year, max(ma.doc_year) AS max_year
"""

TOOL_MICRO_FULLTEXT = """
CALL db.index.fulltext.queryNodes($index, $q) YIELD node, score
WHERE node:Micro
OPTIONAL MATCH (node)-[:IN_MACRO]->(ma:Macro)
RETURN
  node.id AS micro_id,
  node.doc_id AS doc_id,
  ma.id AS macro_id,
  node.start AS start,
  coalesce(node.end, node.start) AS end,
  node.speaker AS speaker,
  substring(coalesce(node.text,""), 0, $snippet_len) AS snippet,
  node.url AS url,
  ma.doc_title AS doc_title,
  ma.doc_year AS doc_year,
  ma.doc_url AS doc_url,
  ma.youtube_id AS youtube_id,
  score AS score
ORDER BY score DESC, node.start ASC
LIMIT $limit
"""

TOOL_MICRO_CONTAINS = """
MATCH (m:Micro)
WHERE
  toLower(coalesce(m.text, "")) CONTAINS toLower($q)
  OR toLower(coalesce(m.speaker, "")) CONTAINS toLower($q)
OPTIONAL MATCH (m)-[:IN_MACRO]->(ma:Macro)
RETURN
  m.id AS micro_id,
  m.doc_id AS doc_id,
  ma.id AS macro_id,
  m.start AS start,
  coalesce(m.end, m.start) AS end,
  m.speaker AS speaker,
  substring(coalesce(m.text,""), 0, $snippet_len) AS snippet,
  m.url AS url,
  ma.doc_title AS doc_title,
  ma.doc_year AS doc_year,
  ma.doc_url AS doc_url,
  ma.youtube_id AS youtube_id,
  0.0 AS score
ORDER BY coalesce(ma.doc_year, 9999) ASC, m.start ASC
LIMIT $limit
"""

TOOL_SPEAKER_MICROS = """
MATCH (s:Speaker {id:$speaker_id})-[:SPOKE]->(m:Micro)
OPTIONAL MATCH (m)-[:IN_MACRO]->(ma:Macro)
RETURN
  m.id AS micro_id,
  m.doc_id AS doc_id,
  ma.id AS macro_id,
  m.start AS start,
  coalesce(m.end, m.start) AS end,
  m.speaker AS speaker,
  substring(coalesce(m.text,""), 0, $snippet_len) AS snippet,
  m.url AS url,
  ma.doc_title AS doc_title,
  ma.doc_year AS doc_year,
  ma.doc_url AS doc_url,
  ma.youtube_id AS youtube_id
ORDER BY coalesce(ma.doc_year, 9999) ASC, m.start ASC
LIMIT $limit
"""

TOOL_SPEAKER_MENTIONED_ENTITIES = """
MATCH (s:Speaker {id:$speaker_id})-[:SPOKE]->(m:Micro)-[:IN_MACRO]->(ma:Macro)-[:MENTIONS]->(e:Entity)
RETURN
  e.id AS entity_id,
  e.canonical AS entity_name,
  e.type_label AS type_label,
  e.type AS type,
  count(*) AS mentions
ORDER BY mentions DESC, entity_name ASC
LIMIT $limit
"""

TOOL_SPEAKER_KEYWORD_MENTIONS = """
MATCH (s:Speaker {id:$speaker_id})-[:SPOKE]->(m:Micro)-[:IN_MACRO]->(ma:Macro)
WHERE any(t IN $terms WHERE toLower(coalesce(m.text,"")) CONTAINS toLower(t))
MATCH (ma)-[:MENTIONS]->(e:Entity)
WITH e,
     count(DISTINCT ma.id) AS mentions,
     collect(DISTINCT {
        micro_id: m.id,
        start: m.start,
        end: coalesce(m.end, m.start),
        speaker: m.speaker,
        snippet: substring(coalesce(m.text,""), 0, $snippet_len),
        url: m.url,
        doc_id: m.doc_id,
        macro_id: ma.id,
        doc_title: ma.doc_title,
        doc_year: ma.doc_year,
        doc_url: ma.doc_url,
        youtube_id: ma.youtube_id
     }) AS ev
WHERE e IS NOT NULL
  AND (
    $only_books = false
    OR toLower(coalesce(e.type_label,"")) CONTAINS "book"
    OR toLower(coalesce(e.type,"")) CONTAINS "book"
  )
RETURN
  e.id AS entity_id,
  e.canonical AS entity_name,
  e.type_label AS object_type_label,
  e.type AS object_type,
  mentions,
  ev[0..$evidence_k] AS evidence
ORDER BY mentions DESC, entity_name ASC
LIMIT $limit
"""

TOOL_MICRO_CONTEXT_AROUND = """
MATCH (m:Micro {id:$micro_id})-[:IN_MACRO]->(ma:Macro)
MATCH (mx:Micro)-[:IN_MACRO]->(ma)
WITH ma, m, mx
ORDER BY mx.start ASC, coalesce(mx.end, mx.start) ASC
WITH ma, m, collect(mx) AS ms
WITH ma, m, ms,
     [i IN range(0, size(ms)-1) WHERE ms[i].id = m.id][0] AS idx
WITH ma, m, ms, idx,
     CASE
        WHEN idx IS NULL THEN 0
        WHEN idx - $window < 0 THEN 0
        ELSE idx - $window
     END AS lo,
     CASE
        WHEN idx IS NULL THEN size(ms)-1
        WHEN idx + $window >= size(ms) THEN size(ms)-1
        ELSE idx + $window
     END AS hi
WITH ma, m, ms[lo..hi+1] AS ctx
UNWIND ctx AS c
RETURN
  ma.id AS macro_id,
  ma.doc_id AS doc_id,
  ma.doc_title AS doc_title,
  ma.doc_year AS doc_year,
  ma.doc_url AS doc_url,
  ma.youtube_id AS youtube_id,
  c.id AS micro_id,
  c.start AS start,
  coalesce(c.end, c.start) AS end,
  c.speaker AS speaker,
  substring(coalesce(c.text,""), 0, $snippet_len) AS snippet,
  c.url AS url
ORDER BY start ASC
"""


def merge_evidence(rel_list: Any, prop_list: Any, cap: int = 3) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for src in [rel_list, prop_list]:
        if not isinstance(src, list):
            continue
        for e in src:
            if not isinstance(e, dict):
                continue
            mid = e.get("micro_id")
            if not mid or mid in seen:
                continue
            seen.add(mid)
            out.append(e)
            if len(out) >= cap:
                return out
    return out


def has_micro_evidence(ev: Any) -> bool:
    if not isinstance(ev, list) or not ev:
        return False
    for e in ev:
        if isinstance(e, dict) and e.get("micro_id"):
            return True
    return False


def evidence_text(ev: Any) -> str:
    if not isinstance(ev, list):
        return ""
    parts = []
    for e in ev:
        if isinstance(e, dict):
            parts.append(normalize_str(e.get("snippet")))
    return " ".join([p for p in parts if p])


def format_micro_evidence(ev_list: Any, max_items: int = 3) -> str:
    if not isinstance(ev_list, list) or not ev_list:
        return "(no micro evidence)"
    lines = []
    n = 0
    for e in ev_list:
        if n >= max_items:
            break
        if not isinstance(e, dict) or not e.get("micro_id"):
            continue
        n += 1
        mid = e.get("micro_id")
        start = e.get("start")
        end = e.get("end")
        sp = e.get("speaker")
        sn = normalize_str(e.get("snippet"))
        url = normalize_str(e.get("url"))
        doc_title = normalize_str(e.get("doc_title"))
        doc_year = e.get("doc_year")
        macro_id = normalize_str(e.get("macro_id"))

        header = f"- micro_id={mid} [{start}-{end}] speaker={sp}"
        meta = " | ".join([str(x) for x in [doc_year, doc_title] if x])
        if macro_id:
            meta = (meta + " | " if meta else "") + f"macro_id={macro_id}"
        if meta:
            header += f" | {meta}"
        if sn:
            header += f': "{sn}"'
        if url:
            header += f" | {url}"
        lines.append(header)
    return "\n".join(lines) if lines else "(no micro evidence)"


def fetch_micro_context(
    driver: neo4j.Driver,
    db: Optional[str],
    micro_id: str,
    window: int,
    snippet_len: int,
    debug_lines: List[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    df = try_read_df(
        driver,
        db,
        TOOL_MICRO_CONTEXT_AROUND,
        params={"micro_id": micro_id, "window": int(window), "snippet_len": int(snippet_len)},
        debug_lines=debug_lines,
    )
    if df.empty:
        return {}, []
    rows = df.to_dict(orient="records")
    ma = {
        "macro_id": rows[0].get("macro_id"),
        "doc_id": rows[0].get("doc_id"),
        "doc_title": rows[0].get("doc_title"),
        "doc_year": rows[0].get("doc_year"),
        "doc_url": rows[0].get("doc_url"),
        "youtube_id": rows[0].get("youtube_id"),
    }
    micros = [
        {
            "micro_id": r.get("micro_id"),
            "start": r.get("start"),
            "end": r.get("end"),
            "speaker": r.get("speaker"),
            "snippet": r.get("snippet"),
            "url": r.get("url"),
        }
        for r in rows
    ]
    return ma, micros


def hard_filter_year_show(df: pd.DataFrame, year: Optional[int], show_query: Optional[str], strict: bool) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()

    if strict and year is not None:
        y = int(year)

        def ok_year(row) -> bool:
            ev = row.get("evidence")
            if isinstance(ev, list) and ev:
                for e in ev:
                    if safe_int(e.get("doc_year")) == y:
                        return True
                    if str(y) in normalize_str(e.get("doc_id")):
                        return True
            if safe_int(row.get("doc_year")) == y:
                return True
            if str(y) in normalize_str(row.get("doc_id")):
                return True
            return False

        out = out[out.apply(lambda rr: ok_year(rr.to_dict()), axis=1)].copy()

    if strict and show_query:
        sq = show_query.lower().strip()

        def ok_show(row) -> bool:
            ev = row.get("evidence")
            if isinstance(ev, list) and ev:
                for e in ev:
                    if sq in normalize_str(e.get("doc_title")).lower():
                        return True
                    if sq in normalize_str(e.get("doc_id")).lower():
                        return True
            if sq in normalize_str(row.get("doc_title")).lower():
                return True
            if sq in normalize_str(row.get("doc_id")).lower():
                return True
            return False

        out = out[out.apply(lambda rr: ok_show(rr.to_dict()), axis=1)].copy()

    return out


def hard_exclude(df: pd.DataFrame, exclude_terms: List[str], strict: bool) -> pd.DataFrame:
    if df is None or df.empty or not exclude_terms:
        return df
    ex = [t.lower().strip() for t in exclude_terms if t and t.strip()]
    if not ex:
        return df
    out = df.copy()

    def bad(row) -> bool:
        txt = " ".join(
            [
                normalize_str(row.get("subject")).lower(),
                normalize_str(row.get("object")).lower(),
                normalize_str(row.get("other_entity")).lower(),
                normalize_str(row.get("entity_name")).lower(),
                normalize_str(row.get("anchor_entity_name")).lower(),
                normalize_str(row.get("title")).lower(),
                evidence_text(row.get("evidence")).lower(),
                normalize_str(row.get("snippet")).lower(),
            ]
        )
        return any(t in txt for t in ex)

    return out[~out.apply(lambda rr: bad(rr.to_dict()), axis=1)].copy() if strict else out


def keyword_micros(
    driver: neo4j.Driver,
    db: Optional[str],
    query: str,
    micro_ft_indexes: List[str],
    snippet_len: int,
    limit: int,
    debug_lines: List[str],
) -> pd.DataFrame:
    q = query.strip()
    if not q:
        return pd.DataFrame()

    if micro_ft_indexes:
        ix = micro_ft_indexes[0]
        df = try_read_df(
            driver,
            db,
            TOOL_MICRO_FULLTEXT,
            params={"index": ix, "q": q, "snippet_len": int(snippet_len), "limit": int(limit)},
            debug_lines=debug_lines,
        )
        if not df.empty:
            return df

    return try_read_df(
        driver,
        db,
        TOOL_MICRO_CONTAINS,
        params={"q": q, "snippet_len": int(snippet_len), "limit": int(limit)},
        debug_lines=debug_lines,
    )


ALLOWED_ACTIONS = [
    "keyword_micros",
    "first_mention",
    "term_disambiguate",
    "entity_facts",
    "speaker_micros",
    "speaker_mentions",
    "generic_text2cypher",
]

PLANNER_SYSTEM = f"""
Return ONLY JSON (no markdown). Required keys:
- action: one of {ALLOWED_ACTIONS}
- speaker_query: string|null
- entity_query: string|null
- entity_queries: list[str]
- entity_mode: "SINGLE"|"MULTI_UNION"|"MULTI_RELATION"
- keyword_query: string|null
- year: int|null
- show_query: string|null
- exclude_terms: list[str]
- require_evidence: bool

Planning rules:
- Default action=entity_facts for most questions.
- If question mentions 2+ distinct entities:
  - Fill entity_queries with them.
  - If user asks "between X and Y", "relationship", "connected", "X vs Y": entity_mode="MULTI_RELATION"
  - Else entity_mode="MULTI_UNION"
- If only one entity: entity_mode="SINGLE"
- Even if user asks "recommended by X": action=entity_facts with speaker_query="X" and/or entity_query="X".
  (Do NOT output any special fallback action.)
- If user asks exact mention/timestamp/quote/clip of a term -> action=keyword_micros and keyword_query must be the term/phrase to search.
- If user asks first mention/patient zero -> action=first_mention and keyword_query must be the term/phrase.
- If user asks to distinguish meanings -> action=term_disambiguate and keyword_query must be the ambiguous term.
- If user asks counts/aggregation like "how many speakers", "number of facts", "top 10 predicates", "schema-ish" -> action=generic_text2cypher.
- If the question includes a year (e.g. 2021) set year=2021.
- If the question includes a show/podcast constraint, set show_query to the show name (short).
- If the question includes exclusion constraints, fill exclude_terms with concrete strings.
- require_evidence must be true when uat_strict is true OR when user asks for evidence/quotes/timestamps OR when any constraint (year/show/exclude) exists.
- If action is keyword_micros/first_mention/term_disambiguate, keyword_query MUST NOT be null: choose the best search phrase (prefer quoted phrases; else a concise term; <= 80 chars).
""".strip()

KEYWORD_EXTRACT_SYSTEM = """
Return ONLY JSON (no markdown):
{"keyword_query":"..."}

Task:
Given the user's question, extract the best short search phrase for finding matching Micro text.
Rules:
- Prefer quoted text if present.
- Otherwise pick the most specific term/name/topic in the question.
- Keep it concise (<= 80 chars).
- Do NOT include extra commentary.
""".strip()


def parse_json_loose(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I).strip()
    t = re.sub(r"\s*```$", "", t).strip()
    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def minimal_plan(question: str, uat_mode: bool) -> Dict[str, Any]:
    return {
        "action": "entity_facts",
        "speaker_query": None,
        "entity_query": None,
        "entity_queries": [],
        "entity_mode": "SINGLE",
        "keyword_query": None,
        "year": None,
        "show_query": None,
        "exclude_terms": [],
        "require_evidence": bool(uat_mode),
    }


def llm_extract_keyword_query(client: Any, model: str, question: str, debug_lines: List[str]) -> Optional[str]:
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": KEYWORD_EXTRACT_SYSTEM},
                {"role": "user", "content": json.dumps({"question": question}, ensure_ascii=False)},
            ],
        )
        txt = (resp.choices[0].message.content or "").strip()
        obj = parse_json_loose(txt) or {}
        kq = normalize_str(obj.get("keyword_query"))
        return kq if kq else None
    except Exception as e:
        debug_log(debug_lines, f"[llm_extract_keyword_query] failed: {e}")
        return None


GENERIC_ANCHOR_QUERIES = {
    "book",
    "books",
    "movie",
    "movies",
    "podcast",
    "podcasts",
    "show",
    "episode",
    "episodes",
    "recommendation",
    "recommendations",
}


def llm_plan(
    client: Any,
    model: str,
    question: str,
    schema_hint: str,
    caps: Dict[str, Any],
    uat_mode: bool,
    debug_lines: List[str],
) -> Dict[str, Any]:
    if client is None:
        return minimal_plan(question, uat_mode)

    try:
        payload = {
            "schema_hint": schema_hint,
            "capabilities": {
                "labels": caps.get("labels", [])[:60],
                "relationship_types": caps.get("relationship_types", [])[:60],
            },
            "uat_strict": uat_mode,
            "question": question,
        }
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        txt = (resp.choices[0].message.content or "").strip()
        plan = parse_json_loose(txt) or minimal_plan(question, uat_mode)

        if plan.get("action") not in ALLOWED_ACTIONS:
            plan["action"] = "entity_facts"

        plan["year"] = safe_int(plan.get("year"))
        if not isinstance(plan.get("exclude_terms"), list):
            plan["exclude_terms"] = []
        plan["require_evidence"] = bool(plan.get("require_evidence", uat_mode))

        plan.setdefault("speaker_query", None)
        plan.setdefault("entity_query", None)
        plan.setdefault("entity_queries", [])
        plan.setdefault("entity_mode", "SINGLE")
        plan.setdefault("keyword_query", None)
        plan.setdefault("show_query", None)

        eqs = plan.get("entity_queries")
        if not isinstance(eqs, list):
            eqs = []
        eqs = [normalize_str(x) for x in eqs if normalize_str(x)]
        plan["entity_queries"] = eqs

        if normalize_str(plan.get("entity_query")) and not plan["entity_queries"]:
            plan["entity_queries"] = [normalize_str(plan.get("entity_query"))]

        if plan.get("action") == "entity_facts" and normalize_str(plan.get("speaker_query")):
            eq = to_lower(plan.get("entity_query"))
            if (not eq) or (eq in GENERIC_ANCHOR_QUERIES):
                plan["entity_query"] = plan.get("speaker_query")
                if not plan["entity_queries"]:
                    plan["entity_queries"] = [normalize_str(plan.get("speaker_query"))]

        mode = normalize_str(plan.get("entity_mode")).upper()
        if mode not in ["SINGLE", "MULTI_UNION", "MULTI_RELATION"]:
            mode = "SINGLE" if len(plan["entity_queries"]) <= 1 else "MULTI_UNION"
        plan["entity_mode"] = mode

        if plan.get("action") in ["keyword_micros", "first_mention", "term_disambiguate"]:
            if not normalize_str(plan.get("keyword_query")):
                kq = llm_extract_keyword_query(client, model, question, debug_lines)
                if kq:
                    plan["keyword_query"] = kq

        return plan
    except Exception as e:
        debug_log(debug_lines, f"[llm_plan] failed: {e}")
        return minimal_plan(question, uat_mode)


ENTITY_SELECT_SYSTEM = """
Return ONLY JSON (no markdown):
{"selected_entity_ids":["..."]}

You are selecting which Entity candidates best match the user's question and the mention queries.
Use ONLY the provided candidates.
Select up to max_select.
Prefer name/alias matches, type consistency, and candidates mentioned by multiple sources.
If nothing is relevant, return {"selected_entity_ids": []}.
""".strip()

FACT_SELECT_SYSTEM = """
Return ONLY JSON (no markdown):
{"selected_fact_ids":["..."]}

You are selecting which Facts best answer the user's question.
Use ONLY the provided candidate facts and their Micro evidence snippets.

Selection principles:
- Prefer candidates that directly answer the question intent (e.g., recommendations, preferences, claims).
- Prefer candidates whose evidence snippet explicitly supports the claim.
- If the question is about "books", prioritize objects with type_label/type containing "book".
- If multiple duplicates exist, pick the most specific/explicit ones.
- Select up to max_select.
- If nothing is relevant, return {"selected_fact_ids":[]}.
""".strip()

MICRO_SELECT_SYSTEM = """
Return ONLY JSON (no markdown):
{"selected_micro_ids":["..."]}

Select Micro snippets that best answer the user's question.
Use ONLY the provided candidates.

Rules:
- Prefer snippets with explicit wording matching the question intent.
- If nothing is relevant, return {"selected_micro_ids":[]}.
""".strip()

MAX_ENTITY_SELECT = 5
MAX_FACT_SELECT = 5


def llm_select_relevant_entity_ids(
    client: Any,
    model: str,
    question: str,
    candidates: List[Dict[str, Any]],
    max_select: int = MAX_ENTITY_SELECT,
    max_rows: int = 60,
    debug_lines: Optional[List[str]] = None,
) -> List[str]:
    if client is None or not candidates:
        return []

    packed = []
    for r in candidates[:max_rows]:
        eid = normalize_str(r.get("entity_id"))
        if not eid:
            continue
        score = r.get("best_score")
        try:
            score = float(score) if score is not None else None
        except Exception:
            score = None
        packed.append(
            {
                "entity_id": eid,
                "entity_name": normalize_str(r.get("entity_name")),
                "canonical_key": normalize_str(r.get("canonical_key")),
                "type_label": normalize_str(r.get("type_label")),
                "type": normalize_str(r.get("type")),
                "aliases_json": r.get("aliases_json"),
                "best_score": score,
                "sources": r.get("sources"),
                "mention_queries": r.get("mention_queries"),
            }
        )

    if not packed:
        return []

    try:
        payload = {"question": question, "max_select": int(max_select), "candidates": packed}
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": ENTITY_SELECT_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
            ],
        )
        txt = (resp.choices[0].message.content or "").strip()
        obj = parse_json_loose(txt) or {}
        ids = obj.get("selected_entity_ids", [])
        if not isinstance(ids, list):
            return []

        allowed = {str(x.get("entity_id")) for x in packed if x.get("entity_id")}
        out: List[str] = []
        for x in ids:
            sx = str(x)
            if sx in allowed and sx not in out:
                out.append(sx)
            if len(out) >= int(max_select):
                break
        return out
    except Exception as e:
        if debug_lines is not None:
            debug_log(debug_lines, f"[llm_select_relevant_entity_ids] failed: {e}")
        return []


def llm_select_relevant_fact_ids(
    client: Any,
    model: str,
    question: str,
    anchor_name: str,
    rows: List[Dict[str, Any]],
    max_rows: int = 180,
    max_evidence_each: int = 2,
    max_select: int = MAX_FACT_SELECT,
    debug_lines: Optional[List[str]] = None,
) -> List[str]:
    if client is None or not rows:
        return []

    packed = []
    for r in rows[:max_rows]:
        ev = r.get("evidence") or []
        if not isinstance(ev, list) or not ev:
            continue
        evidence_clean = []
        for e in ev:
            if isinstance(e, dict) and e.get("micro_id"):
                evidence_clean.append(
                    {
                        "micro_id": e.get("micro_id"),
                        "start": e.get("start"),
                        "end": e.get("end"),
                        "speaker": e.get("speaker"),
                        "snippet": e.get("snippet"),
                    }
                )
        if not evidence_clean:
            continue

        packed.append(
            {
                "fact_id": r.get("fact_id"),
                "predicate": r.get("predicate"),
                "anchor_role": r.get("anchor_role"),
                "anchor_entity_name": r.get("anchor_entity_name"),
                "anchor_entity_id": r.get("anchor_entity_id"),
                "subject_id": r.get("subject_id"),
                "subject": r.get("subject"),
                "object_id": r.get("object_id"),
                "object": r.get("object"),
                "other_entity_id": r.get("other_entity_id"),
                "other_entity": r.get("other_entity"),
                "support_count": r.get("support_count"),
                "object_type_label": r.get("object_type_label"),
                "object_type": r.get("object_type"),
                "other_type_label": r.get("other_type_label"),
                "other_type": r.get("other_type"),
                "evidence": evidence_clean[:max_evidence_each],
            }
        )

    if not packed:
        return []

    try:
        payload = {"question": question, "anchor": anchor_name, "max_select": int(max_select), "candidates": packed}
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": FACT_SELECT_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
            ],
        )
        txt = (resp.choices[0].message.content or "").strip()
        obj = parse_json_loose(txt) or {}
        ids = obj.get("selected_fact_ids", [])
        if not isinstance(ids, list):
            return []

        allowed = {str(x.get("fact_id")) for x in packed if x.get("fact_id")}
        out: List[str] = []
        for x in ids:
            sx = str(x)
            if sx in allowed and sx not in out:
                out.append(sx)
            if len(out) >= int(max_select):
                break
        return out
    except Exception as e:
        if debug_lines is not None:
            debug_log(debug_lines, f"[llm_select_relevant_fact_ids] failed: {e}")
        return []


def llm_select_relevant_micro_ids(
    client: Any,
    model: str,
    question: str,
    rows: List[Dict[str, Any]],
    max_rows: int = 220,
    debug_lines: Optional[List[str]] = None,
) -> List[str]:
    if client is None or not rows:
        return []

    packed = []
    for r in rows[:max_rows]:
        mid = r.get("micro_id")
        if not mid:
            continue
        packed.append(
            {
                "micro_id": mid,
                "start": r.get("start"),
                "end": r.get("end"),
                "speaker": r.get("speaker"),
                "snippet": r.get("snippet"),
                "doc_title": r.get("doc_title"),
                "doc_year": r.get("doc_year"),
            }
        )

    if not packed:
        return []

    try:
        payload = {"question": question, "candidates": packed}
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": MICRO_SELECT_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        txt = (resp.choices[0].message.content or "").strip()
        obj = parse_json_loose(txt) or {}
        ids = obj.get("selected_micro_ids", [])
        if not isinstance(ids, list):
            return []

        allowed = {str(x.get("micro_id")) for x in packed if x.get("micro_id")}
        out: List[str] = []
        for x in ids:
            sx = str(x)
            if sx in allowed and sx not in out:
                out.append(sx)
        return out
    except Exception as e:
        if debug_lines is not None:
            debug_log(debug_lines, f"[llm_select_relevant_micro_ids] failed: {e}")
        return []


DISAMBIG_SYSTEM = """
Return ONLY JSON (no markdown):
{
  "sense_catalog": ["SENSE_1_NAME","SENSE_2_NAME"],
  "labels":[{"micro_id":"...","sense_name":"SENSE_1_NAME"|"SENSE_2_NAME"|"UNKNOWN"}]
}

Rules:
- Pick two dominant meanings from snippets and name them clearly (e.g., MOVIE, BOOK; or PERSON, COMPANY).
- Use only those exact names in labels.
- If a snippet does not match either, label UNKNOWN.
""".strip()


def llm_disambiguate(
    client: Any,
    model: str,
    term: str,
    rows: List[Dict[str, Any]],
    debug_lines: List[str],
) -> Tuple[List[str], Dict[str, str]]:
    if client is None or not rows:
        return ([], {})
    payload = [{"micro_id": r.get("micro_id"), "snippet": r.get("snippet")} for r in rows[:140]]
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": DISAMBIG_SYSTEM},
                {"role": "user", "content": json.dumps({"term": term, "snippets": payload}, ensure_ascii=False)},
            ],
        )
        txt = (resp.choices[0].message.content or "").strip()
        obj = parse_json_loose(txt) or {}

        sense_catalog = []
        if isinstance(obj, dict) and isinstance(obj.get("sense_catalog"), list):
            sense_catalog = [normalize_str(x) for x in obj["sense_catalog"] if normalize_str(x)]

        out: Dict[str, str] = {}
        labs = obj.get("labels", []) if isinstance(obj, dict) else []
        if isinstance(labs, list):
            for it in labs:
                if isinstance(it, dict) and it.get("micro_id") and it.get("sense_name"):
                    out[str(it["micro_id"])] = normalize_str(it["sense_name"]) or "UNKNOWN"

        return (sense_catalog, out)
    except Exception as e:
        debug_log(debug_lines, f"[llm_disambiguate] failed: {e}")
        return ([], {})


ANSWER_SYSTEM = """
You are an evidence-first assistant for a podcast/video knowledge graph.
Use ONLY the provided evidence items.
Every claim must be backed by at least one micro evidence (micro_id + timestamp + speaker + snippet).
If evidence is insufficient, say so clearly.

Output:
- A short answer
- Then bullet points with citations per item (micro_id, start-end, speaker, doc_year/doc_title, url if present)
Do NOT invent facts or context.
""".strip()


def build_answer_prompt(question: str, items: List[Dict[str, Any]]) -> str:
    return json.dumps(
        {"language": "en", "question": question, "evidence_items": items},
        ensure_ascii=False,
        indent=2,
    )


def llm_answer(client: Any, model: str, question: str, evidence_items: List[Dict[str, Any]], debug_lines: List[str]) -> Optional[str]:
    if client is None or not evidence_items:
        return None
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": build_answer_prompt(question, evidence_items)},
            ],
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception as e:
        debug_log(debug_lines, f"[llm_answer] failed: {e}")
        return None


def pack_micro_evidence_for_answer(
    df: pd.DataFrame,
    action: str,
    max_items: int = 12,
    max_micros_each: int = 3,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if df is None or df.empty:
        return items

    if action == "keyword_mentions_list":
        for _, r in df.head(max_items).iterrows():
            ev = r.get("evidence")
            if not isinstance(ev, list) or not ev:
                continue
            items.append(
                {
                    "type": "keyword_mentions",
                    "entity_name": normalize_str(r.get("entity_name")),
                    "object_type": normalize_str(r.get("object_type_label") or r.get("object_type")),
                    "evidence": [e for e in ev if isinstance(e, dict) and e.get("micro_id")][:max_micros_each],
                }
            )
        return items

    if action in ["keyword_micros", "first_mention", "speaker_micros"]:
        for _, r in df.head(max_items).iterrows():
            items.append(
                {
                    "type": "micro",
                    "micro_id": r.get("micro_id"),
                    "start": r.get("start"),
                    "end": r.get("end"),
                    "speaker": r.get("speaker"),
                    "snippet": r.get("snippet"),
                    "url": r.get("url"),
                    "doc_id": r.get("doc_id"),
                    "macro_id": r.get("macro_id"),
                    "doc_title": r.get("doc_title"),
                    "doc_year": r.get("doc_year"),
                }
            )
        return items

    for _, r in df.head(max_items).iterrows():
        ev = r.get("evidence")
        if not isinstance(ev, list) or not ev:
            continue
        ev_clean = [e for e in ev if isinstance(e, dict) and e.get("micro_id")]
        if not ev_clean:
            continue
        items.append(
            {
                "type": "fact",
                "anchor": normalize_str(r.get("anchor_entity_name")),
                "predicate": normalize_str(r.get("predicate")),
                "subject": normalize_str(r.get("subject")),
                "object": normalize_str(r.get("object")),
                "other_entity": normalize_str(r.get("other_entity")),
                "anchor_role": normalize_str(r.get("anchor_role")),
                "support_count": r.get("support_count"),
                "evidence": ev_clean[:max_micros_each],
            }
        )
    return items


def enforce_readonly_or_raise(cypher: str):
    if not cypher:
        raise RuntimeError("No cypher generated.")
    if FORBIDDEN_CYPHER.search(cypher):
        raise RuntimeError("Generated Cypher includes write/admin keywords. Refusing to execute.")


def ensure_limit(cypher: str, limit: int) -> str:
    c = cypher.strip().rstrip(";")
    if re.search(r"(?is)\bLIMIT\b", c):
        return c
    if re.search(r"(?is)\bRETURN\b\s+count\s*\(", c):
        return c
    return f"{c}\nLIMIT {int(limit)}"


T2C_SYSTEM = """
Return ONLY a Cypher statement (no markdown, no explanations).

STRICT:
- READ-ONLY ONLY. No CREATE/MERGE/DELETE/SET/DROP/LOAD CSV.
- Always include a LIMIT unless returning only an aggregation.

FULLTEXT RULE (CRITICAL):
- You may use: CALL db.index.fulltext.queryNodes(indexName, query)
  ONLY IF indexName is present in the user payload field `fulltext_indexes`.
- Use a literal index name string (e.g. "ft_entity_search"), NOT a parameter like $index.
- If `fulltext_indexes` lists are empty, DO NOT call fulltext procedures. Use CONTAINS-based MATCH instead.
- Never attempt to discover indexes (no SHOW INDEXES / SHOW FULLTEXT INDEXES / CALL db.indexes()).

YEAR FILTER RULE (CRITICAL):
- Micro nodes do NOT have doc_year.
- If filtering by year, use (m:Micro)-[:IN_MACRO]->(ma:Macro) and filter on ma.doc_year.

When returning Micro evidence fields (preferred when relevant):
  micro_id, start, end, speaker, snippet (substring), url, doc_id, doc_title, doc_year.
""".strip()


def build_t2c_user(schema_text: str, question: str, fulltext_indexes: Dict[str, List[str]]) -> str:
    payload = {
        "schema": compact_schema(schema_text, 140),
        "question": question,
        "fulltext_indexes": [
            str(x) for x in (fulltext_indexes.get("entity", []) + fulltext_indexes.get("micro", []))
            if str(x).strip()
        ],
        "fulltext_indexes_by_type": {
            "entity": list(fulltext_indexes.get("entity", []) or []),
            "micro": list(fulltext_indexes.get("micro", []) or []),
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def llm_generate_cypher(
    client: Any,
    model: str,
    schema_text: str,
    question: str,
    fulltext_indexes: Dict[str, List[str]],
) -> str:
    if client is None:
        raise RuntimeError("LLM client unavailable.")
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": T2C_SYSTEM},
            {"role": "user", "content": build_t2c_user(schema_text, question, fulltext_indexes)},
        ],
    )
    cy = (resp.choices[0].message.content or "").strip()
    cy = re.sub(r"^```(?:cypher)?\s*", "", cy, flags=re.I).strip()
    cy = re.sub(r"\s*```$", "", cy).strip()
    return cy


def resolve_speakers(driver: neo4j.Driver, db: Optional[str], frag: str, top_k: int, debug_lines: List[str]) -> pd.DataFrame:
    return try_read_df(
        driver,
        db,
        TOOL_SPEAKER_CANDIDATES,
        params={"frag": frag, "k": int(top_k)},
        debug_lines=debug_lines,
    )


def resolve_entities(
    driver: neo4j.Driver,
    db: Optional[str],
    q: str,
    top_k: int,
    entity_ft_indexes: List[str],
    debug_lines: List[str],
) -> pd.DataFrame:
    frames = []

    try:
        df_sr = try_read_df(
            driver,
            db,
            TOOL_SPEAKER_RESOLVES_TO_ENTITY,
            params={"frag": q, "k": int(top_k)},
            debug_lines=debug_lines,
        )
        if not df_sr.empty:
            df_sr2 = df_sr.dropna(subset=["entity_id"]).copy()
            if not df_sr2.empty:
                if "aliases_json" not in df_sr2.columns:
                    df_sr2["aliases_json"] = None
                df_sr2["source"] = "speaker_resolves_to"
                df_sr2["score"] = 10.0
                frames.append(
                    df_sr2[
                        [
                            "entity_id",
                            "entity_name",
                            "canonical_key",
                            "type_label",
                            "type",
                            "aliases_json",
                            "score",
                            "source",
                        ]
                    ]
                )
    except Exception:
        pass

    for ix in (entity_ft_indexes[:2] if entity_ft_indexes else []):
        try:
            df_ft = try_read_df(
                driver,
                db,
                TOOL_ENTITY_FULLTEXT,
                params={"index": ix, "q": q, "k": int(top_k)},
                debug_lines=debug_lines,
            )
            if not df_ft.empty:
                df_ft["source"] = f"fulltext:{ix}"
                frames.append(
                    df_ft[
                        [
                            "entity_id",
                            "entity_name",
                            "canonical_key",
                            "type_label",
                            "type",
                            "aliases_json",
                            "score",
                            "source",
                        ]
                    ]
                )
        except Exception:
            continue

    try:
        df_cb = try_read_df(
            driver,
            db,
            TOOL_ENTITY_CONTAINS,
            params={"q": q, "k": int(top_k)},
            debug_lines=debug_lines,
        )
        if not df_cb.empty:
            df_cb["source"] = "contains"
            frames.append(
                df_cb[
                    [
                        "entity_id",
                        "entity_name",
                        "canonical_key",
                        "type_label",
                        "type",
                        "aliases_json",
                        "score",
                        "source",
                    ]
                ]
            )
    except Exception:
        pass

    if not frames:
        return pd.DataFrame(
            columns=["entity_id", "entity_name", "canonical_key", "type_label", "type", "aliases_json", "score", "source"]
        )

    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["entity_id"], keep="first")
    out = out.sort_values("score", ascending=False)
    return out


def graph_prepare(question: str, settings: GraphSettings, debug_lines: List[str]) -> GraphPrep:
    if not (NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD):
        raise RuntimeError("Neo4j connection info missing (URI/username/password).")

    llm_entity_selector = True
    llm_needed = (
        settings.planner_enabled
        or settings.llm_row_selector
        or llm_entity_selector
        or settings.generate_answer
        or settings.enable_safe_text2cypher
    )
    if llm_needed and not OPENROUTER_API_KEY.strip():
        raise RuntimeError("OPENROUTER_API_KEY is missing.")
    if llm_needed and not OPENROUTER_BASE_URL.strip():
        raise RuntimeError("OPENROUTER_BASE_URL is missing.")
    if llm_needed and not OPENROUTER_MODEL.strip():
        raise RuntimeError("Model is empty (OPENROUTER_MODEL).")

    if not question.strip():
        raise RuntimeError("Query is empty.")

    debug_log(debug_lines, "=== RUN START ===")
    debug_log(debug_lines, f"User query: {question}")
    debug_log(debug_lines, f"UAT strict: {settings.uat_mode}")

    driver = get_driver(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    run_read_records(driver, NEO4J_DATABASE or None, "RETURN 1 AS ok")

    if settings.use_manual_schema:
        if not settings.schema_text.strip():
            raise RuntimeError("Manual schema mode ON but schema is empty.")
        effective_schema = settings.schema_text.strip()
    else:
        if settings.schema_text.strip():
            effective_schema = settings.schema_text.strip()
        else:
            effective_schema = try_fetch_schema(driver, NEO4J_DATABASE or None)

    schema_hint = compact_schema(effective_schema)

    caps = get_capabilities(driver, NEO4J_DATABASE or None)
    idx_df = detect_indexes(driver, NEO4J_DATABASE or None, debug_lines)
    entity_ft_indexes = pick_fulltext_index(idx_df, "Entity", ["canonical", "canonical_key", "aliases_json"])
    micro_ft_indexes = pick_fulltext_index(idx_df, "Micro", ["text"])
    debug_log(debug_lines, f"Entity FULLTEXT indexes: {entity_ft_indexes}")
    debug_log(debug_lines, f"Micro FULLTEXT indexes: {micro_ft_indexes}")

    min_year = max_year = None
    try:
        b = run_read_records(driver, NEO4J_DATABASE or None, TOOL_DATASET_YEAR_BOUNDS)
        if b:
            min_year = safe_int(b[0].get("min_year"))
            max_year = safe_int(b[0].get("max_year"))
    except Exception:
        pass

    client = build_openrouter_client(OPENROUTER_API_KEY, OPENROUTER_BASE_URL)

    plan = llm_plan(client, OPENROUTER_MODEL, question, schema_hint, caps, settings.uat_mode, debug_lines) if settings.planner_enabled else minimal_plan(question, settings.uat_mode)

    if plan.get("action") in ["recommendations_list", "keyword_mentions_list"]:
        plan["action"] = "entity_facts"
        if not normalize_str(plan.get("entity_query")) and normalize_str(plan.get("speaker_query")):
            plan["entity_query"] = plan.get("speaker_query")
        if not isinstance(plan.get("entity_queries"), list) or not plan["entity_queries"]:
            plan["entity_queries"] = [normalize_str(plan.get("entity_query") or plan.get("speaker_query"))]
        if normalize_str(plan.get("entity_mode")).upper() not in ["SINGLE", "MULTI_UNION", "MULTI_RELATION"]:
            plan["entity_mode"] = "SINGLE"

    req_year = plan.get("year")
    if settings.uat_mode and req_year is not None and min_year is not None and max_year is not None:
        if int(req_year) < int(min_year) or int(req_year) > int(max_year):
            raise RuntimeError(f"Requested year {req_year} is outside dataset bounds ({min_year}-{max_year}).")

    plan_entity_queries = plan.get("entity_queries")
    if not isinstance(plan_entity_queries, list):
        plan_entity_queries = []
    plan_entity_queries = [normalize_str(x) for x in plan_entity_queries if normalize_str(x)]

    single_entity_q = normalize_str(plan.get("entity_query"))
    if not plan_entity_queries and single_entity_q:
        plan_entity_queries = [single_entity_q]

    speaker_q = normalize_str(plan.get("speaker_query"))
    sp_df = resolve_speakers(driver, NEO4J_DATABASE or None, speaker_q, settings.top_candidates, debug_lines) if speaker_q else pd.DataFrame()

    entity_frames: List[pd.DataFrame] = []
    per_query_top: Dict[str, Optional[str]] = {}
    for q in plan_entity_queries[: int(settings.max_anchors)]:
        dfx = resolve_entities(driver, NEO4J_DATABASE or None, q, settings.top_candidates, entity_ft_indexes, debug_lines)
        if dfx is None or dfx.empty:
            per_query_top[q] = None
            continue
        dfx2 = dfx.copy()
        dfx2["mention_query"] = q
        entity_frames.append(dfx2)
        top_id = normalize_str(dfx2.iloc[0].get("entity_id"))
        per_query_top[q] = top_id if top_id else None

    if entity_frames:
        en_multi = pd.concat(entity_frames, ignore_index=True)
    else:
        en_multi = pd.DataFrame(
            columns=["entity_id", "entity_name", "canonical_key", "type_label", "type", "aliases_json", "score", "source", "mention_query"]
        )

    if not en_multi.empty:
        d = en_multi.copy()
        d["score_num"] = pd.to_numeric(d.get("score"), errors="coerce").fillna(0)
        agg = (
            d.sort_values(["score_num"], ascending=False)
            .groupby(["entity_id"], dropna=False)
            .agg(
                entity_name=("entity_name", "first"),
                canonical_key=("canonical_key", "first"),
                type_label=("type_label", "first"),
                type=("type", "first"),
                aliases_json=("aliases_json", lambda x: sorted(list({normalize_str(v) for v in x if normalize_str(v)}))),
                best_score=("score_num", "max"),
                sources=("source", lambda x: sorted(list({str(v) for v in x if str(v)}))),
                mention_queries=("mention_query", lambda x: sorted(list({str(v) for v in x if str(v)}))),
            )
            .reset_index()
            .sort_values(["best_score", "entity_name"], ascending=[False, True])
        )
    else:
        agg = pd.DataFrame(
            columns=["entity_id", "entity_name", "canonical_key", "type_label", "type", "aliases_json", "best_score", "sources", "mention_queries"]
        )

    return GraphPrep(
        driver=driver,
        plan=plan,
        caps=caps,
        schema_text=effective_schema,
        idx_df=idx_df,
        entity_ft_indexes=entity_ft_indexes,
        micro_ft_indexes=micro_ft_indexes,
        dataset_min_year=min_year,
        dataset_max_year=max_year,
        speaker_candidates=sp_df,
        entity_candidates=agg,
        per_query_top=per_query_top,
        plan_entity_queries=plan_entity_queries,
    )


def graph_pick_entities(
    client: Any,
    question: str,
    candidates_df: pd.DataFrame,
    max_anchors: int,
    debug_lines: List[str],
) -> List[str]:
    if client is None or candidates_df is None or candidates_df.empty:
        return []
    entity_select_cap = min(int(max_anchors), MAX_ENTITY_SELECT)
    llm_ids = llm_select_relevant_entity_ids(
        client=client,
        model=OPENROUTER_MODEL,
        question=question,
        candidates=candidates_df.to_dict(orient="records"),
        max_select=entity_select_cap,
        debug_lines=debug_lines,
    )
    return llm_ids


def graph_execute(
    question: str,
    settings: GraphSettings,
    prep: GraphPrep,
    picked_speaker_id: Optional[str],
    picked_entity_ids: List[str],
    debug_lines: List[str],
) -> GraphResult:
    executed: List[Dict[str, str]] = []

    def log_cypher(label: str, cypher: str):
        executed.append({"label": label, "cypher": cypher})
        debug_log(debug_lines, f"=== CYPHER [{label}] ===\n{cypher}")

    result = GraphResult(
        plan=prep.plan,
        schema_text=prep.schema_text,
        entity_ft_indexes=prep.entity_ft_indexes,
        micro_ft_indexes=prep.micro_ft_indexes,
        dataset_bounds={"min_year": prep.dataset_min_year, "max_year": prep.dataset_max_year},
        executed_cyphers=executed,
        debug_lines=debug_lines,
    )

    client = build_openrouter_client(OPENROUTER_API_KEY, OPENROUTER_BASE_URL)

    picked_entity_id: Optional[str] = picked_entity_ids[0] if picked_entity_ids else None

    action = prep.plan.get("action") or "entity_facts"
    strict = bool(settings.uat_mode)
    year = prep.plan.get("year")
    show_query = normalize_str(prep.plan.get("show_query")) or None
    exclude_terms = prep.plan.get("exclude_terms") or []
    require_evidence = bool(prep.plan.get("require_evidence", strict))

    state: Dict[str, Any] = {
        "result_df": pd.DataFrame(),
        "result_action": action,
        "fallback_reason": None,
    }

    def set_fallback(reason: str):
        state["fallback_reason"] = reason
        state["result_action"] = "generic_text2cypher"

    try:
        if action == "entity_facts":
            mode = normalize_str(prep.plan.get("entity_mode")).upper() or "SINGLE"
            if mode not in ["SINGLE", "MULTI_UNION", "MULTI_RELATION"]:
                mode = "SINGLE" if len(picked_entity_ids) <= 1 else "MULTI_UNION"

            if (not picked_entity_ids) and picked_speaker_id:
                log_cypher("speaker_to_entity_by_id", TOOL_SPEAKER_TO_ENTITY_BY_ID)
                sp2e = try_read_df(
                    prep.driver,
                    NEO4J_DATABASE or None,
                    TOOL_SPEAKER_TO_ENTITY_BY_ID,
                    params={"speaker_id": picked_speaker_id},
                    debug_lines=debug_lines,
                )
                if not sp2e.empty and normalize_str(sp2e.iloc[0].get("entity_id")):
                    picked_entity_ids = [normalize_str(sp2e.iloc[0].get("entity_id"))]
                    picked_entity_id = picked_entity_ids[0]

            if not picked_entity_ids:
                set_fallback("No entities resolved for entity_facts (multi-entity included).")
            else:
                df = pd.DataFrame()

                if mode == "SINGLE" and len(picked_entity_ids) > 1:
                    anchors = picked_entity_ids[: int(settings.max_anchors)]
                    per_anchor = max(30, int(settings.limit_rows) // max(1, len(anchors)))
                    log_cypher("multi_entity_facts_auto", TOOL_MULTI_ENTITY_FACTS_EVIDENCE)
                    df = try_read_df(
                        prep.driver,
                        NEO4J_DATABASE or None,
                        TOOL_MULTI_ENTITY_FACTS_EVIDENCE,
                        params={
                            "entity_ids": anchors,
                            "per_anchor_limit": int(per_anchor),
                            "evidence_k": int(settings.evidence_k),
                            "snippet_len": int(settings.snippet_len),
                        },
                        debug_lines=debug_lines,
                    )
                    if not df.empty:
                        df["evidence"] = df.apply(
                            lambda rr: merge_evidence(rr.get("evidence_rel"), rr.get("evidence_prop"), cap=int(settings.evidence_k)),
                            axis=1,
                        )
                        df = hard_filter_year_show(df, year, show_query, strict=strict)
                        df = hard_exclude(df, exclude_terms, strict=strict)
                        if require_evidence:
                            df = df[df["evidence"].apply(has_micro_evidence)].copy()

                elif mode == "SINGLE" or len(picked_entity_ids) <= 1:
                    if not picked_entity_id:
                        picked_entity_id = picked_entity_ids[0]
                    log_cypher("entity_facts", TOOL_ENTITY_FACTS_EVIDENCE)
                    df = try_read_df(
                        prep.driver,
                        NEO4J_DATABASE or None,
                        TOOL_ENTITY_FACTS_EVIDENCE,
                        params={
                            "entity_id": picked_entity_id,
                            "limit": int(settings.limit_rows),
                            "evidence_k": int(settings.evidence_k),
                            "snippet_len": int(settings.snippet_len),
                        },
                        debug_lines=debug_lines,
                    )
                    if not df.empty:
                        df["evidence"] = df.apply(
                            lambda rr: merge_evidence(rr.get("evidence_rel"), rr.get("evidence_prop"), cap=int(settings.evidence_k)),
                            axis=1,
                        )
                        df = hard_filter_year_show(df, year, show_query, strict=strict)
                        df = hard_exclude(df, exclude_terms, strict=strict)
                        if require_evidence:
                            df = df[df["evidence"].apply(has_micro_evidence)].copy()

                elif mode == "MULTI_RELATION":
                    log_cypher("entity_relation_facts", TOOL_ENTITY_RELATION_FACTS_EVIDENCE)
                    df = try_read_df(
                        prep.driver,
                        NEO4J_DATABASE or None,
                        TOOL_ENTITY_RELATION_FACTS_EVIDENCE,
                        params={
                            "entity_ids": picked_entity_ids[: int(settings.max_anchors)],
                            "limit": int(settings.limit_rows),
                            "evidence_k": int(settings.evidence_k),
                            "snippet_len": int(settings.snippet_len),
                        },
                        debug_lines=debug_lines,
                    )
                    if not df.empty:
                        df["evidence"] = df["evidence_rel"]
                        df = hard_filter_year_show(df, year, show_query, strict=strict)
                        df = hard_exclude(df, exclude_terms, strict=strict)
                        if require_evidence:
                            df = df[df["evidence"].apply(has_micro_evidence)].copy()

                else:
                    anchors = picked_entity_ids[: int(settings.max_anchors)]
                    per_anchor = max(30, int(settings.limit_rows) // max(1, len(anchors)))
                    log_cypher("multi_entity_facts", TOOL_MULTI_ENTITY_FACTS_EVIDENCE)
                    df = try_read_df(
                        prep.driver,
                        NEO4J_DATABASE or None,
                        TOOL_MULTI_ENTITY_FACTS_EVIDENCE,
                        params={
                            "entity_ids": anchors,
                            "per_anchor_limit": int(per_anchor),
                            "evidence_k": int(settings.evidence_k),
                            "snippet_len": int(settings.snippet_len),
                        },
                        debug_lines=debug_lines,
                    )
                    if not df.empty:
                        df["evidence"] = df.apply(
                            lambda rr: merge_evidence(rr.get("evidence_rel"), rr.get("evidence_prop"), cap=int(settings.evidence_k)),
                            axis=1,
                        )
                        df = hard_filter_year_show(df, year, show_query, strict=strict)
                        df = hard_exclude(df, exclude_terms, strict=strict)
                        if require_evidence:
                            df = df[df["evidence"].apply(has_micro_evidence)].copy()

                if df.empty:
                    spid = picked_speaker_id

                    if (not spid) and picked_entity_ids:
                        log_cypher("entity_to_speaker", TOOL_ENTITY_TO_SPEAKER)
                        sps = try_read_df(
                            prep.driver,
                            NEO4J_DATABASE or None,
                            TOOL_ENTITY_TO_SPEAKER,
                            params={"entity_id": picked_entity_ids[0]},
                            debug_lines=debug_lines,
                        )
                        if not sps.empty and normalize_str(sps.iloc[0].get("speaker_id")):
                            spid = normalize_str(sps.iloc[0].get("speaker_id"))

                    keyword_query = normalize_str(prep.plan.get("keyword_query"))
                    if not keyword_query and client is not None:
                        keyword_query = llm_extract_keyword_query(client, OPENROUTER_MODEL, question, debug_lines) or ""
                    terms = split_keyword_terms(keyword_query)

                    if spid and terms:
                        only_books = ("book" in question.lower()) or ("books" in question.lower())
                        log_cypher("speaker_keyword_mentions", TOOL_SPEAKER_KEYWORD_MENTIONS)
                        mentions_df = try_read_df(
                            prep.driver,
                            NEO4J_DATABASE or None,
                            TOOL_SPEAKER_KEYWORD_MENTIONS,
                            params={
                                "speaker_id": spid,
                                "terms": terms,
                                "only_books": bool(only_books),
                                "snippet_len": int(settings.snippet_len),
                                "evidence_k": int(settings.evidence_k),
                                "limit": int(settings.limit_rows),
                            },
                            debug_lines=debug_lines,
                        )

                        mentions_df = hard_filter_year_show(mentions_df, year, show_query, strict=strict)
                        mentions_df = hard_exclude(mentions_df, exclude_terms, strict=strict)
                        if require_evidence and "evidence" in mentions_df.columns:
                            mentions_df = mentions_df[mentions_df["evidence"].apply(has_micro_evidence)].copy()

                        if not mentions_df.empty:
                            state["result_df"] = mentions_df
                            state["result_action"] = "keyword_mentions_list"
                        else:
                            set_fallback("Facts empty and keyword-mentions fallback returned no rows.")
                    else:
                        if not spid:
                            set_fallback("Facts empty and speaker could not be resolved for keyword-mentions fallback.")
                        else:
                            set_fallback("Facts empty and no keyword query available for keyword-mentions fallback.")
                else:
                    if settings.llm_row_selector and client is not None and state.get("result_action") != "keyword_mentions_list":
                        if "anchor_entity_id" in df.columns and df["anchor_entity_id"].nunique(dropna=True) > 1:
                            kept_chunks: List[pd.DataFrame] = []
                            for (aid, aname), g in df.groupby(["anchor_entity_id", "anchor_entity_name"], dropna=False):
                                cand_rows = g.to_dict(orient="records")
                                picked_ids = llm_select_relevant_fact_ids(
                                    client=client,
                                    model=OPENROUTER_MODEL,
                                    question=question,
                                    anchor_name=normalize_str(aname) or "anchor",
                                    rows=cand_rows,
                                    max_rows=min(len(cand_rows), 220),
                                    max_evidence_each=min(2, int(settings.evidence_k)),
                                    max_select=MAX_FACT_SELECT,
                                    debug_lines=debug_lines,
                                )
                                if picked_ids:
                                    order = {fid: i for i, fid in enumerate(picked_ids)}
                                    gg = g.copy()
                                    gg["__rank"] = gg["fact_id"].astype(str).map(order).fillna(10**9)
                                    gg = gg[gg["fact_id"].astype(str).isin(set(order.keys()))].copy()
                                    gg = gg.sort_values(["__rank", "support_count", "fact_id"], ascending=[True, False, False]).drop(columns=["__rank"])
                                    kept_chunks.append(gg)
                                else:
                                    gg = g.copy()
                                    gg["__sc"] = pd.to_numeric(gg.get("support_count"), errors="coerce").fillna(0)
                                    fallback_cap = min(int(settings.selector_keep_facts), MAX_FACT_SELECT)
                                    gg = gg.sort_values(["__sc", "fact_id"], ascending=[False, False]).head(fallback_cap)
                                    gg = gg.drop(columns=["__sc"])
                                    kept_chunks.append(gg)
                            if kept_chunks:
                                df = pd.concat(kept_chunks, ignore_index=True)
                        else:
                            anchor_name = normalize_str(df.iloc[0].get("anchor_entity_name")) or normalize_str(
                                prep.plan.get("entity_query") or prep.plan.get("speaker_query")
                            )
                            cand_rows = df.to_dict(orient="records")
                            picked_ids = llm_select_relevant_fact_ids(
                                client=client,
                                model=OPENROUTER_MODEL,
                                question=question,
                                anchor_name=anchor_name,
                                rows=cand_rows,
                                max_rows=min(int(settings.limit_rows), 220),
                                max_evidence_each=min(2, int(settings.evidence_k)),
                                max_select=MAX_FACT_SELECT,
                                debug_lines=debug_lines,
                            )

                            if picked_ids:
                                order = {fid: i for i, fid in enumerate(picked_ids)}
                                df["__rank"] = df["fact_id"].astype(str).map(order).fillna(10**9)
                                df = df[df["fact_id"].astype(str).isin(set(order.keys()))].copy()
                                df = df.sort_values(["__rank", "support_count", "fact_id"], ascending=[True, False, False]).drop(columns=["__rank"])
                            else:
                                df["__sc"] = pd.to_numeric(df.get("support_count"), errors="coerce").fillna(0)
                                fallback_cap = min(int(settings.selector_keep_facts), MAX_FACT_SELECT)
                                df = df.sort_values(["__sc", "fact_id"], ascending=[False, False]).head(fallback_cap).drop(columns=["__sc"])

                    state["result_df"] = df
                    state["result_action"] = "entity_facts"

        elif action == "speaker_micros":
            if not picked_speaker_id:
                set_fallback("Could not resolve speaker for speaker_micros.")
            else:
                log_cypher("speaker_micros", TOOL_SPEAKER_MICROS)
                df = try_read_df(
                    prep.driver,
                    NEO4J_DATABASE or None,
                    TOOL_SPEAKER_MICROS,
                    params={"speaker_id": picked_speaker_id, "limit": int(settings.limit_rows), "snippet_len": int(settings.snippet_len)},
                    debug_lines=debug_lines,
                )

                df = hard_filter_year_show(df, year, show_query, strict=strict)
                df = hard_exclude(df, exclude_terms, strict=strict)

                if df.empty:
                    set_fallback("speaker_micros returned no rows under current constraints.")
                else:
                    if settings.llm_row_selector and client is not None:
                        cand_rows = df.to_dict(orient="records")
                        picked_mids = llm_select_relevant_micro_ids(
                            client=client,
                            model=OPENROUTER_MODEL,
                            question=question,
                            rows=cand_rows,
                            max_rows=min(int(settings.limit_rows), 240),
                            debug_lines=debug_lines,
                        )
                        if picked_mids:
                            order = {mid: i for i, mid in enumerate(picked_mids)}
                            df["__rank"] = df["micro_id"].astype(str).map(order).fillna(10**9)
                            df = df[df["micro_id"].astype(str).isin(set(order.keys()))].copy()
                            df = df.sort_values(["__rank", "start"], ascending=[True, True]).drop(columns=["__rank"])
                        else:
                            df = df.head(int(settings.selector_keep_micros))

                    state["result_df"] = df
                    state["result_action"] = "speaker_micros"

        elif action == "speaker_mentions":
            if not picked_speaker_id:
                set_fallback("Could not resolve speaker for speaker_mentions.")
            else:
                log_cypher("speaker_mentions", TOOL_SPEAKER_MENTIONED_ENTITIES)
                df = try_read_df(
                    prep.driver,
                    NEO4J_DATABASE or None,
                    TOOL_SPEAKER_MENTIONED_ENTITIES,
                    params={"speaker_id": picked_speaker_id, "limit": int(settings.limit_rows)},
                    debug_lines=debug_lines,
                )
                df = hard_exclude(df, exclude_terms, strict=strict)

                if df.empty:
                    set_fallback("speaker_mentions returned no rows.")
                else:
                    state["result_df"] = df
                    state["result_action"] = "speaker_mentions"

        elif action in ["keyword_micros", "first_mention", "term_disambiguate"]:
            keyword_query = normalize_str(prep.plan.get("keyword_query"))

            if not keyword_query and client is not None:
                keyword_query = llm_extract_keyword_query(client, OPENROUTER_MODEL, question, debug_lines) or ""
            if not keyword_query:
                keyword_query = question.strip()

            log_cypher("keyword_micros", TOOL_MICRO_FULLTEXT if prep.micro_ft_indexes else TOOL_MICRO_CONTAINS)
            micros_df = keyword_micros(
                prep.driver,
                NEO4J_DATABASE or None,
                keyword_query,
                prep.micro_ft_indexes,
                int(settings.snippet_len),
                int(settings.limit_rows),
                debug_lines,
            )

            micros_df = hard_filter_year_show(micros_df, year, show_query, strict=strict)
            micros_df = hard_exclude(micros_df, exclude_terms, strict=strict)

            if micros_df.empty:
                set_fallback("No matching Micro evidence found under current constraints.")
            else:
                if settings.llm_row_selector and client is not None and action in ["keyword_micros", "first_mention"]:
                    cand_rows = micros_df.to_dict(orient="records")
                    picked_mids = llm_select_relevant_micro_ids(
                        client=client,
                        model=OPENROUTER_MODEL,
                        question=question,
                        rows=cand_rows,
                        max_rows=min(int(settings.limit_rows), 260),
                        debug_lines=debug_lines,
                    )
                    if picked_mids:
                        order = {mid: i for i, mid in enumerate(picked_mids)}
                        micros_df["__rank"] = micros_df["micro_id"].astype(str).map(order).fillna(10**9)
                        micros_df = micros_df[micros_df["micro_id"].astype(str).isin(set(order.keys()))].copy()
                        micros_df = micros_df.sort_values(["__rank", "start"], ascending=[True, True]).drop(columns=["__rank"])
                    else:
                        micros_df = micros_df.head(int(settings.selector_keep_micros))

                if action == "keyword_micros":
                    state["result_df"] = micros_df
                    state["result_action"] = "keyword_micros"
                elif action == "first_mention":
                    micros_df2 = micros_df.copy()
                    micros_df2["doc_year_num"] = pd.to_numeric(micros_df2.get("doc_year"), errors="coerce")
                    micros_df2 = micros_df2.sort_values(["doc_year_num", "start"], ascending=True)
                    state["result_df"] = micros_df2.head(1)
                    state["result_action"] = "first_mention"
                else:
                    rows = micros_df.head(220).to_dict(orient="records")
                    sense_catalog, labels = llm_disambiguate(client, OPENROUTER_MODEL, keyword_query, rows, debug_lines)
                    micros_df["sense_name"] = micros_df["micro_id"].astype(str).map(labels).fillna("UNKNOWN")
                    if sense_catalog:
                        debug_log(debug_lines, f"Disambiguation senses: {', '.join(sense_catalog)}")
                    state["result_df"] = micros_df
                    state["result_action"] = "term_disambiguate"

        else:
            state["result_action"] = "generic_text2cypher"

    except Exception as e:
        debug_log(debug_lines, f"Tool execution failed: {e}")
        set_fallback(f"Tool execution failed: {e}")

    result.result_action = state["result_action"]
    result.result_df = state["result_df"]
    result.fallback_reason = state.get("fallback_reason")

    return result


def graph_safe_text2cypher(
    question: str,
    settings: GraphSettings,
    schema_text: str,
    entity_ft_indexes: List[str],
    micro_ft_indexes: List[str],
    debug_lines: List[str],
) -> Tuple[str, pd.DataFrame]:
    client = build_openrouter_client(OPENROUTER_API_KEY, OPENROUTER_BASE_URL)
    if client is None or not OPENROUTER_API_KEY.strip():
        raise RuntimeError("OpenRouter client unavailable (missing OPENROUTER_API_KEY / openai package).")

    ft_allow = {
        "entity": list(entity_ft_indexes[:2]) if entity_ft_indexes else [],
        "micro": list(micro_ft_indexes[:2]) if micro_ft_indexes else [],
    }
    cypher = llm_generate_cypher(client, OPENROUTER_MODEL, schema_text, question, fulltext_indexes=ft_allow)
    enforce_readonly_or_raise(cypher)
    cypher = ensure_limit(cypher, int(settings.limit_rows))

    df_fb = try_read_df(get_driver(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD), NEO4J_DATABASE or None, cypher, debug_lines=debug_lines)
    return cypher, df_fb


def graph_close_driver(driver: neo4j.Driver):
    try:
        driver.close()
    except Exception:
        pass


