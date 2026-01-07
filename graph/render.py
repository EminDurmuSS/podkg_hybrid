from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from podkg_hybrid.graph.engine import (
    fetch_micro_context,
    format_micro_evidence,
    json_dumps_safe,
    normalize_str,
)
from podkg_hybrid.shared.types import GraphResult, GraphSettings


def render_micro_context_block(
    driver,
    db: Optional[str],
    micro_id: str,
    window: int,
    snippet_len: int,
    debug_lines: List[str],
):
    ma, ctx = fetch_micro_context(driver, db, micro_id, window, snippet_len, debug_lines)
    if not ctx:
        st.caption("No macro context found for this micro.")
        return

    title_bits = []
    if ma.get("doc_year"):
        title_bits.append(str(ma.get("doc_year")))
    if ma.get("doc_title"):
        title_bits.append(normalize_str(ma.get("doc_title")))
    if ma.get("macro_id"):
        title_bits.append(f"macro_id={ma.get('macro_id')}")
    st.caption(" | ".join(title_bits) if title_bits else "Macro context")

    lines = []
    for r in ctx:
        is_center = str(r.get("micro_id")) == str(micro_id)
        prefix = ">> " if is_center else "   "
        head = f"{prefix}{r.get('micro_id')} [{r.get('start')}-{r.get('end')}] speaker={r.get('speaker')}"
        sn = normalize_str(r.get("snippet"))
        if is_center:
            head = f"**{head}**"
            if sn:
                head += f"\n\n> **{sn}**"
        else:
            if sn:
                head += f"\n\n> {sn}"
        if r.get("url"):
            head += f"\n\n{r.get('url')}"
        lines.append(head)

    st.markdown("\n\n---\n\n".join(lines))


def render_macro_summary_from_micros(df: pd.DataFrame):
    if df is None or df.empty:
        return
    if "macro_id" not in df.columns:
        return
    d = df.copy()
    d["macro_id"] = d["macro_id"].astype(str)
    grp = (
        d.groupby(["macro_id", "doc_id", "doc_year", "doc_title"], dropna=False)
        .size()
        .reset_index(name="micro_count")
        .sort_values(["micro_count", "doc_year", "doc_title"], ascending=[False, True, True])
    )
    st.subheader("Macro list (derived from these micros)")
    st.caption("This list is built from the returned Micro rows, so the selected Micro is guaranteed to be included.")
    st.dataframe(grp.head(200), use_container_width=True)


def render_graph_results(
    result: GraphResult,
    settings: GraphSettings,
    driver,
    db: Optional[str],
):
    if result.error:
        st.error(result.error)
        return

    if settings.debug_show_schema and result.schema_text:
        st.subheader("Schema")
        st.code(result.schema_text, language="text")

    if settings.show_plan and result.plan:
        st.subheader("Plan JSON")
        st.code(json_dumps_safe(result.plan), language="json")

    bounds = result.dataset_bounds or {}
    if bounds.get("min_year") is not None and bounds.get("max_year") is not None:
        st.caption(f"Dataset years (Macro.doc_year): {bounds.get('min_year')} -> {bounds.get('max_year')}")

    if settings.show_executed_cyphers and result.executed_cyphers:
        with st.expander("Executed Cyphers"):
            for it in result.executed_cyphers:
                st.markdown(f"**{it['label']}**")
                st.code(it["cypher"], language="cypher")

    st.subheader(f"Action: {result.result_action}")

    df = result.result_df
    action = result.result_action

    if action == "keyword_mentions_list" and df is not None and not df.empty:
        st.subheader("Keyword mentions (speaker micros -> macro mentions)")
        st.dataframe(df.head(200), use_container_width=True)

        st.subheader("Evidence-first view")
        for i in range(min(15, len(df))):
            r = df.iloc[i].to_dict()
            hdr = f"{i+1}) {normalize_str(r.get('entity_name'))}"
            with st.expander(hdr):
                st.write(format_micro_evidence(r.get("evidence"), max_items=int(settings.evidence_k)))
                if settings.show_macro_context and isinstance(r.get("evidence"), list) and r["evidence"]:
                    st.divider()
                    st.markdown("**Macro context (center micro guaranteed):**")
                    mid0 = None
                    for e in r["evidence"]:
                        if isinstance(e, dict) and e.get("micro_id"):
                            mid0 = normalize_str(e.get("micro_id"))
                            break
                    if mid0:
                        render_micro_context_block(
                            driver,
                            db,
                            micro_id=mid0,
                            window=int(settings.macro_window),
                            snippet_len=int(settings.macro_context_snippet_len),
                            debug_lines=result.debug_lines,
                        )

    elif action == "entity_facts" and df is not None and not df.empty:
        if "anchor_entity_id" in df.columns and df["anchor_entity_id"].nunique(dropna=True) > 1:
            st.subheader("Facts (multi-entity, grouped by anchor)")
            for (aid, aname), g in df.groupby(["anchor_entity_id", "anchor_entity_name"], dropna=False):
                st.subheader(f"{normalize_str(aname)} ({normalize_str(aid)})")
                st.dataframe(g.head(120), use_container_width=True)

                st.markdown("**Evidence-first view (top rows)**")
                gg = g.copy()
                for i in range(min(8, len(gg))):
                    r = gg.iloc[i].to_dict()
                    other = normalize_str(r.get("other_entity") or r.get("object") or "")
                    pred = normalize_str(r.get("predicate"))
                    with st.expander(f"{i+1}) {other}  -  {pred}"):
                        st.write(format_micro_evidence(r.get("evidence"), max_items=int(settings.evidence_k)))

                        if settings.show_macro_context and isinstance(r.get("evidence"), list) and r["evidence"]:
                            st.divider()
                            st.markdown("**Macro context (center micro guaranteed):**")
                            mid0 = None
                            for e in r["evidence"]:
                                if isinstance(e, dict) and e.get("micro_id"):
                                    mid0 = normalize_str(e.get("micro_id"))
                                    break
                            if mid0:
                                render_micro_context_block(
                                    driver,
                                    db,
                                    micro_id=mid0,
                                    window=int(settings.macro_window),
                                    snippet_len=int(settings.macro_context_snippet_len),
                                    debug_lines=result.debug_lines,
                                )
        else:
            st.subheader("Facts (evidence-gated)")
            st.dataframe(df.head(200), use_container_width=True)

            st.subheader("Evidence-first view")
            for i in range(min(12, len(df))):
                r = df.iloc[i].to_dict()
                other = normalize_str(r.get("other_entity") or r.get("object") or "")
                pred = normalize_str(r.get("predicate"))
                with st.expander(f"{i+1}) {other}  -  {pred}"):
                    st.write(format_micro_evidence(r.get("evidence"), max_items=int(settings.evidence_k)))

                    if settings.show_macro_context and isinstance(r.get("evidence"), list) and r["evidence"]:
                        st.divider()
                        st.markdown("**Macro context (center micro guaranteed):**")
                        mid0 = None
                        for e in r["evidence"]:
                            if isinstance(e, dict) and e.get("micro_id"):
                                mid0 = normalize_str(e.get("micro_id"))
                                break
                        if mid0:
                            render_micro_context_block(
                                driver,
                                db,
                                micro_id=mid0,
                                window=int(settings.macro_window),
                                snippet_len=int(settings.macro_context_snippet_len),
                                debug_lines=result.debug_lines,
                            )

    elif action in ["keyword_micros", "first_mention", "speaker_micros"] and df is not None and not df.empty:
        st.subheader("Micros")
        st.dataframe(df.head(200), use_container_width=True)

        render_macro_summary_from_micros(df)

        st.subheader("Evidence view")
        for i in range(min(20, len(df))):
            r = df.iloc[i].to_dict()
            hdr = f"{i+1}) micro_id={r.get('micro_id')} [{r.get('start')}-{r.get('end')}] speaker={r.get('speaker')} | macro_id={r.get('macro_id')}"
            with st.expander(hdr):
                st.write(f"\"{normalize_str(r.get('snippet'))}\"")
                if r.get("url"):
                    st.write(r.get("url"))
                if r.get("doc_year") or r.get("doc_title"):
                    st.caption(f"{r.get('doc_year')} | {normalize_str(r.get('doc_title'))}")

                if settings.show_macro_context:
                    st.divider()
                    st.markdown("**Macro context (center micro guaranteed):**")
                    mid = normalize_str(r.get("micro_id"))
                    if mid:
                        render_micro_context_block(
                            driver,
                            db,
                            micro_id=mid,
                            window=int(settings.macro_window),
                            snippet_len=int(settings.macro_context_snippet_len),
                            debug_lines=result.debug_lines,
                        )

    elif action == "term_disambiguate" and df is not None and not df.empty:
        st.subheader("Disambiguation (sense_name)")
        st.dataframe(df.head(250), use_container_width=True)

        if "sense_name" in df.columns:
            senses = [x for x in df["sense_name"].dropna().unique().tolist() if x]
            senses_sorted = sorted(senses, key=lambda x: (x == "UNKNOWN", str(x)))
            for s in senses_sorted:
                st.subheader(f"Sense: {s}")
                sdf = df[df["sense_name"] == s].copy()
                if not sdf.empty:
                    st.dataframe(sdf.head(200), use_container_width=True)
                else:
                    st.info("No rows.")

    elif action == "speaker_mentions" and df is not None and not df.empty:
        st.subheader("Mentioned entities")
        st.dataframe(df.head(400), use_container_width=True)

    elif action != "generic_text2cypher":
        st.warning("No rows returned.")

    with st.expander("Debug log"):
        st.code("\n\n".join(result.debug_lines), language="text")
