from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List

import streamlit as st

from podkg_hybrid.shared.types import VectorResult, VectorSettings
from podkg_hybrid.vector.engine import GROUP_SIZE_FIXED


def _render_sanity_block(info: Dict[str, Any]) -> None:
    st.write("collection:", info.get("collection", ""))

    if "has_collection_error" in info:
        st.error(f"has_collection failed: {info['has_collection_error']}")
    elif "has_collection" in info:
        st.write("has_collection:", info["has_collection"])

    if "load_collection_error" in info:
        st.warning(f"load_collection failed: {info['load_collection_error']}")
    elif "load_collection" in info:
        st.write("load_collection:", info["load_collection"])

    if "collection_stats_error" in info:
        st.warning(f"get_collection_stats failed: {info['collection_stats_error']}")
    elif "collection_stats" in info:
        st.json(info["collection_stats"])

    if "schema_error" in info:
        st.warning(f"describe_collection failed: {info['schema_error']}")
    else:
        if info.get("schema"):
            st.json(info["schema"])
        if info.get("schema_fields"):
            st.write("schema fields:", info["schema_fields"])

    if "indexes_error" in info:
        st.warning(f"list_indexes failed: {info['indexes_error']}")
    elif info.get("indexes") is not None:
        st.json(info["indexes"])

    cfg = info.get("config", {})
    st.caption(
        "embed_model='{embed_model}' answer_model='{answer_model}' judge_model='{judge_model}' "
        "dense_field='{dense_field}' metric='{metric}' nprobe={nprobe_default} level={level} "
        "bm25={bm25_enabled} bm25_field='{bm25_field}'".format(**cfg)
    )


def render_vector_debug_and_results(result: VectorResult, settings: VectorSettings) -> None:
    if settings.show_sanity:
        with st.expander("Debug: Collection sanity", expanded=True):
            _render_sanity_block(result.debug_sanity or {})

    st.write("Filter expr:", result.filter_expr or "(none)")

    if result.error:
        st.warning(result.error)
        return

    st.write("Retrieval mode:", "HYBRID (Dense+BM25)" if result.used_hybrid else "DENSE ONLY")

    if settings.grouping_enabled:
        if result.used_grouping:
            st.success(
                "Grouping Search applied: group_by_field={group_by_field} groups={groups} "
                "group_size={group_size} (approx {approx} chunks)".format(
                    group_by_field=result.group_by_field or "",
                    groups=int(result.retrieval_limit),
                    group_size=GROUP_SIZE_FIXED,
                    approx=int(result.approx_total_chunks),
                )
            )
        else:
            st.warning(
                "Grouping Search requested but this MilvusClient did not accept grouping parameters. "
                "Search ran without grouping."
            )

    if result.output_fields:
        st.write("output_fields (effective):", result.output_fields)
    if result.missing_fields:
        st.warning("Missing fields in schema (not requested): " + ", ".join(result.missing_fields))

    if result.term_scan_block:
        st.markdown("### Debug: Term Scan")
        st.code(result.term_scan_block)

    st.markdown("### Retrieval Results (Detailed)")

    sources = result.sources or []

    if settings.grouping_enabled and result.group_by_field:
        groups: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
        for s in sources:
            gval = s.get(str(result.group_by_field), None)
            key = str(gval) if (gval is not None and str(gval).strip() != "") else "(missing)"
            groups.setdefault(key, []).append(s)

        for gk, items in groups.items():
            with st.expander(f"{result.group_by_field} = {gk}  |  {len(items)} chunks", expanded=False):
                for s in items:
                    title = s.get("title") or "(no title)"
                    speaker = s.get("primary_speaker") or ""
                    tsec = int(float(s.get("start_sec") or 0))
                    hdr = f"{s['sid']} | {title} | t={tsec}s | {speaker}".strip()

                    with st.expander(hdr, expanded=False):
                        if s.get("ts_url"):
                            st.markdown(f"**Timestamp evidence:** [{s['ts_url']}]({s['ts_url']})")

                        meta = {k: v for k, v in s.items() if k not in ("text", "text_full", "_raw")}
                        st.markdown("**Metadata (all):**")
                        st.json(meta)

                        st.markdown("**Text:**")
                        st.markdown(s.get("text") or "")

                        if settings.show_raw_hit:
                            st.markdown("**Raw hit (debug):**")
                            st.json(s.get("_raw", {}))
    else:
        for s in sources:
            title = s.get("title") or "(no title)"
            speaker = s.get("primary_speaker") or ""
            tsec = int(float(s.get("start_sec") or 0))
            hdr = f"{s['sid']} | {title} | t={tsec}s | {speaker}".strip()

            with st.expander(hdr, expanded=False):
                if s.get("ts_url"):
                    st.markdown(f"**Timestamp evidence:** [{s['ts_url']}]({s['ts_url']})")

                meta = {k: v for k, v in s.items() if k not in ("text", "text_full", "_raw")}
                st.markdown("**Metadata (all):**")
                st.json(meta)

                st.markdown("**Text:**")
                st.markdown(s.get("text") or "")

                if settings.show_raw_hit:
                    st.markdown("**Raw hit (debug):**")
                    st.json(s.get("_raw", {}))

    if result.timings:
        st.markdown("### Latency")
        st.write(
            {
                "embed_s": result.timings.get("embed_s"),
                "search_s": result.timings.get("search_s"),
                "total_s": result.timings.get("total_s"),
                "retrieval_kind": "groups" if settings.grouping_enabled else "chunks",
                "retrieval_limit": int(result.retrieval_limit),
                "group_size": GROUP_SIZE_FIXED if settings.grouping_enabled else 0,
                "approx_total_chunks": int(result.approx_total_chunks),
                "used_hybrid": bool(result.used_hybrid),
                "grouping_requested": bool(settings.grouping_enabled),
                "grouping_applied": bool(result.used_grouping),
                "group_by_field": result.group_by_field or "",
                "output_fields_effective": result.output_fields,
                "output_fields_missing": result.missing_fields,
                "num_hits_returned": len(sources),
            }
        )
