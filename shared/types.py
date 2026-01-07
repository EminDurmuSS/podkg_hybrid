from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class VectorSettings:
    grouping_enabled: bool
    group_by_field: Optional[str]
    total_chunks_budget: int
    strict_group_size: bool
    top_k_chunks: int
    hybrid_enabled: bool
    ranker_kind: str
    dense_limit: int
    bm25_limit: int
    nprobe: int
    bm25_drop_ratio_search: float
    w_dense: float
    w_bm25: float
    video_id: str
    doc_id: str
    year: int
    show: str
    show_sanity: bool
    show_raw_hit: bool
    mode: str


@dataclass
class VectorResult:
    hits: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    sources_text: str = ""
    term_scan_block: str = ""
    term_scan: Dict[str, Any] = field(default_factory=dict)
    used_hybrid: bool = False
    used_grouping: bool = False
    retrieval_limit: int = 0
    approx_total_chunks: int = 0
    output_fields: List[str] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    group_by_field: Optional[str] = None
    filter_expr: Optional[str] = None
    error: Optional[str] = None
    debug_sanity: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)


@dataclass
class GraphSettings:
    planner_enabled: bool
    llm_row_selector: bool
    generate_answer: bool
    enable_safe_text2cypher: bool
    show_macro_context: bool
    macro_window: int
    macro_context_snippet_len: int
    selector_keep_facts: int
    selector_keep_micros: int
    use_manual_schema: bool
    schema_text: str
    top_candidates: int
    limit_rows: int
    evidence_k: int
    snippet_len: int
    max_anchors: int
    show_plan: bool
    show_executed_cyphers: bool
    debug_show_schema: bool
    debug_print_to_console: bool
    uat_mode: bool = True


@dataclass
class GraphPrep:
    driver: Any
    plan: Dict[str, Any]
    caps: Dict[str, Any]
    schema_text: str
    idx_df: pd.DataFrame
    entity_ft_indexes: List[str]
    micro_ft_indexes: List[str]
    dataset_min_year: Optional[int]
    dataset_max_year: Optional[int]
    speaker_candidates: pd.DataFrame
    entity_candidates: pd.DataFrame
    per_query_top: Dict[str, Optional[str]]
    plan_entity_queries: List[str]


@dataclass
class GraphResult:
    result_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    result_action: str = ""
    plan: Dict[str, Any] = field(default_factory=dict)
    fallback_reason: Optional[str] = None
    executed_cyphers: List[Dict[str, str]] = field(default_factory=list)
    debug_lines: List[str] = field(default_factory=list)
    schema_text: str = ""
    dataset_bounds: Dict[str, Optional[int]] = field(default_factory=dict)
    entity_ft_indexes: List[str] = field(default_factory=list)
    micro_ft_indexes: List[str] = field(default_factory=list)
    generated_cypher: Optional[str] = None
    error: Optional[str] = None
