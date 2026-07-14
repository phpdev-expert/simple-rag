"""Assemble the LangGraph screening pipeline."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import extract_profile, load_cv, score_candidate
from .state import ScreeningState


def build_graph():
    """Wire the linear pipeline: START -> load -> extract -> score -> END."""
    builder = StateGraph(ScreeningState)
    builder.add_node("load_cv", load_cv)
    builder.add_node("extract_profile", extract_profile)
    builder.add_node("score_candidate", score_candidate)

    builder.add_edge(START, "load_cv")
    builder.add_edge("load_cv", "extract_profile")
    builder.add_edge("extract_profile", "score_candidate")
    builder.add_edge("score_candidate", END)

    return builder.compile()


# Compiled once and reused.
GRAPH = build_graph()
