"""Graph state definition shared across nodes."""
from __future__ import annotations

from typing import List, Optional, TypedDict

from .schemas import CandidateProfile, CandidateScore


class ScreeningState(TypedDict, total=False):
    """State threaded through the LangGraph pipeline for a single CV."""

    cv_path: str            # input: path to the CV PDF
    job_description: str     # input: role we are screening for
    raw_text: str            # after load_cv
    profile: CandidateProfile  # after extract_profile
    score: CandidateScore      # after score_candidate
    error: Optional[str]      # populated if any node fails
