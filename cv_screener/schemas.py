"""Pydantic schemas for structured LLM extraction and scoring."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class CandidateProfile(BaseModel):
    """Structured data extracted from a CV."""

    name: str = Field(description="Candidate full name")
    title: str = Field(default="", description="Current or most recent job title")
    email: str = Field(default="", description="Email address if present")
    location: str = Field(default="", description="City/country if present")
    years_experience: float = Field(default=0, description="Total years of professional experience")
    skills: List[str] = Field(default_factory=list, description="Technical skills and tools")
    summary: str = Field(default="", description="One-sentence professional summary")


class CandidateScore(BaseModel):
    """LLM assessment of a candidate against a job description."""

    score: int = Field(description="Fit score from 0 to 100")
    matched_skills: List[str] = Field(default_factory=list, description="Job skills the candidate has")
    missing_skills: List[str] = Field(default_factory=list, description="Job skills the candidate lacks")
    rationale: str = Field(description="Two-sentence explanation of the score")
