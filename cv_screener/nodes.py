"""LangGraph nodes: load -> extract -> score.

Each node is a pure function State -> partial State update. When an
ANTHROPIC_API_KEY is present the extract/score nodes use Claude via LangChain;
otherwise they fall back to deterministic heuristics.
"""
from __future__ import annotations

import re
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

from .llm import get_llm
from .schemas import CandidateProfile, CandidateScore
from .state import ScreeningState

# A small skill vocabulary used by the heuristic fallback.
KNOWN_SKILLS = [
    "python", "java", "javascript", "typescript", "go", "swift", "kotlin", "r",
    "django", "spring boot", "react", "node.js", "graphql", "rest apis",
    "postgresql", "mysql", "mongodb", "sql", "spark",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd",
    "pytorch", "tensorflow", "scikit-learn", "pandas", "langchain", "langgraph",
    "transformers", "mlops", "cuda", "microservices", "system design", "leadership",
]


def _has_skill(text_lower: str, skill: str) -> bool:
    """Whole-token match so 'r' / 'go' don't match inside 'years' / 'goals'."""
    pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
    return re.search(pattern, text_lower) is not None


# --------------------------------------------------------------------------- #
# Node 1: load the PDF into raw text via LangChain's PyPDFLoader.
# --------------------------------------------------------------------------- #
def load_cv(state: ScreeningState) -> ScreeningState:
    try:
        docs = PyPDFLoader(state["cv_path"]).load()
        text = "\n".join(d.page_content for d in docs).strip()
        if not text:
            return {"error": "empty PDF text"}
        return {"raw_text": text}
    except Exception as exc:  # noqa: BLE001 - surface any loader failure in state
        return {"error": f"load_cv failed: {exc}"}


# --------------------------------------------------------------------------- #
# Node 2: extract a structured CandidateProfile.
# --------------------------------------------------------------------------- #
_EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You extract structured data from a CV. Be accurate; leave fields empty if unknown."),
    ("human", "Extract the candidate profile from this CV:\n\n{cv_text}"),
])


def extract_profile(state: ScreeningState) -> ScreeningState:
    if state.get("error"):
        return {}
    text = state["raw_text"]
    llm = get_llm()
    if llm is not None:
        try:
            chain = _EXTRACT_PROMPT | llm.with_structured_output(CandidateProfile)
            profile = chain.invoke({"cv_text": text[:6000]})
            return {"profile": profile}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"extract_profile failed: {exc}"}
    return {"profile": _heuristic_profile(text)}


def _heuristic_profile(text: str) -> CandidateProfile:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    name = lines[0] if lines else "Unknown"
    title = lines[1] if len(lines) > 1 else ""

    email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    years_match = re.search(r"(\d+(?:\.\d+)?)\s*years", text, re.IGNORECASE)
    lower = text.lower()
    skills = [s for s in KNOWN_SKILLS if _has_skill(lower, s)]

    return CandidateProfile(
        name=name,
        title=title,
        email=email_match.group(0) if email_match else "",
        years_experience=float(years_match.group(1)) if years_match else 0,
        skills=skills,
        summary=title,
    )


# --------------------------------------------------------------------------- #
# Node 3: score the candidate against the job description.
# --------------------------------------------------------------------------- #
_SCORE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a technical recruiter. Score candidate fit from 0-100 against the job."),
    ("human", "Job description:\n{job}\n\nCandidate profile:\n{profile}\n\nScore the fit."),
])


def score_candidate(state: ScreeningState) -> ScreeningState:
    if state.get("error"):
        return {}
    profile: CandidateProfile = state["profile"]
    job = state["job_description"]
    llm = get_llm()
    if llm is not None:
        try:
            chain = _SCORE_PROMPT | llm.with_structured_output(CandidateScore)
            score = chain.invoke({"job": job, "profile": profile.model_dump_json(indent=2)})
            return {"score": score}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"score_candidate failed: {exc}"}
    return {"score": _heuristic_score(profile, job)}


def _heuristic_score(profile: CandidateProfile, job: str) -> CandidateScore:
    job_skills = _skills_from_job(job)
    have = {s.lower() for s in profile.skills}
    matched = [s for s in job_skills if s in have]
    missing = [s for s in job_skills if s not in have]

    skill_ratio = len(matched) / len(job_skills) if job_skills else 0
    exp_bonus = min(profile.years_experience / 10, 1.0)  # cap at 10 years
    score = int(round(100 * (0.75 * skill_ratio + 0.25 * exp_bonus)))

    return CandidateScore(
        score=score,
        matched_skills=matched,
        missing_skills=missing,
        rationale=(
            f"Matches {len(matched)}/{len(job_skills)} required skills "
            f"with {profile.years_experience:g} years experience."
        ),
    )


def _skills_from_job(job: str) -> List[str]:
    lower = job.lower()
    return [s for s in KNOWN_SKILLS if _has_skill(lower, s)]
