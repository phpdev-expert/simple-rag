"""Run the CV-screening pipeline over all generated CVs and rank candidates.

Usage:
    python main.py                 # uses the built-in sample job description
    python main.py "job text..."   # screen against your own job description
"""
from __future__ import annotations

import glob
import os
import sys

from dotenv import load_dotenv

from cv_screener.graph import GRAPH
from cv_screener.llm import active_model_label, has_api_key

load_dotenv()

CV_DIR = os.path.join(os.path.dirname(__file__), "data", "cvs")

DEFAULT_JOB = (
    "Senior Backend Engineer. We need strong Python and Django, PostgreSQL, "
    "and cloud experience with AWS, Docker, and Kubernetes. REST APIs and "
    "system design skills required. 5+ years experience preferred."
)


def main() -> None:
    job = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JOB

    cv_paths = sorted(glob.glob(os.path.join(CV_DIR, "*.pdf")))
    if not cv_paths:
        print("No CVs found. Run `python generate_cvs.py` first.")
        sys.exit(1)

    mode = active_model_label() if has_api_key() else "heuristic fallback (no LLM key)"
    print(f"Screening {len(cv_paths)} CVs using {mode}.\n")
    print(f"Job:\n  {job}\n")

    results = []
    for path in cv_paths:
        final = GRAPH.invoke({"cv_path": path, "job_description": job})
        if final.get("error"):
            print(f"  ! {os.path.basename(path)}: {final['error']}")
            continue
        results.append((final["profile"], final["score"]))

    results.sort(key=lambda r: r[1].score, reverse=True)

    print("\n=== Ranked candidates ===")
    for rank, (profile, score) in enumerate(results, start=1):
        print(f"\n#{rank}  {profile.name} — {profile.title}  ({score.score}/100)")
        print(f"     experience: {profile.years_experience:g} yrs | {profile.email}")
        if score.matched_skills:
            print(f"     matched: {', '.join(score.matched_skills)}")
        if score.missing_skills:
            print(f"     missing: {', '.join(score.missing_skills)}")
        print(f"     {score.rationale}")


if __name__ == "__main__":
    main()
