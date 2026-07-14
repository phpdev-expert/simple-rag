"""Build (or rebuild) the Chroma vector index over the CVs.

Usage:
    python build_index.py            # build if missing
    python build_index.py --rebuild  # wipe and rebuild
"""
from __future__ import annotations

import sys

from cv_screener.rag import PERSIST_DIR, build_index

if __name__ == "__main__":
    rebuild = "--rebuild" in sys.argv
    print(f"{'Rebuilding' if rebuild else 'Building'} Chroma index at {PERSIST_DIR} ...")
    count = build_index(rebuild=rebuild)
    print(f"Indexed {count} chunks.")
