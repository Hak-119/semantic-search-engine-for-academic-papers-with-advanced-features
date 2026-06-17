"""
backend package init.

Sets critical environment variables BEFORE any ML library (torch, faiss,
transformers) is imported anywhere in this package. This file runs
automatically first whenever any backend submodule is imported.

Why this is needed: on macOS, torch, faiss, and scikit-learn each bring
their own native multi-threaded math engine (OpenMP). When more than one
tries to run in the same process at the same time, they collide at the
C++ level and crash the whole program with a segmentation fault -
something Python's try/except cannot catch, because it happens below
the Python layer entirely.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")