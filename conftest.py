"""Project-level pytest config.

Adds the project root to ``sys.path`` so that ``from src.* import ...``
works regardless of where pytest is invoked from.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
