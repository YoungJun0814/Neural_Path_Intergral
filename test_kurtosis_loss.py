"""
Thin shim kept for backwards compatibility.

The actual tests have moved to ``tests/test_kurtosis_loss.py`` and can be run
with ``pytest``. Running this file directly still exercises the key code paths
but is no longer the preferred entry point.
"""
from __future__ import annotations

import os
import sys

# Ensure project root is on sys.path regardless of CWD
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> int:
    """Manually drive the smoke check for CLI parity."""
    from tests.test_kurtosis_loss import test_kurtosis_penalty_runs

    test_kurtosis_penalty_runs()
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
