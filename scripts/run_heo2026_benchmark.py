#!/usr/bin/env python3
"""Run the paper-matched cross-subject semantic decoding benchmark."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tada.heo2026_benchmark import main


if __name__ == "__main__":
    main()
