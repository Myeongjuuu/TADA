#!/usr/bin/env python3
"""Train and evaluate the SRM-CNN baseline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tada.srm_cnn_baseline import main


if __name__ == "__main__":
    main()
