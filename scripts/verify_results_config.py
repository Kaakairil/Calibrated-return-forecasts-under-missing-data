"""Verify that results/metrics.csv matches the recorded grid dimensions.

Exits non-zero (with a list of inconsistencies) when the results table was
produced with a different grid than the one recorded in run_metadata.json —
the machine-checkable guard for the config-staleness bug found in the audit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibmiss.run_experiment import verify_results_config

if __name__ == "__main__":
    mismatches = verify_results_config()
    if mismatches:
        for line in mismatches:
            print(f"INCONSISTENT: {line}")
        raise SystemExit(1)
    print("Results are consistent with the recorded grid dimensions.")
