#!/usr/bin/env python3
"""Reproduce Paper 0 from the bundled, analysis-ready inputs.

Runs every estimation and figure step from the bundled country reductions and
GDP files. The 1 km versus 2 km step reads its cached reduction and queries
Earth Engine only if that cache is incomplete. `--verify` checks the frozen
outputs against OUTPUT_MANIFEST.csv without running anything.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STEPS = [
    ("HSW-era DMSP validation", "global_hsw/validate_hsw.py"),
    ("DMSP validation with satellite-specific data", "global_hsw/analyze_p0_dmsp_satellite.py"),
    ("VIIRS specification battery", "global_hsw/analyze_global_pilot.py"),
    ("rolling sensor-era windows", "global_hsw/analyze_rolling_windows.py"),
    ("window-length checks", "global_hsw/analyze_window_robustness.py"),
    ("four-band comparison and 2013 bridge", "global_hsw/analyze_bands_bridge.py"),
    ("monthly temporal-cleaning comparison", "global_hsw/analyze_pstt_country.py"),
    ("pixel-background country comparison", "global_hsw/analyze_bgnoise_country.py"),
    ("pixel-background resolution check", "global_hsw/analyze_bgnoise_scale.py"),
    ("common-sample bands and threshold variants", "global_hsw/analyze_p0_common_sample.py"),  # writes the four-band panel read by the heterogeneity step
    ("paired band-gap and held-out tests", "global_hsw/analyze_p0_oos.py"),
    ("held-out masking gain by brightness, income and capacity", "global_hsw/analyze_p0_heterogeneity.py"),
    ("DMSP held-out reference", "global_hsw/analyze_p0_dmsp_heldout.py"),
    ("estimates with Vanuatu included", "global_hsw/analyze_p0_vanuatu_sensitivity.py"),
    ("2012 and 2013 sensor overlap", "global_hsw/analyze_p0_overlap_2012.py"),
    ("calendar-year lights against fiscal-year GDP", "global_hsw/analyze_p0_fiscal_year.py"),
    ("fixed-support whole-history reassignment", "global_hsw/analyze_p0_permutation.py"),
    ("descriptive statistics", "global_hsw/analyze_p0_descriptives.py"),
    ("1 km versus 2 km band sensitivity", "global_hsw/extract_p0_band_resolution.py"),
    ("paper figures", "global_hsw/make_paper_figures.py"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="list steps without running")
    parser.add_argument("--verify", action="store_true", help="only check frozen outputs against OUTPUT_MANIFEST.csv")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="stop at the first failed step instead of reporting all results",
    )
    args = parser.parse_args()
    if args.list:
        for label, script in STEPS:
            print(f"{label}: {script}")
        return
    if args.verify:
        raise SystemExit(verify())

    results = []
    start = time.time()
    for number, (label, script) in enumerate(STEPS, start=1):
        print(f"\n[{number}/{len(STEPS)}] {label}", flush=True)
        step_start = time.time()
        completed = subprocess.run(
            [sys.executable, str(ROOT / script)], cwd=ROOT, check=False
        )
        elapsed = time.time() - step_start
        status = "OK" if completed.returncode == 0 else "FAIL"
        results.append((status, script, elapsed, completed.returncode))
        if completed.returncode and args.stop_on_error:
            break

    print("\nSummary")
    for status, script, elapsed, returncode in results:
        note = f"{elapsed:.1f}s" if status == "OK" else f"exit {returncode}"
        print(f"  {status:4} {script:48} {note}")
    failures = sum(status == "FAIL" for status, *_ in results)
    print(f"\n{len(results) - failures} ok, {failures} failed in {time.time() - start:.1f}s")
    raise SystemExit(1 if failures else 0)


def verify() -> int:
    """Compare frozen outputs with OUTPUT_MANIFEST.csv (path, bytes, sha256)."""
    import csv
    import hashlib
    bad = 0
    with (ROOT / "OUTPUT_MANIFEST.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        f = ROOT / r["path"]
        if not f.exists():
            print("MISSING", r["path"]); bad += 1
        elif hashlib.sha256(f.read_bytes()).hexdigest() != r["sha256"]:
            print("CHANGED", r["path"]); bad += 1
    print(f"{len(rows) - bad} of {len(rows)} frozen outputs verified")
    return 1 if bad else 0


if __name__ == "__main__":
    main()
