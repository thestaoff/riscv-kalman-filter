#!/usr/bin/env python3
"""
generate_m4_output_csvs.py

Milestone 4 helper script.
Runs the vectorised LKF/EKF programs and standardises their generated CSVs as:
  output/lkf_vector_output.csv
  output/ekf_vector_output.csv

Use this after your Makefile successfully builds the vector executables.

Examples:
  python generate_m4_output_csvs.py

  python generate_m4_output_csvs.py \
    --lkf-cmd ./lkf_vector \
    --ekf-cmd ./ekf_vector \
    --out-dir output

  # If you already ran the programs and know the CSV paths:
  python generate_m4_output_csvs.py --no-run \
    --lkf-source output/lkf_output.csv \
    --ekf-source output/ekf_output.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

EXPECTED_STATE_COLS = 276

LKF_CANDIDATES = [
    "output/lkf_vector_output.csv",
    "lkf_vector_output.csv",
    "output/lkf_output.csv",
    "lkf_output.csv",
    "output/lkf_asm_output.csv",
    "lkf_asm_output.csv",
]

EKF_CANDIDATES = [
    "output/ekf_vector_output.csv",
    "ekf_vector_output.csv",
    "output/ekf_output.csv",
    "ekf_output.csv",
    "output/ekf_asm_output.csv",
    "ekf_asm_output.csv",
]


def run_command(command: str, label: str) -> None:
    print(f"\n[run] {label}: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"{label} command failed with exit code {result.returncode}: {command}")


def newest_csv(root: Path, patterns: Iterable[str], since: float) -> Optional[Path]:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(root.glob(pattern))
    matches = [p for p in matches if p.is_file() and p.stat().st_mtime >= since]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def find_csv(root: Path, candidates: list[str], broad_patterns: list[str], since: float) -> Optional[Path]:
    for candidate in candidates:
        path = root / candidate
        if path.is_file():
            return path
    return newest_csv(root, broad_patterns, since)


def looks_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def validate_csv(path: Path, expected_state_cols: int = EXPECTED_STATE_COLS) -> tuple[int, int, bool]:
    """
    Returns (data_rows, column_count, has_header).

    Accepts either:
      - 276 columns: raw state values
      - 277+ columns: frame/time columns plus 276 state values
    """
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            raise ValueError(f"{path} is empty")

        first = [x.strip() for x in first]
        has_header = not all(looks_numeric(x) for x in first if x != "")
        col_count = len(first)
        data_rows = 0

        if has_header:
            if col_count < expected_state_cols:
                raise ValueError(
                    f"{path} header has {col_count} columns; expected at least {expected_state_cols} state columns"
                )
        else:
            data_rows += 1
            if col_count < expected_state_cols:
                raise ValueError(
                    f"{path} first data row has {col_count} columns; expected at least {expected_state_cols} state columns"
                )

        for row_number, row in enumerate(reader, start=2):
            if not row or all(cell.strip() == "" for cell in row):
                continue
            data_rows += 1
            if len(row) < expected_state_cols:
                raise ValueError(
                    f"{path} row {row_number} has {len(row)} columns; expected at least {expected_state_cols}"
                )

    if data_rows == 0:
        raise ValueError(f"{path} has no data rows")
    return data_rows, col_count, has_header


def standardise_csv(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    print(f"[ok] {source} -> {destination}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and standardise Milestone 4 vector output CSV files.")
    parser.add_argument("--lkf-cmd", default="./lkf_vector", help="Command that runs the vectorised LKF program")
    parser.add_argument("--ekf-cmd", default="./ekf_vector", help="Command that runs the vectorised EKF program")
    parser.add_argument("--lkf-source", default=None, help="Known path to LKF CSV if the program already generated it")
    parser.add_argument("--ekf-source", default=None, help="Known path to EKF CSV if the program already generated it")
    parser.add_argument("--out-dir", default="output", help="Directory where final CSVs should be written")
    parser.add_argument("--no-run", action="store_true", help="Do not run executables; only copy/validate existing CSVs")
    parser.add_argument("--root", default=".", help="Project root directory")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    final_lkf = out_dir / "lkf_vector_output.csv"
    final_ekf = out_dir / "ekf_vector_output.csv"

    started = time.time()

    try:
        if not args.no_run:
            run_command(args.lkf_cmd, "LKF vector")
            run_command(args.ekf_cmd, "EKF vector")

        if args.lkf_source:
            lkf_source = root / args.lkf_source
        else:
            lkf_source = find_csv(
                root,
                LKF_CANDIDATES,
                ["**/*lkf*vector*.csv", "**/*lkf*output*.csv", "**/lkf*.csv"],
                started,
            )

        if args.ekf_source:
            ekf_source = root / args.ekf_source
        else:
            ekf_source = find_csv(
                root,
                EKF_CANDIDATES,
                ["**/*ekf*vector*.csv", "**/*ekf*output*.csv", "**/ekf*.csv"],
                started,
            )

        if lkf_source is None or not lkf_source.is_file():
            raise FileNotFoundError(
                "Could not find LKF output CSV. Pass it explicitly with --lkf-source path/to/file.csv"
            )
        if ekf_source is None or not ekf_source.is_file():
            raise FileNotFoundError(
                "Could not find EKF output CSV. Pass it explicitly with --ekf-source path/to/file.csv"
            )

        standardise_csv(lkf_source, final_lkf)
        standardise_csv(ekf_source, final_ekf)

        for label, path in [("LKF", final_lkf), ("EKF", final_ekf)]:
            rows, cols, has_header = validate_csv(path)
            print(f"[validate] {label}: rows={rows}, columns={cols}, header={has_header}, file={path}")

        print("\nDone. Final Milestone 4 CSV files:")
        print(f"  {final_lkf}")
        print(f"  {final_ekf}")
        return 0

    except Exception as exc:
        print(f"\n[error] {exc}", file=sys.stderr)
        print("\nExpected final files:", file=sys.stderr)
        print(f"  {final_lkf}", file=sys.stderr)
        print(f"  {final_ekf}", file=sys.stderr)
        print("\nCommon fix:", file=sys.stderr)
        print("  1. make clean && make", file=sys.stderr)
        print("  2. run your vector executables manually", file=sys.stderr)
        print("  3. re-run this script with --no-run --lkf-source ... --ekf-source ...", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
