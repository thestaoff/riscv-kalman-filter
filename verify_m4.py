#!/usr/bin/env python3
"""
Milestone 4 numerical verification script.

Compares M4 vector outputs against M3 scalar outputs for LKF and EKF.
Produces the required verification tables:
  1) Average absolute error per joint
  2) Average absolute error per state group/component
  3) Global max/min absolute error
  4) Direct M3 vs M4 comparison table

Optional: if M2 reference CSVs are supplied, it also reports M3-vs-M2 and
M4-vs-M2 errors to show vectorisation did not degrade scalar accuracy.

Expected state layout per row:
  23 joints × 12 states = 276 numeric values
  joint state order: [px, vx, ax, jx, py, vy, ay, jy, pz, vz, az, jz]

The CSV may have a header. A frame/time/index column is allowed and will be
ignored. If more than 276 numeric state-like columns remain, use
--state-column-mode first276 or last276.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

N_JOINTS = 23
N_STATE_PER_JOINT = 12
N_FULL = N_JOINTS * N_STATE_PER_JOINT
TOL_DEFAULT = 1e-9

JOINT_NAMES = [
    "pelvis", "L5", "L3", "T12", "T8", "neck", "head",
    "shoulderRight", "upperArmRight", "forearmRight", "handRight",
    "shoulderLeft", "upperArmLeft", "forearmLeft", "handLeft",
    "upperLegRight", "lowerLegRight", "footRight", "toeRight",
    "upperLegLeft", "lowerLegLeft", "footLeft", "toeLeft",
]

STATE_NAMES = ["px", "vx", "ax", "jx", "py", "vy", "ay", "jy", "pz", "vz", "az", "jz"]
STATE_GROUPS = {
    "position": [0, 4, 8],
    "velocity": [1, 5, 9],
    "acceleration": [2, 6, 10],
    "jerk": [3, 7, 11],
}

COMMON_NON_STATE_COLUMNS = {
    "frame", "frames", "t", "time", "timestamp", "index", "idx", "k", "step",
}


def _read_csv_attempt(path: Path, header) -> pd.DataFrame:
    return pd.read_csv(path, header=header)


def load_state_csv(path: str | Path, state_column_mode: str = "auto") -> np.ndarray:
    """Load a CSV and return a float64 array of shape (frames, 276)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    # First try normal header. If not enough numeric data, try header=None.
    last_error: Optional[Exception] = None
    for header in (0, None):
        try:
            df = _read_csv_attempt(path, header=header)
            arr = select_state_columns(df, path, state_column_mode)
            return arr
        except Exception as exc:  # try alternate header before failing
            last_error = exc
            continue

    assert last_error is not None
    raise last_error


def select_state_columns(df: pd.DataFrame, path: Path, state_column_mode: str) -> np.ndarray:
    # Drop obvious non-state columns by name if headers exist.
    df_work = df.copy()
    named_cols_to_drop = []
    for col in df_work.columns:
        name = str(col).strip().lower()
        if name in COMMON_NON_STATE_COLUMNS:
            named_cols_to_drop.append(col)
    if named_cols_to_drop:
        df_work = df_work.drop(columns=named_cols_to_drop)

    # Convert each column to numeric where possible.
    numeric_df = df_work.apply(pd.to_numeric, errors="coerce")

    # Keep columns that are mostly numeric. Header=None may create all numeric columns.
    numeric_ratio = numeric_df.notna().mean(axis=0)
    numeric_df = numeric_df.loc[:, numeric_ratio > 0.95]

    # Drop fully empty rows caused by repeated text headers, etc.
    numeric_df = numeric_df.dropna(axis=0, how="all")

    ncols = numeric_df.shape[1]
    if ncols == N_FULL:
        state_df = numeric_df
    elif ncols > N_FULL:
        if state_column_mode == "first276":
            state_df = numeric_df.iloc[:, :N_FULL]
        elif state_column_mode == "last276":
            state_df = numeric_df.iloc[:, -N_FULL:]
        else:
            # Auto: if exactly one obvious non-state numeric column remains at front, drop it.
            # This catches header=None reading a frame index column.
            if ncols == N_FULL + 1:
                state_df = numeric_df.iloc[:, 1:]
            else:
                raise ValueError(
                    f"{path} has {ncols} numeric columns after dropping common frame/time columns; "
                    f"expected 276. Re-run with --state-column-mode first276 or last276."
                )
    else:
        raise ValueError(
            f"{path} has only {ncols} numeric columns; expected 276 state values per row."
        )

    arr = state_df.to_numpy(dtype=np.float64)
    if np.isnan(arr).any():
        bad_rows = np.where(np.isnan(arr).any(axis=1))[0][:10]
        raise ValueError(f"{path} contains nonnumeric/NaN state values in rows {bad_rows.tolist()}.")
    if arr.ndim != 2 or arr.shape[1] != N_FULL:
        raise ValueError(f"{path} loaded shape {arr.shape}, expected (*, 276).")
    return arr


def validate_pair(name: str, scalar: np.ndarray, vector: np.ndarray) -> None:
    if scalar.shape != vector.shape:
        raise ValueError(f"{name}: scalar shape {scalar.shape} != vector shape {vector.shape}")
    if scalar.shape[1] != N_FULL:
        raise ValueError(f"{name}: expected 276 state columns, got {scalar.shape[1]}")


def per_joint_table(err: np.ndarray) -> pd.DataFrame:
    e = err.reshape(err.shape[0], N_JOINTS, N_STATE_PER_JOINT)
    return pd.DataFrame({
        "joint_index": np.arange(N_JOINTS),
        "joint": JOINT_NAMES,
        "avg_abs_error": e.mean(axis=(0, 2)),
        "max_abs_error": e.max(axis=(0, 2)),
        "min_abs_error": e.min(axis=(0, 2)),
    })


def per_state_12_table(err: np.ndarray) -> pd.DataFrame:
    e = err.reshape(err.shape[0], N_JOINTS, N_STATE_PER_JOINT)
    rows = []
    for s_idx, state in enumerate(STATE_NAMES):
        vals = e[:, :, s_idx]
        rows.append({
            "state_index_within_joint": s_idx,
            "state": state,
            "avg_abs_error": vals.mean(),
            "max_abs_error": vals.max(),
            "min_abs_error": vals.min(),
        })
    return pd.DataFrame(rows)


def per_state_group_table(err: np.ndarray) -> pd.DataFrame:
    e = err.reshape(err.shape[0], N_JOINTS, N_STATE_PER_JOINT)
    rows = []
    for group, idxs in STATE_GROUPS.items():
        vals = e[:, :, idxs]
        rows.append({
            "state_group": group,
            "states_included": ",".join(STATE_NAMES[i] for i in idxs),
            "avg_abs_error": vals.mean(),
            "max_abs_error": vals.max(),
            "min_abs_error": vals.min(),
        })
    return pd.DataFrame(rows)


def global_summary_table(err: np.ndarray, tol: float) -> pd.DataFrame:
    failures = err > tol
    max_pos = np.unravel_index(np.argmax(err), err.shape)
    min_pos = np.unravel_index(np.argmin(err), err.shape)
    max_col = int(max_pos[1])
    min_col = int(min_pos[1])
    max_joint = max_col // N_STATE_PER_JOINT
    max_state = max_col % N_STATE_PER_JOINT
    min_joint = min_col // N_STATE_PER_JOINT
    min_state = min_col % N_STATE_PER_JOINT
    return pd.DataFrame([{
        "frames": err.shape[0],
        "state_columns": err.shape[1],
        "total_values": int(err.size),
        "tolerance": tol,
        "passed": bool(not failures.any()),
        "num_failures": int(failures.sum()),
        "mean_abs_error": float(err.mean()),
        "max_abs_error": float(err.max()),
        "max_error_frame": int(max_pos[0]),
        "max_error_joint": JOINT_NAMES[max_joint],
        "max_error_state": STATE_NAMES[max_state],
        "min_abs_error": float(err.min()),
        "min_error_frame": int(min_pos[0]),
        "min_error_joint": JOINT_NAMES[min_joint],
        "min_error_state": STATE_NAMES[min_state],
    }])


def direct_m3_m4_table(err: np.ndarray) -> pd.DataFrame:
    """Direct M3 scalar vs M4 vector table by joint and state group."""
    e = err.reshape(err.shape[0], N_JOINTS, N_STATE_PER_JOINT)
    rows = []
    for j_idx, joint in enumerate(JOINT_NAMES):
        for group, idxs in STATE_GROUPS.items():
            vals = e[:, j_idx, idxs]
            rows.append({
                "joint_index": j_idx,
                "joint": joint,
                "state_group": group,
                "avg_abs_m4_vector_minus_m3_scalar": vals.mean(),
                "max_abs_m4_vector_minus_m3_scalar": vals.max(),
            })
    return pd.DataFrame(rows)


def m2_reference_comparison(m2: np.ndarray, m3_scalar: np.ndarray, m4_vector: np.ndarray) -> pd.DataFrame:
    """Optional table: M3-vs-M2 vs M4-vs-M2 by joint and state group."""
    if m2.shape != m3_scalar.shape or m2.shape != m4_vector.shape:
        raise ValueError(
            f"M2, M3, M4 shapes must match. Got M2 {m2.shape}, M3 {m3_scalar.shape}, M4 {m4_vector.shape}."
        )
    e3 = np.abs(m3_scalar - m2).reshape(m2.shape[0], N_JOINTS, N_STATE_PER_JOINT)
    e4 = np.abs(m4_vector - m2).reshape(m2.shape[0], N_JOINTS, N_STATE_PER_JOINT)
    rows = []
    for j_idx, joint in enumerate(JOINT_NAMES):
        for group, idxs in STATE_GROUPS.items():
            m3_vals = e3[:, j_idx, idxs]
            m4_vals = e4[:, j_idx, idxs]
            rows.append({
                "joint_index": j_idx,
                "joint": joint,
                "state_group": group,
                "m3_scalar_avg_abs_error_vs_m2": m3_vals.mean(),
                "m4_vector_avg_abs_error_vs_m2": m4_vals.mean(),
                "delta_m4_minus_m3": m4_vals.mean() - m3_vals.mean(),
            })
    return pd.DataFrame(rows)


def df_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    # Keep report tables readable: scientific notation for floats.
    return df.to_latex(
        index=False,
        escape=False,
        float_format=lambda x: f"{x:.3e}",
        caption=caption,
        label=label,
    )


def write_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, float_format="%.17e")


def write_markdown_summary(out_dir: Path, results: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    md = ["# Milestone 4 Numerical Verification Summary\n"]
    for filt, tables in results.items():
        md.append(f"\n## {filt.upper()}\n")
        global_df = tables["global_summary"]
        md.append(global_df.to_markdown(index=False))
        md.append("\n\n### Average absolute error per joint\n")
        md.append(tables["per_joint"].to_markdown(index=False))
        md.append("\n\n### Average absolute error per state group\n")
        md.append(tables["per_state_group"].to_markdown(index=False))
    (out_dir / "verification_summary.md").write_text("\n".join(md), encoding="utf-8")


def verify_filter(
    filt: str,
    scalar_path: str | Path,
    vector_path: str | Path,
    out_dir: Path,
    tol: float,
    state_column_mode: str,
    m2_reference_path: Optional[str | Path] = None,
) -> Tuple[Dict[str, pd.DataFrame], bool]:
    scalar = load_state_csv(scalar_path, state_column_mode)
    vector = load_state_csv(vector_path, state_column_mode)
    validate_pair(filt, scalar, vector)

    err = np.abs(vector - scalar)
    tables: Dict[str, pd.DataFrame] = {
        "per_joint": per_joint_table(err),
        "per_state_group": per_state_group_table(err),
        "per_state_12component": per_state_12_table(err),
        "global_summary": global_summary_table(err, tol),
        "direct_m3_m4": direct_m3_m4_table(err),
    }

    if m2_reference_path:
        m2 = load_state_csv(m2_reference_path, state_column_mode)
        tables["m2_reference_comparison"] = m2_reference_comparison(m2, scalar, vector)

    filt_dir = out_dir / filt.lower()
    filt_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        write_table(df, filt_dir / f"{filt.lower()}_{name}.csv")
        (filt_dir / f"{filt.lower()}_{name}.tex").write_text(
            df_to_latex(
                df,
                caption=f"{filt.upper()} {name.replace('_', ' ')}",
                label=f"tab:{filt.lower()}_{name}",
            ),
            encoding="utf-8",
        )

    passed = bool(tables["global_summary"].iloc[0]["passed"])
    return tables, passed


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Milestone 4 vector CSVs against Milestone 3 scalar CSVs.")
    parser.add_argument("--m3-lkf", required=True, help="M3 scalar LKF output CSV")
    parser.add_argument("--m4-lkf", required=True, help="M4 vector LKF output CSV")
    parser.add_argument("--m3-ekf", required=True, help="M3 scalar EKF output CSV")
    parser.add_argument("--m4-ekf", required=True, help="M4 vector EKF output CSV")
    parser.add_argument("--m2-lkf", default=None, help="Optional M2 LKF reference output CSV")
    parser.add_argument("--m2-ekf", default=None, help="Optional M2 EKF reference output CSV")
    parser.add_argument("--out-dir", default="m4_verification_tables", help="Output directory for tables")
    parser.add_argument("--tol", type=float, default=TOL_DEFAULT, help="Absolute tolerance; default 1e-9")
    parser.add_argument(
        "--state-column-mode",
        choices=["auto", "first276", "last276"],
        default="auto",
        help="How to select 276 state columns when CSV has extra numeric columns.",
    )
    parser.add_argument(
        "--no-fail-on-tolerance",
        action="store_true",
        help="Return exit code 0 even if tolerance fails.",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Dict[str, pd.DataFrame]] = {}
    try:
        lkf_tables, lkf_passed = verify_filter(
            "lkf", args.m3_lkf, args.m4_lkf, out_dir, args.tol, args.state_column_mode, args.m2_lkf
        )
        ekf_tables, ekf_passed = verify_filter(
            "ekf", args.m3_ekf, args.m4_ekf, out_dir, args.tol, args.state_column_mode, args.m2_ekf
        )
        all_results["lkf"] = lkf_tables
        all_results["ekf"] = ekf_tables
        write_markdown_summary(out_dir, all_results)

        print("\nMilestone 4 numerical verification complete.")
        for filt, tables in all_results.items():
            summary = tables["global_summary"].iloc[0]
            print(
                f"{filt.upper()}: passed={summary['passed']} | "
                f"max={summary['max_abs_error']:.17e} | "
                f"mean={summary['mean_abs_error']:.17e} | "
                f"failures={int(summary['num_failures'])}"
            )
        print(f"\nTables written to: {out_dir.resolve()}")

        if (not lkf_passed or not ekf_passed) and not args.no_fail_on_tolerance:
            return 2
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
