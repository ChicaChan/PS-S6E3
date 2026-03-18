# Usage:
# python src/local/validate_submission.py \
#   --submission-path outputs/baseline/submission.csv \
#   --sample-submission-path sample_submission.csv \
#   --id-col id \
#   --target-col Churn

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def validate_schema(df: pd.DataFrame, id_col: str, target_col: str) -> list[str]:
    errors: list[str] = []
    expected_columns = [id_col, target_col]
    if list(df.columns) != expected_columns:
        errors.append(f"Columns mismatch: expected {expected_columns}, got {list(df.columns)}")
    return errors


def validate_range(df: pd.DataFrame, target_col: str) -> list[str]:
    errors: list[str] = []
    if df[target_col].isna().any():
        errors.append(f"Target column '{target_col}' contains NaN values.")
        return errors
    arr = df[target_col].to_numpy(dtype=np.float64)
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        errors.append(f"Target column '{target_col}' must be in [0, 1].")
    return errors


def validate_row_alignment(sub_df: pd.DataFrame, sample_df: pd.DataFrame, id_col: str) -> list[str]:
    errors: list[str] = []
    if len(sub_df) != len(sample_df):
        errors.append(f"Row count mismatch: submission={len(sub_df)} sample={len(sample_df)}")
        return errors
    if not sub_df[id_col].equals(sample_df[id_col]):
        errors.append(f"ID order/content mismatch for column '{id_col}'.")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate submission file for PS-S6E3.")
    parser.add_argument("--submission-path", type=Path, required=True)
    parser.add_argument("--sample-submission-path", type=Path, required=True)
    parser.add_argument("--id-col", type=str, default="id")
    parser.add_argument("--target-col", type=str, default="Churn")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sub_df = pd.read_csv(args.submission_path)
    sample_df = pd.read_csv(args.sample_submission_path)

    errors: list[str] = []
    errors.extend(validate_schema(sub_df, args.id_col, args.target_col))
    if args.target_col in sub_df.columns:
        errors.extend(validate_range(sub_df, args.target_col))
    if args.id_col in sub_df.columns and args.id_col in sample_df.columns:
        errors.extend(validate_row_alignment(sub_df, sample_df, args.id_col))

    if errors:
        print("Submission validation failed:")
        for item in errors:
            print(f"- {item}")
        raise SystemExit(1)

    print("Submission validation passed.")
    print(f"Rows: {len(sub_df)}")


if __name__ == "__main__":
    main()
