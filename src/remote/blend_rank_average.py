# Usage:
# python src/remote/blend_rank_average.py \
#   --sample-submission-path sample_submission.csv \
#   --submission-paths outputs/model_a.csv outputs/model_b.csv \
#   --method weighted_rank \
#   --weights 0.6 0.4 \
#   --output-path outputs/submission_blend.csv

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def infer_prediction_column(df: pd.DataFrame, target_col: str) -> str:
    if target_col in df.columns:
        return target_col
    if len(df.columns) < 2:
        raise ValueError("Submission file must contain at least 2 columns.")
    return df.columns[1]


def normalize_weights(weights: list[float], n: int) -> np.ndarray:
    if not weights:
        return np.full(n, 1.0 / n, dtype=np.float64)
    if len(weights) != n:
        raise ValueError(f"weights length must equal submission count ({n}).")
    arr = np.asarray(weights, dtype=np.float64)
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("weights sum must be positive.")
    return arr / total


def rank_percentiles(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=np.float64)


def load_prediction_files(
    sample_df: pd.DataFrame,
    submission_paths: list[Path],
    id_col: str,
    target_col: str,
) -> list[np.ndarray]:
    if id_col not in sample_df.columns:
        raise ValueError(f"Sample submission missing id column: '{id_col}'.")
    if sample_df[id_col].duplicated().any():
        raise ValueError("Sample submission contains duplicate ids.")

    preds: list[np.ndarray] = []
    ids = sample_df[id_col].values
    for path in submission_paths:
        df = pd.read_csv(path)
        if id_col not in df.columns:
            raise ValueError(f"Submission '{path}' missing id column: '{id_col}'.")
        if df[id_col].duplicated().any():
            raise ValueError(f"Submission '{path}' contains duplicate ids.")

        pred_col = infer_prediction_column(df, target_col)
        indexed = df.set_index(id_col)
        missing_ids = pd.Index(ids).difference(indexed.index)
        if len(missing_ids) > 0:
            preview = ", ".join(str(x) for x in missing_ids[:5].tolist())
            raise ValueError(
                f"Submission '{path}' is missing {len(missing_ids)} ids from sample. "
                f"Examples: {preview}"
            )
        aligned = indexed.loc[ids, pred_col].to_numpy(dtype=np.float64)
        if np.isnan(aligned).any():
            raise ValueError(f"Submission '{path}' contains NaN predictions after alignment.")
        preds.append(aligned)
    return preds


def blend_predictions(
    prediction_list: list[np.ndarray], method: str, weights: np.ndarray
) -> np.ndarray:
    matrix = np.vstack(prediction_list)
    if method == "mean":
        blended = np.average(matrix, axis=0, weights=weights)
    elif method == "weighted_rank":
        rank_matrix = np.vstack([rank_percentiles(row) for row in matrix])
        blended = np.average(rank_matrix, axis=0, weights=weights)
    else:
        raise ValueError(f"Unsupported method: {method}")
    return np.clip(blended, 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend PS-S6E3 submission files.")
    parser.add_argument("--sample-submission-path", type=Path, required=True)
    parser.add_argument("--submission-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--method", choices=["mean", "weighted_rank"], default="weighted_rank")
    parser.add_argument("--weights", type=float, nargs="*", default=[])
    parser.add_argument("--id-col", type=str, default="id")
    parser.add_argument("--target-col", type=str, default="Churn")
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_df = pd.read_csv(args.sample_submission_path)
    weights = normalize_weights(args.weights, len(args.submission_paths))
    preds = load_prediction_files(
        sample_df=sample_df,
        submission_paths=args.submission_paths,
        id_col=args.id_col,
        target_col=args.target_col,
    )
    blended = blend_predictions(prediction_list=preds, method=args.method, weights=weights)

    out_df = sample_df[[args.id_col]].copy()
    out_df[args.target_col] = blended.astype(np.float32)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_path, index=False)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
