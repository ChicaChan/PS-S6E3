# Usage:
# python scripts/smoke/smoke_phase15_orig_fe_xgb.py \
#   --max-train-rows 4000 \
#   --max-test-rows 1500

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal local smoke test for phase15_orig_fe_xgb.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=4000)
    parser.add_argument("--max-test-rows", type=int, default=1500)
    return parser.parse_args()


def sample_train(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows >= len(df):
        return df.copy()
    sampled, _ = train_test_split(
        df,
        train_size=max_rows,
        random_state=seed,
        stratify=df["Churn"],
    )
    return sampled.reset_index(drop=True)


def sample_test(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows >= len(df):
        return df.copy()
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def assert_artifacts(output_dir: Path) -> None:
    expected = [
        output_dir / "oof_phase15_orig_fe_xgb.csv",
        output_dir / "submission_phase15_orig_fe_xgb.csv",
        output_dir / "cv_metrics.json",
        output_dir / "run_summary.json",
        output_dir / "feature_importance.csv",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts: {missing}")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    smoke_root = project_root / ".artifacts" / "smoke_phase15_orig_fe_xgb"
    input_root = smoke_root / "input"
    output_root = smoke_root / "output"
    input_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(project_root / "train.csv")
    test_df = pd.read_csv(project_root / "test.csv")
    sampled_train = sample_train(train_df, max_rows=args.max_train_rows, seed=args.seed)
    sampled_test = sample_test(test_df, max_rows=args.max_test_rows, seed=args.seed)
    sample_submission = pd.DataFrame({
        "id": sampled_test["id"].values,
        "Churn": 0.0,
    })

    sampled_train.to_csv(input_root / "train.csv", index=False)
    sampled_test.to_csv(input_root / "test.csv", index=False)
    sample_submission.to_csv(input_root / "sample_submission.csv", index=False)

    script_path = project_root / "kaggle_kernel" / "phase15_orig_fe_xgb" / "train_phase15_orig_fe_xgb.py"
    config_path = project_root / "kaggle_kernel" / "phase15_orig_fe_xgb" / "config_phase15_orig_fe_xgb_smoke.json"
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--train-path",
            str(input_root / "train.csv"),
            "--test-path",
            str(input_root / "test.csv"),
            "--sample-submission-path",
            str(input_root / "sample_submission.csv"),
            "--config-path",
            str(config_path),
            "--output-dir",
            str(output_root),
        ],
        check=True,
        cwd=project_root,
    )

    assert_artifacts(output_root)
    print("Phase15 orig FE XGB smoke test passed.")


if __name__ == "__main__":
    main()
