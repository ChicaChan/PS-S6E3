# Usage:
# python scripts/smoke/smoke_phase16_catboost_orig_transfer.py \
#   --max-train-rows 4000 \
#   --max-test-rows 1500

from __future__ import annotations

import argparse
import importlib.util
import py_compile
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal local smoke test for phase16_catboost_orig_transfer.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=4000)
    parser.add_argument("--max-test-rows", type=int, default=1500)
    return parser.parse_args()


def sample_train(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows >= len(df):
        return df.copy()
    sampled, _ = train_test_split(df, train_size=max_rows, random_state=seed, stratify=df["Churn"])
    return sampled.reset_index(drop=True)


def sample_test(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows >= len(df):
        return df.copy()
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def run_cmd(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, text=True, capture_output=True, cwd=cwd)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit(result.returncode)


def validate_submission(project_root: Path, submission_path: Path, sample_path: Path, id_col: str, target_col: str) -> None:
    validator = project_root / "src" / "local" / "validate_submission.py"
    run_cmd(
        [
            sys.executable,
            str(validator),
            "--submission-path",
            str(submission_path),
            "--sample-submission-path",
            str(sample_path),
            "--id-col",
            id_col,
            "--target-col",
            target_col,
        ],
        cwd=project_root,
    )


def assert_artifacts(output_dir: Path) -> None:
    expected = [
        output_dir / "oof_phase16_catboost_orig_transfer.csv",
        output_dir / "submission_phase16_catboost_orig_transfer.csv",
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
    script_path = project_root / "kaggle_kernel" / "phase16_catboost_orig_transfer" / "train_phase16_catboost_orig_transfer.py"
    config_path = project_root / "kaggle_kernel" / "phase16_catboost_orig_transfer" / "config_phase16_catboost_orig_transfer_smoke.json"

    py_compile.compile(str(script_path), doraise=True)
    print("Py-compile passed for phase16 CatBoost orig transfer pipeline.")

    if importlib.util.find_spec("catboost") is None:
        print("Skip runtime smoke for phase16: catboost is not installed locally.")
        return

    smoke_root = project_root / ".artifacts" / "smoke_phase16_catboost_orig_transfer"
    input_root = smoke_root / "input"
    output_root = smoke_root / "output"
    input_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(project_root / "train.csv")
    test_df = pd.read_csv(project_root / "test.csv")
    sample_submission = pd.read_csv(project_root / "sample_submission.csv")
    sampled_train = sample_train(train_df, max_rows=args.max_train_rows, seed=args.seed)
    sampled_test = sample_test(test_df, max_rows=args.max_test_rows, seed=args.seed)
    sample_small = sample_submission.set_index("id").loc[sampled_test["id"].values].reset_index()

    sampled_train.to_csv(input_root / "train.csv", index=False)
    sampled_test.to_csv(input_root / "test.csv", index=False)
    sample_small.to_csv(input_root / "sample_submission.csv", index=False)

    run_cmd(
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
        cwd=project_root,
    )

    assert_artifacts(output_root)
    validate_submission(project_root, output_root / "submission_phase16_catboost_orig_transfer.csv", input_root / "sample_submission.csv", "id", "Churn")
    print("Phase16 CatBoost orig transfer smoke test passed.")


if __name__ == "__main__":
    main()
