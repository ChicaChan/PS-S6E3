# Usage:
# python src/local/smoke_test_pipeline.py \
#   --train-path train.csv \
#   --test-path test.csv \
#   --sample-submission-path sample_submission.csv \
#   --id-col id \
#   --target-col Churn \
#   --max-train-rows 4000 \
#   --max-test-rows 1500

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def build_small_sample(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_sub_df: pd.DataFrame,
    id_col: str,
    target_col: str,
    max_train_rows: int,
    max_test_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if max_train_rows < len(train_df):
        train_small, _ = train_test_split(
            train_df,
            train_size=max_train_rows,
            random_state=seed,
            stratify=train_df[target_col],
        )
        train_small = train_small.reset_index(drop=True)
    else:
        train_small = train_df.copy()

    if max_test_rows < len(test_df):
        test_small = test_df.sample(n=max_test_rows, random_state=seed).reset_index(drop=True)
    else:
        test_small = test_df.copy()

    sample_small = (
        sample_sub_df.set_index(id_col)
        .loc[test_small[id_col].values]
        .reset_index()
    )
    return train_small, test_small, sample_small


def assert_artifacts(output_dir: Path) -> None:
    expected = [
        output_dir / "submission.csv",
        output_dir / "oof_predictions.csv",
        output_dir / "cv_metrics.json",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal local smoke test for PS-S6E3 pipeline.")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--sample-submission-path", type=Path, required=True)
    parser.add_argument("--id-col", type=str, default="id")
    parser.add_argument("--target-col", type=str, default="Churn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=4000)
    parser.add_argument("--max-test-rows", type=int, default=1500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    smoke_root = project_root / ".artifacts" / "smoke"
    input_dir = smoke_root / "input"
    output_dir = smoke_root / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    sample_sub_df = pd.read_csv(args.sample_submission_path)

    train_small, test_small, sample_small = build_small_sample(
        train_df=train_df,
        test_df=test_df,
        sample_sub_df=sample_sub_df,
        id_col=args.id_col,
        target_col=args.target_col,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
        seed=args.seed,
    )

    train_small_path = input_dir / "train.csv"
    test_small_path = input_dir / "test.csv"
    sample_small_path = input_dir / "sample_submission.csv"
    config_path = input_dir / "smoke_config.json"

    train_small.to_csv(train_small_path, index=False)
    test_small.to_csv(test_small_path, index=False)
    sample_small.to_csv(sample_small_path, index=False)

    smoke_config = {
        "target_col": args.target_col,
        "id_col": args.id_col,
        "positive_label": "Yes",
        "negative_label": "No",
        "seed": args.seed,
        "n_folds": 2,
        "inner_folds": 2,
        "use_pseudo_label": False,
        "pseudo_label_threshold": 0.995,
        "min_pseudo_label_count": 50,
        "xgb_params": {
            "n_estimators": 120,
            "learning_rate": 0.1,
            "max_depth": 4,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "gamma": 0.0,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "early_stopping_rounds": 20,
            "enable_categorical": True,
            "tree_method": "hist",
            "device": "cpu",
            "verbosity": 0,
            "n_jobs": 1
        }
    }
    config_path.write_text(json.dumps(smoke_config, ensure_ascii=False, indent=2), encoding="utf-8")

    trainer_script = project_root / "src" / "remote" / "train_baseline_xgb_te.py"
    validator_script = project_root / "src" / "local" / "validate_submission.py"

    train_cmd = [
        sys.executable,
        str(trainer_script),
        "--train-path",
        str(train_small_path),
        "--test-path",
        str(test_small_path),
        "--sample-submission-path",
        str(sample_small_path),
        "--config-path",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]
    train_result = subprocess.run(train_cmd, text=True, capture_output=True, cwd=project_root)
    print(train_result.stdout)
    if train_result.returncode != 0:
        print(train_result.stderr)
        raise SystemExit(train_result.returncode)

    assert_artifacts(output_dir)

    validate_cmd = [
        sys.executable,
        str(validator_script),
        "--submission-path",
        str(output_dir / "submission.csv"),
        "--sample-submission-path",
        str(sample_small_path),
        "--id-col",
        args.id_col,
        "--target-col",
        args.target_col,
    ]
    validate_result = subprocess.run(validate_cmd, text=True, capture_output=True, cwd=project_root)
    print(validate_result.stdout)
    if validate_result.returncode != 0:
        print(validate_result.stderr)
        raise SystemExit(validate_result.returncode)

    print("Smoke test completed successfully.")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
