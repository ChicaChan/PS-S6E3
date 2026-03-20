# Usage:
# python scripts/smoke/smoke_phase9_realmlp_tabm_diverse.py \
#   --train-path train.csv \
#   --test-path test.csv \
#   --sample-submission-path sample_submission.csv \
#   --max-train-rows 2500 \
#   --max-test-rows 1000

from __future__ import annotations

import argparse
import importlib.util
import json
import py_compile
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

    sample_small = sample_sub_df.set_index(id_col).loc[test_small[id_col].values].reset_index()
    return train_small, test_small, sample_small


def run_cmd(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, text=True, capture_output=True, cwd=cwd)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit(result.returncode)


def validate_submission(
    project_root: Path,
    submission_path: Path,
    sample_path: Path,
    id_col: str,
    target_col: str,
) -> None:
    validator = project_root / "src" / "local" / "validate_submission.py"
    cmd = [
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
    ]
    run_cmd(cmd, cwd=project_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for phase9 RealMLP/TabM pipeline.")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--sample-submission-path", type=Path, required=True)
    parser.add_argument("--id-col", type=str, default="id")
    parser.add_argument("--target-col", type=str, default="Churn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=2500)
    parser.add_argument("--max-test-rows", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "kaggle_kernel" / "phase9_realmlp_tabm_diverse" / "train_realmlp_tabm_diverse.py"

    py_compile.compile(str(script_path), doraise=True)
    print("Py-compile passed for phase9 RealMLP/TabM pipeline.")

    if importlib.util.find_spec("pytabkit") is None:
        print("Skip runtime smoke for phase9: pytabkit is not installed locally.")
        return

    smoke_root = project_root / ".artifacts" / "smoke_phase9"
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
    config_path = input_dir / "smoke_phase9_config.json"

    train_small.to_csv(train_small_path, index=False)
    test_small.to_csv(test_small_path, index=False)
    sample_small.to_csv(sample_small_path, index=False)

    config = {
        "target_col": args.target_col,
        "id_col": args.id_col,
        "positive_label": "Yes",
        "negative_label": "No",
        "seed": args.seed,
        "n_folds": 2,
        "inner_folds": 2,
        "max_te_categories": 500,
        "orig_data_path": "",
        "top_cats_for_ngram": [
            "Contract",
            "InternetService",
            "PaymentMethod",
            "OnlineSecurity",
            "TechSupport",
            "PaperlessBilling"
        ],
        "orig_single_mode": "all_categorical",
        "enable_orig_cross": True,
        "enable_pctrank_orig": True,
        "enable_pctrank_churn_gap": True,
        "conditional_rank_group_cols": [
            "InternetService",
            "Contract"
        ],
        "conditional_rank_value_col": "TotalCharges",
        "pip_install_on_kaggle": False,
        "enable_realmlp": True,
        "enable_tabm": False,
        "realmlp_params": {
            "random_state": args.seed,
            "n_cv": 1,
            "n_refit": 0,
            "n_epochs": 8,
            "batch_size": 256,
            "hidden_sizes": [64, 64],
            "lr": 0.04,
            "use_ls": False,
            "val_metric_name": "cross_entropy",
            "verbosity": 1
        },
        "tabm_params": {
            "random_state": args.seed,
            "n_cv": 1,
            "n_refit": 0,
            "n_epochs": 8,
            "batch_size": 256,
            "verbosity": 1
        }
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(script_path),
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
    run_cmd(cmd, cwd=project_root)
    validate_submission(
        project_root=project_root,
        submission_path=output_dir / "submission.csv",
        sample_path=sample_small_path,
        id_col=args.id_col,
        target_col=args.target_col,
    )


if __name__ == "__main__":
    main()
