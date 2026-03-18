# Usage:
# python scripts/smoke/smoke_phase5_phase6.py \
#   --train-path train.csv \
#   --test-path test.csv \
#   --sample-submission-path sample_submission.csv \
#   --max-train-rows 2500 \
#   --max-test-rows 1000

from __future__ import annotations

import argparse
import importlib.util
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
    parser = argparse.ArgumentParser(description="Smoke test for phase5/phase6/phase7 pipelines.")
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

    smoke_root = project_root / ".artifacts" / "smoke_phase5_6"
    input_dir = smoke_root / "input"
    out_phase5 = smoke_root / "phase5_out"
    out_phase6 = smoke_root / "phase6_out"
    out_blend = smoke_root / "blend_out"

    input_dir.mkdir(parents=True, exist_ok=True)
    out_phase5.mkdir(parents=True, exist_ok=True)
    out_phase6.mkdir(parents=True, exist_ok=True)
    out_blend.mkdir(parents=True, exist_ok=True)

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

    train_small.to_csv(train_small_path, index=False)
    test_small.to_csv(test_small_path, index=False)
    sample_small.to_csv(sample_small_path, index=False)

    # Phase5 smoke config
    phase5_cfg = {
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
        "max_te_categories": 500,
        "top_cats_for_ngram": [
            "Contract",
            "InternetService",
            "PaymentMethod",
            "OnlineSecurity",
            "TechSupport",
            "PaperlessBilling"
        ],
        "orig_data_path": "",
        "xgb_params": {
            "n_estimators": 160,
            "learning_rate": 0.08,
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
    phase5_cfg_path = input_dir / "smoke_phase5_config.json"
    phase5_cfg_path.write_text(json.dumps(phase5_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    phase5_script = project_root / "kaggle_kernel" / "phase5_xgb_advanced" / "train_xgb_advanced.py"
    phase5_cmd = [
        sys.executable,
        str(phase5_script),
        "--train-path",
        str(train_small_path),
        "--test-path",
        str(test_small_path),
        "--sample-submission-path",
        str(sample_small_path),
        "--config-path",
        str(phase5_cfg_path),
        "--output-dir",
        str(out_phase5),
    ]
    run_cmd(phase5_cmd, cwd=project_root)
    validate_submission(
        project_root=project_root,
        submission_path=out_phase5 / "submission.csv",
        sample_path=sample_small_path,
        id_col=args.id_col,
        target_col=args.target_col,
    )

    has_lgbm = importlib.util.find_spec("lightgbm") is not None
    has_cat = importlib.util.find_spec("catboost") is not None

    phase6_ran = False
    if has_lgbm or has_cat:
        phase6_cfg = {
            "target_col": args.target_col,
            "id_col": args.id_col,
            "positive_label": "Yes",
            "negative_label": "No",
            "seed": args.seed,
            "n_folds": 2,
            "inner_folds": 2,
            "max_te_categories": 500,
            "orig_data_path": "",
            "enable_lgbm": bool(has_lgbm),
            "enable_catboost": bool(has_cat),
            "lgbm_params": {
                "n_estimators": 220,
                "learning_rate": 0.06,
                "max_depth": 5,
                "num_leaves": 63,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "min_child_samples": 30,
                "objective": "binary",
                "metric": "auc",
                "random_state": args.seed,
                "verbose": -1
            },
            "cat_params": {
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "learning_rate": 0.08,
                "depth": 5,
                "l2_leaf_reg": 3.0,
                "n_estimators": 280,
                "random_seed": args.seed,
                "verbose": 0,
                "task_type": "CPU"
            }
        }
        phase6_cfg_path = input_dir / "smoke_phase6_config.json"
        phase6_cfg_path.write_text(json.dumps(phase6_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

        phase6_script = project_root / "kaggle_kernel" / "phase6_diverse_tree" / "train_lgbm_cat_diverse.py"
        phase6_cmd = [
            sys.executable,
            str(phase6_script),
            "--train-path",
            str(train_small_path),
            "--test-path",
            str(test_small_path),
            "--sample-submission-path",
            str(sample_small_path),
            "--config-path",
            str(phase6_cfg_path),
            "--output-dir",
            str(out_phase6),
        ]
        run_cmd(phase6_cmd, cwd=project_root)
        validate_submission(
            project_root=project_root,
            submission_path=out_phase6 / "submission.csv",
            sample_path=sample_small_path,
            id_col=args.id_col,
            target_col=args.target_col,
        )
        phase6_ran = True
    else:
        print("Skip phase6 in smoke test: neither lightgbm nor catboost is installed locally.")

    blend_models = [
        {
            "name": "phase5_xgb_advanced",
            "oof_path": str((out_phase5 / "oof_predictions.csv").resolve()),
            "submission_path": str((out_phase5 / "submission.csv").resolve()),
        }
    ]

    if phase6_ran:
        if (out_phase6 / "oof_lgbm.csv").exists() and (out_phase6 / "submission_lgbm.csv").exists():
            blend_models.append(
                {
                    "name": "phase6_lgbm",
                    "oof_path": str((out_phase6 / "oof_lgbm.csv").resolve()),
                    "submission_path": str((out_phase6 / "submission_lgbm.csv").resolve()),
                }
            )
        if (out_phase6 / "oof_cat.csv").exists() and (out_phase6 / "submission_cat.csv").exists():
            blend_models.append(
                {
                    "name": "phase6_cat",
                    "oof_path": str((out_phase6 / "oof_cat.csv").resolve()),
                    "submission_path": str((out_phase6 / "submission_cat.csv").resolve()),
                }
            )

    blend_cfg = {
        "seed": args.seed,
        "corr_threshold": 0.995,
        "n_trials": 300,
        "models": blend_models,
    }
    blend_cfg_path = input_dir / "smoke_blend_config.json"
    blend_cfg_path.write_text(json.dumps(blend_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    blend_script = project_root / "kaggle_kernel" / "phase7_blend_oof" / "blend_rank_oof_search.py"
    blend_cmd = [
        sys.executable,
        str(blend_script),
        "--config-path",
        str(blend_cfg_path),
        "--sample-submission-path",
        str(sample_small_path),
        "--output-dir",
        str(out_blend),
        "--id-col",
        args.id_col,
        "--target-col",
        args.target_col,
    ]
    run_cmd(blend_cmd, cwd=project_root)

    validate_submission(
        project_root=project_root,
        submission_path=out_blend / "submission_blend_opt.csv",
        sample_path=sample_small_path,
        id_col=args.id_col,
        target_col=args.target_col,
    )

    print("Smoke test completed successfully.")
    print(f"Phase5 artifacts: {out_phase5}")
    print(f"Phase6 artifacts: {out_phase6}")
    print(f"Blend artifacts: {out_blend}")


if __name__ == "__main__":
    main()
