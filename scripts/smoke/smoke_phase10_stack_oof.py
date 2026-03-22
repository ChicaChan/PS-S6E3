# Usage:
# python scripts/smoke/smoke_phase10_stack_oof.py \
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


SMOKE_MODELS = [
    {
        "name": "phase8_cat_v1",
        "oof_path": "kaggle_kernel/phase8_catboost_strong/output_v2/oof_cat.csv",
        "submission_path": "kaggle_kernel/phase8_catboost_strong/output_v2/submission_cat.csv",
    },
    {
        "name": "phase9_realmlp_v2",
        "oof_path": "kaggle_kernel/phase9_realmlp_tabm_diverse/output_v2/oof_realmlp.csv",
        "submission_path": "kaggle_kernel/phase9_realmlp_tabm_diverse/output_v2/submission_realmlp.csv",
    },
]

REFERENCE_BLEND = {
    "name": "phase9_blend_best_v1",
    "oof_path": "kaggle_kernel/phase7_blend_oof/output_phase9_realmlp_candidates/oof_blend_opt.csv",
    "submission_path": "kaggle_kernel/phase7_blend_oof/output_phase9_realmlp_candidates/submission_blend_opt.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal local smoke test for phase10 stack OOF pipeline.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=4000)
    parser.add_argument("--max-test-rows", type=int, default=1500)
    return parser.parse_args()


def sample_ids(
    oof_df: pd.DataFrame,
    sub_df: pd.DataFrame,
    max_train_rows: int,
    max_test_rows: int,
    seed: int,
) -> tuple[pd.Series, pd.Series]:
    if max_train_rows < len(oof_df):
        sampled_train, _ = train_test_split(
            oof_df[["id", "target_binary"]],
            train_size=max_train_rows,
            random_state=seed,
            stratify=oof_df["target_binary"],
        )
        train_ids = sampled_train["id"].reset_index(drop=True)
    else:
        train_ids = oof_df["id"].reset_index(drop=True)

    if max_test_rows < len(sub_df):
        test_ids = sub_df["id"].sample(n=max_test_rows, random_state=seed).reset_index(drop=True)
    else:
        test_ids = sub_df["id"].reset_index(drop=True)

    return train_ids, test_ids


def slice_and_save(
    src_oof_path: Path,
    src_sub_path: Path,
    train_ids: pd.Series,
    test_ids: pd.Series,
    dst_oof_path: Path,
    dst_sub_path: Path,
) -> None:
    oof_df = pd.read_csv(src_oof_path).set_index("id").loc[train_ids.values].reset_index()
    sub_df = pd.read_csv(src_sub_path).set_index("id").loc[test_ids.values].reset_index()
    dst_oof_path.parent.mkdir(parents=True, exist_ok=True)
    dst_sub_path.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(dst_oof_path, index=False)
    sub_df.to_csv(dst_sub_path, index=False)


def assert_artifacts(output_dir: Path) -> None:
    expected = [
        output_dir / "oof_stack_best.csv",
        output_dir / "submission_stack_best.csv",
        output_dir / "candidate_summary.json",
        output_dir / "stack_report.json",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts: {missing}")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    smoke_root = project_root / ".artifacts" / "smoke_phase10_stack"
    input_root = smoke_root / "input"
    output_root = smoke_root / "output"
    input_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    base_oof = pd.read_csv(project_root / SMOKE_MODELS[0]["oof_path"])
    base_sub = pd.read_csv(project_root / SMOKE_MODELS[0]["submission_path"])
    train_ids, test_ids = sample_ids(
        oof_df=base_oof,
        sub_df=base_sub,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
        seed=args.seed,
    )

    smoke_models: list[dict[str, str]] = []
    for model in SMOKE_MODELS:
        dst_oof = input_root / model["name"] / "oof.csv"
        dst_sub = input_root / model["name"] / "submission.csv"
        slice_and_save(
            src_oof_path=project_root / model["oof_path"],
            src_sub_path=project_root / model["submission_path"],
            train_ids=train_ids,
            test_ids=test_ids,
            dst_oof_path=dst_oof,
            dst_sub_path=dst_sub,
        )
        smoke_models.append(
            {
                "name": model["name"],
                "oof_path": str(dst_oof),
                "submission_path": str(dst_sub),
            }
        )

    ref_oof = input_root / REFERENCE_BLEND["name"] / "oof.csv"
    ref_sub = input_root / REFERENCE_BLEND["name"] / "submission.csv"
    slice_and_save(
        src_oof_path=project_root / REFERENCE_BLEND["oof_path"],
        src_sub_path=project_root / REFERENCE_BLEND["submission_path"],
        train_ids=train_ids,
        test_ids=test_ids,
        dst_oof_path=ref_oof,
        dst_sub_path=ref_sub,
    )

    smoke_config = {
        "seed": args.seed,
        "id_col": "id",
        "target_col": "Churn",
        "n_meta_folds": 3,
        "min_models": 2,
        "allow_missing_models": False,
        "models": smoke_models,
        "reference_blend": {
            "name": REFERENCE_BLEND["name"],
            "oof_path": str(ref_oof),
            "submission_path": str(ref_sub),
        },
        "candidate_sets": [
            {
                "name": "available_core",
                "models": [item["name"] for item in smoke_models],
            }
        ],
        "feature_modes": ["raw", "raw_rank"],
        "meta_models": [
            {
                "name": "logreg_l2_c1",
                "type": "logreg",
                "params": {
                    "C": 1.0,
                    "max_iter": 1000,
                    "solver": "lbfgs",
                },
            },
            {
                "name": "ridge_a1",
                "type": "ridge",
                "params": {
                    "alpha": 1.0,
                },
            },
        ],
    }

    smoke_config_path = input_root / "stack_config_smoke_runtime.json"
    smoke_config_path.write_text(json.dumps(smoke_config, ensure_ascii=False, indent=2), encoding="utf-8")

    script_path = project_root / "kaggle_kernel" / "phase10_stack_oof" / "stack_oof_search.py"
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--config-path",
            str(smoke_config_path),
            "--output-dir",
            str(output_root),
        ],
        check=True,
        cwd=project_root,
    )

    assert_artifacts(output_root)
    print("Phase10 stack smoke test passed.")


if __name__ == "__main__":
    main()
