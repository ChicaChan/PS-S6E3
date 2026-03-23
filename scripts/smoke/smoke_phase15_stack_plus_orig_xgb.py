# Usage:
# python scripts/smoke/smoke_phase15_stack_plus_orig_xgb.py \
#   --max-train-rows 4000 \
#   --max-test-rows 1500

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


SOURCE_MODELS = [
    {
        "src_oof_path": "kaggle_dataset/phase14_stack_pipeline_inputs/phase13_oof_phase13_best.csv",
        "src_submission_path": "kaggle_dataset/phase14_stack_pipeline_inputs/phase13_submission_phase13_best.csv",
        "dst_oof_name": "phase13_oof_phase13_best.csv",
        "dst_submission_name": "phase13_submission_phase13_best.csv",
    },
    {
        "src_oof_path": "kaggle_dataset/phase14_stack_pipeline_inputs/phase10_oof_stack_best.csv",
        "src_submission_path": "kaggle_dataset/phase14_stack_pipeline_inputs/phase10_submission_stack_best.csv",
        "dst_oof_name": "phase10_oof_stack_best.csv",
        "dst_submission_name": "phase10_submission_stack_best.csv",
    },
    {
        "src_oof_path": "kaggle_dataset/phase14_stack_pipeline_inputs/phase8_oof_cat.csv",
        "src_submission_path": "kaggle_dataset/phase14_stack_pipeline_inputs/phase8_submission_cat.csv",
        "dst_oof_name": "phase8_oof_cat.csv",
        "dst_submission_name": "phase8_submission_cat.csv",
    },
    {
        "src_oof_path": "kaggle_dataset/phase14_stack_pipeline_inputs/phase9_oof_realmlp.csv",
        "src_submission_path": "kaggle_dataset/phase14_stack_pipeline_inputs/phase9_submission_realmlp.csv",
        "dst_oof_name": "phase9_oof_realmlp.csv",
        "dst_submission_name": "phase9_submission_realmlp.csv",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal local smoke test for phase15 stack plus orig xgb.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=4000)
    parser.add_argument("--max-test-rows", type=int, default=1500)
    parser.add_argument(
        "--phase15-output-dir",
        type=Path,
        default=Path(".artifacts/smoke_phase15_orig_fe_xgb/output"),
    )
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
        output_dir / "oof_stack_pipeline_best.csv",
        output_dir / "submission_stack_pipeline_best.csv",
        output_dir / "candidate_summary.json",
        output_dir / "stack_pipeline_report.json",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts: {missing}")


def ensure_phase15_inputs(phase15_output_dir: Path, input_root: Path) -> None:
    oof_src = phase15_output_dir / "oof_phase15_orig_fe_xgb.csv"
    sub_src = phase15_output_dir / "submission_phase15_orig_fe_xgb.csv"
    if not oof_src.exists() or not sub_src.exists():
        raise FileNotFoundError("phase15 smoke output is missing. Run smoke_phase15_orig_fe_xgb first.")
    shutil.copy2(oof_src, input_root / "phase15_oof_orig_fe_xgb.csv")
    shutil.copy2(sub_src, input_root / "phase15_submission_orig_fe_xgb.csv")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    smoke_root = project_root / ".artifacts" / "smoke_phase15_stack"
    input_root = smoke_root / "input"
    output_root = smoke_root / "output"
    input_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    base_oof_df = pd.read_csv(project_root / SOURCE_MODELS[0]["src_oof_path"])
    base_sub_df = pd.read_csv(project_root / SOURCE_MODELS[0]["src_submission_path"])
    train_ids, test_ids = sample_ids(
        oof_df=base_oof_df,
        sub_df=base_sub_df,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
        seed=args.seed,
    )

    for model in SOURCE_MODELS:
        slice_and_save(
            src_oof_path=project_root / model["src_oof_path"],
            src_sub_path=project_root / model["src_submission_path"],
            train_ids=train_ids,
            test_ids=test_ids,
            dst_oof_path=input_root / model["dst_oof_name"],
            dst_sub_path=input_root / model["dst_submission_name"],
        )

    phase15_output_dir = (project_root / args.phase15_output_dir).resolve()
    ensure_phase15_inputs(phase15_output_dir=phase15_output_dir, input_root=input_root)

    script_path = project_root / "kaggle_kernel" / "phase15_stack_plus_orig_xgb" / "stack_pipeline_search.py"
    config_path = project_root / "kaggle_kernel" / "phase15_stack_plus_orig_xgb" / "stack_pipeline_config_smoke.json"
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--config-path",
            str(config_path),
            "--output-dir",
            str(output_root),
        ],
        check=True,
        cwd=project_root,
    )

    assert_artifacts(output_root)
    print("Phase15 stack plus orig xgb smoke test passed.")


if __name__ == "__main__":
    main()
