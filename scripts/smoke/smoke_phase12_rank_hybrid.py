# Usage:
# python scripts/smoke/smoke_phase12_rank_hybrid.py \
#   --max-train-rows 4000 \
#   --max-test-rows 1500

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


REFERENCE_BLEND = {
    "oof_path": "kaggle_kernel/phase7_blend_oof/output_phase9_realmlp_candidates/oof_blend_opt.csv",
    "submission_path": "kaggle_kernel/phase7_blend_oof/output_phase9_realmlp_candidates/submission_blend_opt.csv",
}

STACK_BEST = {
    "oof_path": "kaggle_kernel/phase10_stack_oof/output_v4/phase10_stack_v1/oof_stack_best.csv",
    "submission_path": "kaggle_kernel/phase10_stack_oof/output_v4/phase10_stack_v1/submission_stack_best.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal local smoke test for phase12 rank hybrid pipeline.")
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
        output_dir / "oof_hybrid_best.csv",
        output_dir / "submission_hybrid_best.csv",
        output_dir / "candidate_summary.json",
        output_dir / "hybrid_report.json",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts: {missing}")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    smoke_root = project_root / ".artifacts" / "smoke_phase12_rank_hybrid"
    input_root = smoke_root / "input"
    output_root = smoke_root / "output"
    input_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    ref_oof_df = pd.read_csv(project_root / REFERENCE_BLEND["oof_path"])
    ref_sub_df = pd.read_csv(project_root / REFERENCE_BLEND["submission_path"])
    train_ids, test_ids = sample_ids(
        oof_df=ref_oof_df,
        sub_df=ref_sub_df,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
        seed=args.seed,
    )

    slice_and_save(
        src_oof_path=project_root / REFERENCE_BLEND["oof_path"],
        src_sub_path=project_root / REFERENCE_BLEND["submission_path"],
        train_ids=train_ids,
        test_ids=test_ids,
        dst_oof_path=input_root / "ref_oof_blend_opt.csv",
        dst_sub_path=input_root / "ref_submission_blend_opt.csv",
    )
    slice_and_save(
        src_oof_path=project_root / STACK_BEST["oof_path"],
        src_sub_path=project_root / STACK_BEST["submission_path"],
        train_ids=train_ids,
        test_ids=test_ids,
        dst_oof_path=input_root / "phase10_oof_stack_best.csv",
        dst_sub_path=input_root / "phase10_submission_stack_best.csv",
    )

    script_path = project_root / "kaggle_kernel" / "phase12_rank_hybrid" / "hybrid_search.py"
    config_path = project_root / "kaggle_kernel" / "phase12_rank_hybrid" / "hybrid_config_smoke.json"
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
    print("Phase12 rank hybrid smoke test passed.")


if __name__ == "__main__":
    main()
