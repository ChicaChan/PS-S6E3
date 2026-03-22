# Usage:
# python scripts/smoke/smoke_phase13_hybrid_plus_realmlp.py \
#   --max-train-rows 4000 \
#   --max-test-rows 1500

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


BASE_HYBRID = {
    "oof_path": "kaggle_kernel/phase12_rank_hybrid/output_v2/phase12_rank_hybrid_v1/oof_hybrid_best.csv",
    "submission_path": "kaggle_kernel/phase12_rank_hybrid/output_v2/phase12_rank_hybrid_v1/submission_hybrid_best.csv",
}

REALMLP = {
    "oof_path": "kaggle_kernel/phase9_realmlp_tabm_diverse/output_v2/oof_realmlp.csv",
    "submission_path": "kaggle_kernel/phase9_realmlp_tabm_diverse/output_v2/submission_realmlp.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal local smoke test for phase13 hybrid plus RealMLP pipeline.")
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
        output_dir / "oof_phase13_best.csv",
        output_dir / "submission_phase13_best.csv",
        output_dir / "candidate_summary.json",
        output_dir / "phase13_report.json",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts: {missing}")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    smoke_root = project_root / ".artifacts" / "smoke_phase13_hybrid_plus_realmlp"
    input_root = smoke_root / "input"
    output_root = smoke_root / "output"
    input_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    base_oof_df = pd.read_csv(project_root / BASE_HYBRID["oof_path"])
    base_sub_df = pd.read_csv(project_root / BASE_HYBRID["submission_path"])
    train_ids, test_ids = sample_ids(
        oof_df=base_oof_df,
        sub_df=base_sub_df,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
        seed=args.seed,
    )

    slice_and_save(
        src_oof_path=project_root / BASE_HYBRID["oof_path"],
        src_sub_path=project_root / BASE_HYBRID["submission_path"],
        train_ids=train_ids,
        test_ids=test_ids,
        dst_oof_path=input_root / "phase12_oof_hybrid_best.csv",
        dst_sub_path=input_root / "phase12_submission_hybrid_best.csv",
    )
    slice_and_save(
        src_oof_path=project_root / REALMLP["oof_path"],
        src_sub_path=project_root / REALMLP["submission_path"],
        train_ids=train_ids,
        test_ids=test_ids,
        dst_oof_path=input_root / "phase9_oof_realmlp.csv",
        dst_sub_path=input_root / "phase9_submission_realmlp.csv",
    )

    script_path = project_root / "kaggle_kernel" / "phase13_hybrid_plus_realmlp" / "hybrid_plus_realmlp_search.py"
    config_path = project_root / "kaggle_kernel" / "phase13_hybrid_plus_realmlp" / "phase13_config_smoke.json"
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
    print("Phase13 hybrid plus RealMLP smoke test passed.")


if __name__ == "__main__":
    main()
