# Usage:
# python scripts/prepare/prepare_phase15_stack_inputs.py \
#   --phase15-output-dir "kaggle_kernel/phase15_orig_fe_xgb/output_v1" \
#   --dest-dir "kaggle_dataset/phase15_stack_inputs"

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


DATASET_METADATA = {
    "title": "PS-S6E3 Phase15 Stack Inputs",
    "id": "chicachan/ps-s6e3-phase15-stack-inputs",
    "licenses": [{"name": "CC0-1.0"}],
}

PHASE14_REQUIRED_FILES = [
    "phase13_oof_phase13_best.csv",
    "phase13_submission_phase13_best.csv",
    "phase10_oof_stack_best.csv",
    "phase10_submission_stack_best.csv",
    "phase8_oof_cat.csv",
    "phase8_submission_cat.csv",
    "phase9_oof_realmlp.csv",
    "phase9_submission_realmlp.csv",
]


def infer_prediction_column(df: pd.DataFrame, preferred: str) -> str:
    candidates = [preferred, "oof_prediction", "prediction", "pred", "Churn"]
    for cand in candidates:
        if cand in df.columns:
            return cand
    if len(df.columns) < 2:
        raise ValueError("Prediction dataframe must contain at least 2 columns.")
    return df.columns[-1]


def normalize_oof_frame(path: Path, id_col: str, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if id_col not in df.columns:
        raise ValueError(f"OOF file missing id column: {path}")
    pred_col = infer_prediction_column(df, preferred="oof_prediction")
    target_candidates = ["target_binary", target_col, "target", "label", "y_true"]
    found_target = None
    for cand in target_candidates:
        if cand in df.columns:
            found_target = cand
            break
    if found_target is None:
        raise ValueError(f"OOF file missing target column: {path}")
    out = df[[id_col, found_target, pred_col]].copy()
    out.columns = [id_col, "target_binary", "oof_prediction"]
    return out


def normalize_submission_frame(path: Path, id_col: str, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if id_col not in df.columns:
        raise ValueError(f"Submission file missing id column: {path}")
    pred_col = infer_prediction_column(df, preferred=target_col)
    out = df[[id_col, pred_col]].copy()
    out.columns = [id_col, target_col]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare stack input dataset for phase15.")
    parser.add_argument("--phase14-input-dir", type=Path, default=Path("kaggle_dataset/phase14_stack_pipeline_inputs"))
    parser.add_argument("--phase15-output-dir", type=Path, required=True)
    parser.add_argument("--dest-dir", type=Path, default=Path("kaggle_dataset/phase15_stack_inputs"))
    parser.add_argument("--id-col", type=str, default="id")
    parser.add_argument("--target-col", type=str, default="Churn")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase14_input_dir = args.phase14_input_dir.resolve()
    phase15_output_dir = args.phase15_output_dir.resolve()
    dest_dir = args.dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    for filename in PHASE14_REQUIRED_FILES:
        src = phase14_input_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing required phase14 input file: {src}")
        shutil.copy2(src, dest_dir / src.name)

    phase15_oof_path = phase15_output_dir / "oof_phase15_orig_fe_xgb.csv"
    phase15_submission_path = phase15_output_dir / "submission_phase15_orig_fe_xgb.csv"
    if not phase15_oof_path.exists() or not phase15_submission_path.exists():
        raise FileNotFoundError("phase15 output directory is missing expected OOF or submission files.")

    phase15_oof = normalize_oof_frame(phase15_oof_path, id_col=args.id_col, target_col=args.target_col)
    phase15_sub = normalize_submission_frame(phase15_submission_path, id_col=args.id_col, target_col=args.target_col)
    phase15_oof.to_csv(dest_dir / "phase15_oof_orig_fe_xgb.csv", index=False)
    phase15_sub.to_csv(dest_dir / "phase15_submission_orig_fe_xgb.csv", index=False)

    (dest_dir / "dataset-metadata.json").write_text(
        json.dumps(DATASET_METADATA, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = {
        "phase14_input_dir": str(phase14_input_dir),
        "phase15_output_dir": str(phase15_output_dir),
        "generated_files": [
            *PHASE14_REQUIRED_FILES,
            "phase15_oof_orig_fe_xgb.csv",
            "phase15_submission_orig_fe_xgb.csv",
            "dataset-metadata.json",
            "phase15_stack_manifest.json",
        ],
    }
    (dest_dir / "phase15_stack_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"dest_dir": str(dest_dir), "status": "ok"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
