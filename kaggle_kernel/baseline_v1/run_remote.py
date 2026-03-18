from pathlib import Path
from train_baseline_xgb_te import run_pipeline


def main() -> None:
    run_pipeline(
        train_path=Path("/kaggle/input/playground-series-s6e3/train.csv"),
        test_path=Path("/kaggle/input/playground-series-s6e3/test.csv"),
        sample_submission_path=Path("/kaggle/input/playground-series-s6e3/sample_submission.csv"),
        config_path=Path("config_baseline.json"),
        output_dir=Path("/kaggle/working"),
    )


if __name__ == "__main__":
    main()
