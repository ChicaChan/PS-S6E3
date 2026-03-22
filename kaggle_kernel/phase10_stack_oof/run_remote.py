# Usage:
# python run_remote.py

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    subprocess.run(
        [
            sys.executable,
            str(root / "stack_oof_search.py"),
            "--config-path",
            str(root / "stack_config_remote_v1.json"),
            "--output-dir",
            "/kaggle/working/phase10_stack_v1",
        ],
        check=True,
        cwd=root,
    )


if __name__ == "__main__":
    main()
