from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_app_cli_main():
    repo_root = Path(__file__).resolve().parents[1]
    app_root = repo_root / "app"
    if str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))

    cli_path = app_root / "scripts" / "cli.py"
    spec = importlib.util.spec_from_file_location("_maritime_app_cli", cli_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load app CLI from {cli_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def main() -> None:
    _load_app_cli_main()()


if __name__ == "__main__":
    main()
