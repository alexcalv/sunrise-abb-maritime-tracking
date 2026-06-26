#!/usr/bin/env python3
"""
Drive the MaritimeSim Unity project from Python to render one video plus
matching MOT/JSON ground truth per camera defined in a JSON config.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from glob import glob
from pathlib import Path


# Repo root = the folder that contains this tools/ directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = REPO_ROOT / "MaritimeSim2"
PROJECT_UNITY_VERSION = "2022.3.62f3"
EXECUTE_METHOD = "GeneratorEntry.Run"


def find_unity(explicit: str | None = None, version: str = PROJECT_UNITY_VERSION) -> str:
    """Locate a Unity editor executable. Raises FileNotFoundError if none found."""
    candidates: list[str] = []

    if explicit:
        candidates.append(explicit)
    if os.environ.get("UNITY_PATH"):
        candidates.append(os.environ["UNITY_PATH"])

    if sys.platform.startswith("win"):
        hubs = [r"C:\Program Files\Unity\Hub\Editor"]
        secondary = os.path.join(os.environ.get("APPDATA", ""), "UnityHub", "secondaryInstallPath.json")
        try:
            extra = json.loads(Path(secondary).read_text(encoding="utf-8"))
            if isinstance(extra, str) and extra.strip():
                hubs.append(extra.strip())
        except (OSError, ValueError):
            pass
        for hub in hubs:
            candidates.append(rf"{hub}\{version}\Editor\Unity.exe")
            candidates += sorted(glob(rf"{hub}\*\Editor\Unity.exe"), reverse=True)
    elif sys.platform == "darwin":
        hub = "/Applications/Unity/Hub/Editor"
        candidates.append(f"{hub}/{version}/Unity.app/Contents/MacOS/Unity")
        candidates += sorted(glob(f"{hub}/*/Unity.app/Contents/MacOS/Unity"), reverse=True)
    else:
        hub = os.path.expanduser("~/Unity/Hub/Editor")
        candidates.append(f"{hub}/{version}/Editor/Unity")
        candidates += sorted(glob(f"{hub}/*/Editor/Unity"), reverse=True)

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate

    raise FileNotFoundError(
        "Could not find a Unity editor. Pass --unity <path> or set UNITY_PATH. "
        f"Looked for version {version} in the Unity Hub default location."
    )


def camera_names(config_path: Path) -> list[str]:
    """Return camera output names from a simulation config."""
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cams = cfg.get("cameras") or []
    names = []
    for index, camera in enumerate(cams):
        names.append(camera.get("name") or f"camera_{index}")
    if not names:
        raise ValueError(f"Config {config_path} defines no cameras.")
    return names


def collect_outputs(config: str | os.PathLike, out_dir: str | os.PathLike) -> dict[str, dict[str, str]]:
    """Build the expected per-camera output map for an existing generation directory."""
    config_path = Path(config).resolve()
    output_root = Path(out_dir).resolve()
    return {
        name: {
            "video": str(output_root / f"{name}.mp4"),
            "mot": str(output_root / f"{name}.txt"),
            "json": str(output_root / f"{name}.json"),
        }
        for name in camera_names(config_path)
    }


def validate_outputs(outputs: dict[str, dict[str, str]], require_video: bool = True) -> list[str]:
    """Return human-readable missing/empty output errors."""
    errors: list[str] = []
    for camera, files in outputs.items():
        for key in ("mot", "json"):
            path = Path(files[key])
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"{camera}: missing/empty {key}: {path}")
        if require_video:
            video = Path(files["video"])
            if not video.is_file() or video.stat().st_size == 0:
                errors.append(f"{camera}: missing/empty video: {video}")
    return errors


def generate_videos(
    config: str | os.PathLike,
    out_dir: str | os.PathLike | None = None,
    unity: str | None = None,
    project: str | os.PathLike | None = None,
    timeout: float | None = 1800,
    batchmode: bool = False,
    verbose: bool = False,
    quiet: bool = False,
) -> dict[str, dict[str, str]]:
    """
    Render the scene described by ``config`` and return produced per-camera files.

    Returns ``{"<camera>": {"video": <mp4>, "mot": <txt>, "json": <json>}}``.
    """
    config_path = Path(config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    project_root = Path(project).resolve() if project else PROJECT_ROOT
    if not project_root.exists():
        raise FileNotFoundError(f"Unity project root not found: {project_root}")

    if out_dir is None:
        out_dir = project_root / "Recordings" / config_path.stem
    output_root = Path(out_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    unity_exe = find_unity(unity)
    log_file = output_root / "unity.log"

    cmd = [unity_exe]
    if batchmode:
        cmd.append("-batchmode")
        if not quiet:
            print(
                "[generate_videos] WARNING: -batchmode often yields empty videos "
                "(Recorder capture does not tick headless).",
                flush=True,
            )
    cmd += [
        "-projectPath",
        str(project_root),
        "-executeMethod",
        EXECUTE_METHOD,
        "-simConfig",
        str(config_path),
        "-simOut",
        str(output_root),
        "-logFile",
        str(log_file),
    ]
    if verbose:
        cmd.append("-simVerbose")

    if not quiet:
        print("[generate_videos] Unity:", unity_exe)
        print("[generate_videos] project:", project_root)
        print("[generate_videos] config:", config_path)
        print("[generate_videos] out:", output_root)
        print("[generate_videos] running...", flush=True)

    try:
        proc = subprocess.run(cmd, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Unity timed out after {timeout}s. See log: {log_file}") from exc

    if proc.returncode != 0:
        raise RuntimeError(
            f"Unity exited with code {proc.returncode}.\n"
            f"--- tail of {log_file} ---\n{_tail(log_file)}"
        )

    outputs = collect_outputs(config_path, output_root)
    missing = validate_outputs(outputs)
    if missing:
        raise RuntimeError(
            "Unity finished but expected outputs are missing/empty:\n  "
            + "\n  ".join(missing)
            + f"\n--- tail of {log_file} ---\n{_tail(log_file)}"
        )

    if not quiet:
        print(f"[generate_videos] OK: {len(outputs)} camera(s) -> {output_root}")
    return outputs


def _tail(path: Path, lines: int = 40) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(text[-lines:])
    except OSError:
        return f"(could not read {path})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render MaritimeSim videos from a JSON config.")
    parser.add_argument("config", help="Path to the scenario JSON config.")
    parser.add_argument("--out", help="Output directory (default: MaritimeSim2/Recordings/<config name>).")
    parser.add_argument("--unity", help="Path to Unity executable (default: auto-detect / UNITY_PATH).")
    parser.add_argument("--project", help="Unity project root (default: MaritimeSim2 in this repo).")
    parser.add_argument("--timeout", type=float, default=1800, help="Seconds before giving up.")
    parser.add_argument(
        "--batchmode",
        action="store_true",
        help="Run Unity with -batchmode (not recommended: Recorder often writes empty videos).",
    )
    parser.add_argument("--verbose", action="store_true", help="Log Recorder capture progress to unity.log.")
    args = parser.parse_args(argv)

    try:
        outputs = generate_videos(
            args.config,
            out_dir=args.out,
            unity=args.unity,
            project=args.project,
            timeout=args.timeout,
            batchmode=args.batchmode,
            verbose=args.verbose,
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean message to the shell
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
