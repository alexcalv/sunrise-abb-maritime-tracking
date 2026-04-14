from __future__ import annotations

from pathlib import Path

from common.schemas import FrameResult


def persist_frame_result(base_dir: str, result: FrameResult) -> str:
    out_dir = Path(base_dir) / result.clip_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{result.frame_index:06d}.json"
    out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return str(out_path)
