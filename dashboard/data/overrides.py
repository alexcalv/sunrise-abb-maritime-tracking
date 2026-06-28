"""User ID-override storage (sidecar JSON).

The dashboard never edits the original MOT files. Manual ID corrections are
stored here and applied as an overlay at read time. Each override remaps
``old_id -> new_id`` for every row at ``from_frame`` and later (the realistic
"the tracker swapped this id from frame N onward" case).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_STORE_PATH = Path(__file__).resolve().parent / "overrides.json"


def _load_store() -> dict[str, list[dict[str, int]]]:
    if not _STORE_PATH.exists():
        return {}
    try:
        return json.loads(_STORE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_store(store: dict[str, list[dict[str, int]]]) -> None:
    _STORE_PATH.write_text(json.dumps(store, indent=2), encoding="utf-8")


def get_overrides(run_key: str) -> list[dict[str, int]]:
    """Return the list of override entries for a run."""
    return _load_store().get(run_key, [])


def add_override(run_key: str, old_id: int, new_id: int, from_frame: int) -> None:
    """Persist a new ID override for a run."""
    store = _load_store()
    entries = store.setdefault(run_key, [])
    entries.append({"old_id": int(old_id), "new_id": int(new_id), "from_frame": int(from_frame)})
    _save_store(store)


def remove_override(run_key: str, index: int) -> None:
    """Remove a single override entry (by position) for a run."""
    store = _load_store()
    entries = store.get(run_key)
    if entries and 0 <= index < len(entries):
        entries.pop(index)
        if entries:
            store[run_key] = entries
        else:
            del store[run_key]
        _save_store(store)


def clear_overrides(run_key: str) -> None:
    """Remove all overrides for a run."""
    store = _load_store()
    if run_key in store:
        del store[run_key]
        _save_store(store)


def find_collisions(rows: list[dict[str, Any]], new_id: int, from_frame: int) -> list[int]:
    """Frames at/after ``from_frame`` where ``new_id`` already exists.

    Used to warn that applying a correction would put two boxes carrying the
    same id on the same frame. Returns the sorted list of offending frames
    (empty when the correction is unambiguous).
    """
    target = int(new_id)
    start = int(from_frame)
    return sorted(
        {row["frame"] for row in rows if row["id"] == target and row["frame"] >= start}
    )


def apply_overrides(rows: list[dict[str, Any]], run_key: str) -> list[dict[str, Any]]:
    """Return rows with ID overrides applied (originals are left untouched)."""
    entries = get_overrides(run_key)
    if not entries:
        return rows

    patched: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        for entry in entries:
            if new_row["id"] == entry["old_id"] and new_row["frame"] >= entry["from_frame"]:
                new_row["id"] = entry["new_id"]
        patched.append(new_row)
    return patched
