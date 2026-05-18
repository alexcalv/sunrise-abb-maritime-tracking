from __future__ import annotations

from typing import Any

from colreg import ColregConfig, EncounterType, build_vessel_state, classify_encounter, predict_colreg_state


def _obs(track_id: int, points: list[tuple[int, float, float]]) -> list[dict[str, object]]:
    return [
        {
            "frame": frame,
            "id": track_id,
            "bbox": [x - 5.0, y - 5.0, 10.0, 10.0],
            "confidence": 1.0,
            "class_id": 0,
            "visibility": 1.0,
        }
        for frame, x, y in points
    ]


def _state(track_id: int, points: list[tuple[int, float, float]]):
    return build_vessel_state(_obs(track_id, points), track_id=track_id, min_observations=3)


def _case(name: str, own_points: Any, other_points: Any, expected: EncounterType) -> dict[str, object]:
    result = classify_encounter(_state(1, own_points), _state(2, other_points), ColregConfig())
    return {
        "case": name,
        "passed": result.encounter_type == expected,
        "expected": expected.value,
        "actual": result.encounter_type.value,
        "confidence": result.confidence,
        "reason": result.reason,
    }


def run_checks() -> dict[str, object]:
    """Run deterministic synthetic checks for diagnostic COLREG utilities."""

    cases = [
        _case(
            "head_on",
            [(1, 0.0, 0.0), (2, 10.0, 0.0), (3, 20.0, 0.0)],
            [(1, 100.0, 0.0), (2, 90.0, 0.0), (3, 80.0, 0.0)],
            EncounterType.HEAD_ON,
        ),
        _case(
            "crossing_give_way",
            [(1, 0.0, 0.0), (2, 10.0, 0.0), (3, 20.0, 0.0)],
            [(1, 50.0, 70.0), (2, 50.0, 60.0), (3, 50.0, 50.0)],
            EncounterType.CROSSING_GIVE_WAY,
        ),
        _case(
            "crossing_stand_on",
            [(1, 0.0, 0.0), (2, 10.0, 0.0), (3, 20.0, 0.0)],
            [(1, 50.0, -70.0), (2, 50.0, -60.0), (3, 50.0, -50.0)],
            EncounterType.CROSSING_STAND_ON,
        ),
        _case(
            "overtaking",
            [(1, 0.0, 0.0), (2, 20.0, 0.0), (3, 45.0, 0.0)],
            [(1, 30.0, 0.0), (2, 40.0, 0.0), (3, 55.0, 0.0)],
            EncounterType.OVERTAKING,
        ),
    ]

    unknown = classify_encounter(
        build_vessel_state(_obs(1, [(1, 0.0, 0.0)]), track_id=1, min_observations=1),
        build_vessel_state(_obs(2, [(1, 10.0, 10.0)]), track_id=2, min_observations=1),
        ColregConfig(),
    )
    cases.append(
        {
            "case": "unknown_insufficient_motion",
            "passed": unknown.encounter_type == EncounterType.UNKNOWN,
            "expected": EncounterType.UNKNOWN.value,
            "actual": unknown.encounter_type.value,
            "confidence": unknown.confidence,
            "reason": unknown.reason,
        }
    )

    prior_checks = []
    head_on_state = _state(1, [(1, 0.0, 0.0), (2, 10.0, 0.0), (3, 20.0, 0.0)])
    if head_on_state is not None:
        prediction = predict_colreg_state(head_on_state, EncounterType.HEAD_ON, ColregConfig())
        prior_checks.append(
            {
                "case": "head_on_prior_turns_starboard",
                "passed": prediction.predicted.heading_deg > head_on_state.heading_deg,
                "predicted_heading": prediction.predicted.heading_deg,
                "reason": prediction.reason,
            }
        )

    return {
        "passed": all(bool(case["passed"]) for case in cases) and all(bool(case["passed"]) for case in prior_checks),
        "cases": cases,
        "motion_priors": prior_checks,
    }
