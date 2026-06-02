"""Draw tracking bounding boxes + ID labels onto a video frame."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from utils.colors import track_color


def draw_boxes(frame_rgb: np.ndarray, frame_rows: list[dict[str, Any]]) -> Image.Image:
    """Return a PIL image of the frame with one labelled box per vessel."""
    image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(image)

    for row in frame_rows:
        x, y, w, h = row["bbox"]
        tid = row["id"]
        color = track_color(tid)
        x1, y1, x2, y2 = x, y, x + w, y + h
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = f"#{tid}"
        text_y = max(0, y1 - 14)
        draw.rectangle([x1, text_y, x1 + 8 * len(label) + 6, text_y + 14], fill=color)
        draw.text((x1 + 3, text_y + 1), label, fill=(0, 0, 0))

    return image
