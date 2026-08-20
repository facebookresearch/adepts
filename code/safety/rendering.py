"""Render predicted/target actions on screenshots for visual debugging."""

from __future__ import annotations

import math
import os

from PIL import Image as PILImage, ImageDraw


def _score_to_color(score: float) -> str:
    score = max(0.0, min(1.0, score))
    r = int(255 * (1.0 - score))
    g = int(255 * score)
    return f"#{r:02x}{g:02x}00"


def _extract_image_path_from_sample(sample: dict) -> str | None:
    for msg in reversed(sample.get("messages", [])):
        content = msg.get("content", {})
        if not isinstance(content, dict):
            continue
        if content.get("content_type") != "multimodal_text_message_content":
            continue
        for part in content.get("content", []):
            if isinstance(part, dict) and part.get("content_type") == "image_message_content":
                return part.get("image_path")
    return None


def _draw_bboxes(
    draw: ImageDraw.ImageDraw, boxes: list[dict], w: int, h: int,
    color: str = "lime", width: int = 3, label: str | None = None,
) -> None:
    for i, box in enumerate(boxes):
        points = box.get("points", [])
        if len(points) < 2:
            continue
        try:
            xs = [p["xNorm"] * w for p in points]
            ys = [p["yNorm"] * h for p in points]
        except (KeyError, TypeError):
            continue
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        if label:
            tag = label if len(boxes) == 1 else f"{label}[{i}]"
            draw.text((x1 + 2, max(y1 - 16, 0)), tag, fill=color)


def _draw_swipe(
    draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int,
    color: str = "red", width: int = 3,
) -> None:
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    al = 15
    for da in (30, -30):
        a = angle + math.radians(da)
        draw.line((x2, y2, int(x2 - al * math.cos(a)), int(y2 - al * math.sin(a))),
                  fill=color, width=width)
    draw.ellipse((x1 - 4, y1 - 4, x1 + 4, y1 + 4), fill=color)


def render_sample_image(sample, action_pred, reward, category, render_dir):
    """Render predicted/target actions on screenshot and save to render_dir."""
    from safety.safety_benchmark import TapAction, SwipeAction, InvalidAction

    image_path = _extract_image_path_from_sample(sample)
    if not image_path or not os.path.exists(image_path):
        return None

    grading = sample.get("grading", {})
    bboxes = grading.get("bounding_boxes") or []
    target = grading.get("target", {})
    reverse = grading.get("reverse", False)

    img = PILImage.open(image_path).convert("RGBA")
    w, h = img.size

    base_draw = ImageDraw.Draw(img)
    if bboxes:
        bbox_color = "cyan" if not reverse else "red"
        bbox_label = "target zone" if not reverse else "avoid zone"
        _draw_bboxes(base_draw, bboxes, w, h, color=bbox_color, label=bbox_label)

    target_fn = target.get("function_name", "")
    target_args = target.get("arguments", {})
    if target_fn in ("tap", "long_press") and "x" in target_args and "y" in target_args:
        target_overlay = PILImage.new("RGBA", img.size, (0, 0, 0, 0))
        td = ImageDraw.Draw(target_overlay)
        tx = int(float(target_args["x"]) / 1000 * w)
        ty = int(float(target_args["y"]) / 1000 * h)
        s = 14
        blue = (0, 100, 255, 220)
        td.line((tx - s, ty - s, tx + s, ty + s), fill=blue, width=4)
        td.line((tx - s, ty + s, tx + s, ty - s), fill=blue, width=4)
        td.text((tx + s + 4, ty - 8), "target", fill=blue)
        img = PILImage.alpha_composite(img, target_overlay)

    overlay = PILImage.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    if isinstance(action_pred, TapAction):
        px, py = int(action_pred.x * w), int(action_pred.y * h)
        if reward == 1.0:
            dot_color = (0, 220, 0, 200)
            dot_outer = (0, 220, 0, 80)
            label_text = "pred (correct)"
        else:
            dot_color = (255, 140, 0, 200)
            dot_outer = (255, 140, 0, 80)
            label_text = "pred (wrong)"
        overlay_draw.ellipse((px - 28, py - 28, px + 28, py + 28), fill=dot_outer)
        overlay_draw.ellipse((px - 10, py - 10, px + 10, py + 10), fill=dot_color)
        overlay_draw.text((px + 14, py - 8), label_text, fill=dot_color)
    elif isinstance(action_pred, SwipeAction):
        swipe_color = (0, 220, 0, 200) if reward == 1.0 else (255, 140, 0, 200)
        _draw_swipe(overlay_draw,
                     int(action_pred.x0 * w), int(action_pred.y0 * h),
                     int(action_pred.x1 * w), int(action_pred.y1 * h),
                     color=swipe_color)
    else:
        label = repr(action_pred) if not isinstance(action_pred, InvalidAction) else category
        overlay_draw.text((w // 2 - 80, h // 2), f"pred: {label}", fill=(255, 50, 50, 220))

    img = PILImage.alpha_composite(img, overlay)

    sample_id = sample.get("id", 0)
    score_str = "1" if reward == 1.0 else "0"
    filename = f"{str(sample_id).zfill(4)}_{category}_{score_str}.png"

    os.makedirs(render_dir, exist_ok=True)
    out_path = os.path.join(render_dir, filename)
    img.convert("RGB").save(out_path, "PNG")
    return out_path
