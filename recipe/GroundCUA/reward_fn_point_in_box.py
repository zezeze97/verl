import ast
import json
import math
import re
from typing import Any, Dict


BOX_TAG_PATTERN = re.compile(
    r"<\|box_start\|>\s*\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)\s*,\s*"
    r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)\s*<\|box_end\|>",
    re.DOTALL,
)


def _extract_answer_segment(solution_str: str) -> str:
    if not isinstance(solution_str, str):
        return ""

    match = re.search(r"</think>\s*(.*)", solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return solution_str.strip()


def _try_load_json_like(candidate: str) -> Any:
    try:
        return json.loads(candidate)
    except Exception:
        pass

    try:
        return ast.literal_eval(candidate)
    except Exception:
        return None


def _coerce_number(value: Any):
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        value = value.strip()
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value):
            return float(value)

    return None


def _normalize_box_coords(x1: Any, y1: Any, x2: Any, y2: Any):
    coords = [_coerce_number(v) for v in (x1, y1, x2, y2)]
    if any(v is None for v in coords):
        return None

    x1, y1, x2, y2 = coords
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _extract_point_from_obj(obj: Any):
    if isinstance(obj, dict):
        point = obj.get("point_2d")
        if isinstance(point, list) and len(point) == 2:
            x = _coerce_number(point[0])
            y = _coerce_number(point[1])
            if x is not None and y is not None:
                return x, y

        if all(k in obj for k in ("x", "y")):
            x = _coerce_number(obj["x"])
            y = _coerce_number(obj["y"])
            if x is not None and y is not None:
                return x, y

    if isinstance(obj, list):
        if len(obj) == 2:
            x = _coerce_number(obj[0])
            y = _coerce_number(obj[1])
            if x is not None and y is not None:
                return x, y

        for item in obj:
            point = _extract_point_from_obj(item)
            if point is not None:
                return point

    return None


def _extract_box_from_obj(obj: Any):
    if isinstance(obj, dict):
        if "bbox" in obj:
            box = _extract_box_from_obj(obj["bbox"])
            if box is not None:
                return box

        if "bbox_2d" in obj:
            box = _extract_box_from_obj(obj["bbox_2d"])
            if box is not None:
                return box

        if "box" in obj:
            box = _extract_box_from_obj(obj["box"])
            if box is not None:
                return box

        if all(k in obj for k in ("x1", "y1", "x2", "y2")):
            return _normalize_box_coords(obj["x1"], obj["y1"], obj["x2"], obj["y2"])

    if isinstance(obj, list):
        if len(obj) == 4:
            box = _normalize_box_coords(obj[0], obj[1], obj[2], obj[3])
            if box is not None:
                return box

        for item in obj:
            box = _extract_box_from_obj(item)
            if box is not None:
                return box

    return None


def _extract_box_from_text(text: str):
    if not isinstance(text, str):
        return None

    match = BOX_TAG_PATTERN.search(text)
    if match:
        return _normalize_box_coords(*match.groups())

    fallback = re.search(
        r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)\s*,\s*"
        r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)",
        text,
    )
    if fallback:
        return _normalize_box_coords(*fallback.groups())

    return None


def _extract_point(solution_str: str):
    answer = _extract_answer_segment(solution_str)

    direct = _extract_point_from_obj(_try_load_json_like(answer))
    if direct is not None:
        return direct

    code_block_match = re.search(r"```(?:json)?\s*(.*?)\s*```", answer, re.DOTALL)
    if code_block_match:
        direct = _extract_point_from_obj(_try_load_json_like(code_block_match.group(1)))
        if direct is not None:
            return direct

    for pattern in (r"\{[^{}]*\}", r"\[[^\[\]]*\]"):
        for candidate in re.findall(pattern, answer, re.DOTALL):
            point = _extract_point_from_obj(_try_load_json_like(candidate))
            if point is not None:
                return point

    return None


def _extract_box(solution_str: str):
    answer = _extract_answer_segment(solution_str)

    direct = _extract_box_from_obj(_try_load_json_like(answer))
    if direct is not None:
        return direct

    direct = _extract_box_from_text(answer)
    if direct is not None:
        return direct

    code_block_match = re.search(r"```(?:json)?\s*(.*?)\s*```", answer, re.DOTALL)
    if code_block_match:
        block = code_block_match.group(1)
        direct = _extract_box_from_obj(_try_load_json_like(block))
        if direct is not None:
            return direct
        direct = _extract_box_from_text(block)
        if direct is not None:
            return direct

    for pattern in (r"\{[^{}]*\}", r"\[[^\[\]]*\]"):
        for candidate in re.findall(pattern, answer, re.DOTALL):
            box = _extract_box_from_obj(_try_load_json_like(candidate))
            if box is not None:
                return box

    return None


def _normalize_ground_truth(ground_truth: Any, extra_info=None):
    candidates = [ground_truth]
    if isinstance(extra_info, dict):
        candidates.extend(
            [extra_info.get("answer"), extra_info.get("ground_truth"), extra_info.get("bbox"), extra_info.get("box")]
        )

    for candidate in candidates:
        if candidate is None:
            continue

        if isinstance(candidate, str):
            box = _extract_box_from_text(candidate)
            if box is not None:
                return box
            candidate = _try_load_json_like(candidate)

        box = _extract_box_from_obj(candidate)
        if box is not None:
            return box

    return None


def _point_from_box(box: Dict[str, float]):
    return ((box["x1"] + box["x2"]) / 2.0, (box["y1"] + box["y2"]) / 2.0)


def _format_coord(value: float) -> str:
    rounded = round(value)
    if math.isclose(value, rounded):
        return str(int(rounded))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _format_box(box: Dict[str, float]) -> str:
    return (
        "<|box_start|>"
        f"({_format_coord(box['x1'])},{_format_coord(box['y1'])}),"
        f"({_format_coord(box['x2'])},{_format_coord(box['y2'])})"
        "<|box_end|>"
    )


def compute_score(data_source, solution_str, ground_truth, extra_info=None, format_score: float = 0.1, **kwargs):
    del data_source, kwargs

    box = _normalize_ground_truth(ground_truth, extra_info=extra_info)
    point = _extract_point(solution_str)
    pred_box = _extract_box(solution_str)

    pred_type = ""
    if point is None and pred_box is not None:
        point = _point_from_box(pred_box)
        pred_type = "bbox"
    elif point is not None:
        pred_type = "point"

    format_reward = 1.0 if point is not None or pred_box is not None else 0.0
    accuracy_reward = 0.0
    pred = ""

    if pred_box is not None:
        pred = _format_box(pred_box)
    elif point is not None:
        pred = json.dumps({"point_2d": [point[0], point[1]]})

    if box is not None and point is not None:
        x, y = point
        accuracy_reward = float(box["x1"] <= x <= box["x2"] and box["y1"] <= y <= box["y2"])

    score = (1.0 - format_score) * accuracy_reward + format_score * format_reward

    return {
        "score": score,
        "format": format_reward,
        "accuracy": accuracy_reward,
        "pred": pred,
        "pred_type": pred_type,
        "in_box": int(accuracy_reward > 0),
    }
