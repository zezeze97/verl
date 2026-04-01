import ast
import json
import re
from typing import Any


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


def _extract_point_from_obj(obj: Any):
    if isinstance(obj, dict):
        point = obj.get("point_2d")
        if isinstance(point, list) and len(point) == 2 and all(isinstance(v, (int, float)) for v in point):
            return float(point[0]), float(point[1])

        if all(k in obj for k in ("x", "y")) and all(isinstance(obj[k], (int, float)) for k in ("x", "y")):
            return float(obj["x"]), float(obj["y"])

    if isinstance(obj, list):
        if len(obj) == 2 and all(isinstance(v, (int, float)) for v in obj):
            return float(obj[0]), float(obj[1])

        for item in obj:
            point = _extract_point_from_obj(item)
            if point is not None:
                return point

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


def _normalize_ground_truth(ground_truth: Any):
    if isinstance(ground_truth, str):
        ground_truth = _try_load_json_like(ground_truth)

    if isinstance(ground_truth, dict):
        if "bbox" in ground_truth and isinstance(ground_truth["bbox"], dict):
            ground_truth = ground_truth["bbox"]

        if all(k in ground_truth for k in ("x1", "y1", "x2", "y2")):
            x1, x2 = sorted((float(ground_truth["x1"]), float(ground_truth["x2"])))
            y1, y2 = sorted((float(ground_truth["y1"]), float(ground_truth["y2"])))
            return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    if isinstance(ground_truth, list) and len(ground_truth) == 4 and all(isinstance(v, (int, float)) for v in ground_truth):
        x1, y1, x2, y2 = ground_truth
        x1, x2 = sorted((float(x1), float(x2)))
        y1, y2 = sorted((float(y1), float(y2)))
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    return None


def compute_score(data_source, solution_str, ground_truth, extra_info=None, format_score: float = 0.1, **kwargs):
    del data_source, extra_info, kwargs

    box = _normalize_ground_truth(ground_truth)
    point = _extract_point(solution_str)

    format_reward = 1.0 if point is not None else 0.0
    accuracy_reward = 0.0
    pred = ""

    if point is not None:
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
        "in_box": int(accuracy_reward > 0),
    }
