from __future__ import annotations

import copy
import difflib
import re
from typing import Any

import cv2
import numpy as np

from .ocr import OcrItem, ocr
from .screen import ImageInput, ensure_portrait_image, load_image


DEFAULT_DIALOGUE_RULES: dict[str, Any] = {
    "min_confidence": 0.30,
    "min_box_height_ratio": 0.02,
    "line_gap": 24,
    "name_gap": 70,
    "sender_detection": {
        "method": "position",
        "sender2_min_x_ratio": 0.52,
        "enable_color_fallback": False,
        "color_saturation_threshold": 0.65,
    },
    "ocr_result_excluded_keywords": [],
    "line_excluded_keywords": [],
    "line_and_below_keywords": [],
    "basic_filter": {
        "enable_time_text_filter": True,
        "enable_debug_text_filter": True,
        "enable_reward_text_filter": True,
        "enable_center_y_region_filter": True,
        "center_y_min_ratio": 0.18,
        "center_y_max_ratio": 0.92,
        "enable_max_x_filter": True,
        "max_x_ratio": 0.98,
        "enable_max_width_filter": True,
        "max_width_ratio": 0.90,
        "enable_low_conf_single_char_filter": True,
        "single_char_confidence_threshold": 0.90,
        "enable_right_narrow_non_cjk_filter": True,
        "right_narrow_min_x_ratio": 0.88,
        "right_narrow_max_width_ratio": 0.12,
        "enable_non_cjk_non_emoji_filter": True,
    },
}


DEFAULT_LIST_RULES: dict[str, Any] = {
    "min_confidence": 0.30,
    "min_box_height_ratio": 0.02,
    "line_gap": 24,
    "main_top_ratio": 0.12,
    "main_bottom_ratio": 0.92,
    "max_unread_count": 99,
    "badge_position": "right",
    "system_keywords": [],
    "excluded_keywords": [],
    "content_excluded_keywords": [],
}


DEFAULT_OCR_MODES: dict[str, Any] = {
    "dialogue": {
        "preprocess": {
            "mode": "original",
            "scale": 2,
            "rotate_to_portrait": True,
        },
        "rules": DEFAULT_DIALOGUE_RULES,
    },
    "list": {
        "preprocess": {
            "mode": "original",
            "scale": 2,
            "rotate_to_portrait": True,
        },
        "rules": DEFAULT_LIST_RULES,
    },
}

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _keywords(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(line.strip() for line in value.splitlines() if line.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def normalize_ocr_text(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[\u4e00-\u9fff])", "", cleaned)
    return cleaned.strip(" _-|")


def contains_cjk(text: str) -> bool:
    return re.search(r"[\u4e00-\u9fff]", text) is not None


def contains_emoji(text: str) -> bool:
    return re.search(r"[\U0001F300-\U0001FAFF]", text) is not None


def text_similarity(left: str, right: str) -> float:
    left_normalized = re.sub(r"[\W_]+", "", normalize_ocr_text(left))
    right_normalized = re.sub(r"[\W_]+", "", normalize_ocr_text(right))
    if not left_normalized and not right_normalized:
        return 1.0
    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized == right_normalized:
        return 1.0
    return difflib.SequenceMatcher(None, left_normalized, right_normalized).ratio()


def looks_like_time_text(text: str) -> bool:
    compact = str(text or "").replace(" ", "")
    patterns = (
        r"\d{1,2}[:.]\d{2}",
        r"\d{1,2}[:.]\d{2}[:.]\d{2}",
        r"\d{1,2}/\d{1,2}",
        r"\d{1,2}-\d{1,2}",
        r"\d{1,3}分钟前",
        r"\d{1,3}小时前",
        r"\d{1,3}天前",
    )
    return any(re.fullmatch(pattern, compact) for pattern in patterns) or compact in {"刚刚", "昨天", "前天"}


def looks_like_debug_text(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(token in lowered for token in ("dx:", "dy:", "xv:", "newyv", "prs:", "size:", "kb/s"))


def looks_like_reward_text(text: str) -> bool:
    compact = str(text or "").replace(" ", "")
    return bool(re.search(r"[+￥¥]\s*0\.\d+", compact) or re.search(r"0\.\d+元?[句次]", compact))


def find_keyword(text: str, keywords: tuple[str, ...]) -> str | None:
    for keyword in keywords:
        if keyword and keyword in text:
            return keyword
    return None


def _sample_background_saturation(image_array: np.ndarray, min_x: int, max_x: int, min_y: int, max_y: int) -> float:
    pad_x = max(10, min(28, max(4, (max_x - min_x) // 3)))
    pad_y = max(8, min(20, max(4, (max_y - min_y) // 2)))
    crop = image_array[
        max(0, min_y - pad_y) : min(image_array.shape[0], max_y + pad_y),
        max(0, min_x - pad_x) : min(image_array.shape[1], max_x + pad_x),
    ].astype(np.float32)
    if crop.size == 0:
        return 0.0
    flat = crop.reshape(-1, 3)
    rgb_max = flat.max(axis=1)
    rgb_min = flat.min(axis=1)
    saturation = np.divide(rgb_max - rgb_min, rgb_max, out=np.zeros_like(rgb_max), where=rgb_max > 0)
    value = rgb_max / 255.0
    usable = saturation[value > 0.45]
    if usable.size == 0:
        usable = saturation
    return float(np.median(usable))


def _normalize_region_item(item: OcrItem, region: Any | None) -> OcrItem:
    if region is None:
        return dict(item)
    if not isinstance(region, (list, tuple)) or len(region) != 4:
        return dict(item)
    rx, ry, rw, rh = (float(region[0]), float(region[1]), float(region[2]), float(region[3]))
    normalized = dict(item)
    normalized["x"] = rx + float(item["x"]) * rw
    normalized["y"] = ry + float(item["y"]) * rh
    normalized["w"] = float(item["w"]) * rw
    normalized["h"] = float(item["h"]) * rh
    return normalized


def _to_rule_items(
    image: ImageInput,
    ocr_items: list[OcrItem],
    *,
    region: Any | None = None,
) -> tuple[list[dict[str, Any]], list[OcrItem], tuple[int, int]]:
    pil_image = load_image(image)
    image_width, image_height = pil_image.size
    image_array = np.array(pil_image.convert("RGB"))
    normalized_items = [_normalize_region_item(item, region) for item in ocr_items]
    rule_items: list[dict[str, Any]] = []
    for item in normalized_items:
        text = normalize_ocr_text(str(item.get("text", "")))
        if not text:
            continue
        min_x = int(round(max(0.0, min(1.0, float(item["x"]))) * image_width))
        min_y = int(round(max(0.0, min(1.0, float(item["y"]))) * image_height))
        max_x = int(round(max(0.0, min(1.0, float(item["x"]) + float(item["w"]))) * image_width))
        max_y = int(round(max(0.0, min(1.0, float(item["y"]) + float(item["h"]))) * image_height))
        if max_x <= min_x or max_y <= min_y:
            continue
        confidence = item.get("confidence")
        confidence_value = float(confidence) if confidence is not None else 1.0
        rule_items.append(
            {
                "raw_text": str(item.get("text", "")),
                "text": text,
                "confidence": round(confidence_value, 4),
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
                "center_x": round((min_x + max_x) / 2, 1),
                "center_y": round((min_y + max_y) / 2, 1),
                "width": max_x - min_x,
                "height": max_y - min_y,
                "background_saturation": round(
                    _sample_background_saturation(image_array, min_x, max_x, min_y, max_y), 4
                ),
                "debug_stage": "ocr_kept",
                "debug_reason": "",
            }
        )
    rule_items.sort(key=lambda part: (part["center_y"], part["min_x"]))
    normalized_items.sort(key=lambda part: (part["y"], part["x"]))
    return rule_items, normalized_items, (image_width, image_height)


def _original_point_to_portrait(point: tuple[float, float], original_size: tuple[int, int]) -> tuple[float, float]:
    x, y = point
    width, _height = original_size
    return y, width - x


def _rule_space_items(
    rule_items: list[dict[str, Any]],
    image_size: tuple[int, int],
    mode_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], tuple[int, int], str | None]:
    preprocess = mode_config.get("preprocess", {}) if isinstance(mode_config.get("preprocess", {}), dict) else {}
    image_width, image_height = image_size
    if not preprocess.get("rotate_to_portrait", True) or image_width <= image_height:
        return rule_items, image_size, None
    transformed: list[dict[str, Any]] = []
    for item in rule_items:
        corners = [
            (item["min_x"], item["min_y"]),
            (item["max_x"], item["min_y"]),
            (item["max_x"], item["max_y"]),
            (item["min_x"], item["max_y"]),
        ]
        points = [_original_point_to_portrait(point, image_size) for point in corners]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        copied = dict(item)
        copied["min_x"] = int(round(min(xs)))
        copied["max_x"] = int(round(max(xs)))
        copied["min_y"] = int(round(min(ys)))
        copied["max_y"] = int(round(max(ys)))
        copied["center_x"] = round((copied["min_x"] + copied["max_x"]) / 2, 1)
        copied["center_y"] = round((copied["min_y"] + copied["max_y"]) / 2, 1)
        copied["width"] = copied["max_x"] - copied["min_x"]
        copied["height"] = copied["max_y"] - copied["min_y"]
        transformed.append(copied)
    transformed.sort(key=lambda part: (part["center_y"], part["min_x"]))
    return transformed, (image_height, image_width), "ccw90"


def _round_bounds(bounds: list[float]) -> list[float]:
    return [round(float(value), 3) for value in bounds]


def _order_segments(segments: list[tuple[int, int, str]], line_gap: int) -> list[str]:
    ordered = sorted(segments, key=lambda item: (item[0], item[1]))
    if not ordered:
        return []
    lines: list[list[tuple[int, int, str]]] = [[ordered[0]]]
    current_y = ordered[0][0]
    for item in ordered[1:]:
        if abs(item[0] - current_y) <= line_gap:
            lines[-1].append(item)
            current_y = min(current_y, item[0])
            continue
        lines.append([item])
        current_y = item[0]
    return ["".join(text for _, _, text in sorted(line, key=lambda part: part[1])) for line in lines]


def _group_by_line(items: list[dict[str, Any]], line_gap: int) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for item in sorted(items, key=lambda part: (part["center_y"], part["min_x"])):
        if not groups:
            groups.append([item])
            continue
        group_y = sum(part["center_y"] for part in groups[-1]) / len(groups[-1])
        if abs(item["center_y"] - group_y) <= line_gap:
            groups[-1].append(item)
            continue
        groups.append([item])
    return groups


def _line_text(group: list[dict[str, Any]], line_gap: int) -> str:
    segments = [(item["min_y"], item["min_x"], item["text"]) for item in group]
    return "".join(_order_segments(segments, line_gap)).strip()


def build_dialogue_result(
    rule_items: list[dict[str, Any]],
    image_size: tuple[int, int],
    rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rules = _deep_merge(DEFAULT_DIALOGUE_RULES, rules or {})
    image_width, image_height = image_size
    line_gap = int(rules.get("line_gap", 24))
    name_gap = int(rules.get("name_gap", 110))
    min_confidence = float(rules.get("min_confidence", 0.30))
    min_height = image_height * float(rules.get("min_box_height_ratio", 0.02))
    sender_detection = (
        rules.get("sender_detection", {}) if isinstance(rules.get("sender_detection", {}), dict) else {}
    )
    basic = rules.get("basic_filter", {}) if isinstance(rules.get("basic_filter", {}), dict) else {}
    result_excluded = _keywords(rules.get("ocr_result_excluded_keywords"))
    line_excluded = _keywords(rules.get("line_excluded_keywords"))
    line_and_below = _keywords(rules.get("line_and_below_keywords"))

    def invalid_reason(item: dict[str, Any]) -> str | None:
        text = item["text"]
        if item["confidence"] < min_confidence:
            return "low_confidence"
        if item["height"] < min_height:
            return "box_too_short"
        if basic.get("enable_center_y_region_filter", True):
            if item["center_y"] < image_height * float(basic.get("center_y_min_ratio", 0.18)):
                return "above_dialogue_region"
            if item["center_y"] > image_height * float(basic.get("center_y_max_ratio", 0.92)):
                return "below_dialogue_region"
        if basic.get("enable_time_text_filter", True) and looks_like_time_text(text):
            return "time_text"
        if basic.get("enable_debug_text_filter", True) and looks_like_debug_text(text):
            return "debug_text"
        if basic.get("enable_reward_text_filter", True) and looks_like_reward_text(text):
            return "reward_text"
        if find_keyword(text, result_excluded):
            return "ocr_keyword"
        if basic.get("enable_max_x_filter", True) and item["max_x"] >= image_width * float(basic.get("max_x_ratio", 0.98)):
            return "too_far_right"
        if basic.get("enable_max_width_filter", True) and item["width"] > image_width * float(basic.get("max_width_ratio", 0.90)):
            return "too_wide"
        if basic.get("enable_low_conf_single_char_filter", True):
            threshold = float(basic.get("single_char_confidence_threshold", 0.90))
            if len(text) == 1 and item["confidence"] < threshold and not contains_emoji(text):
                return "low_confidence_single_char"
        if basic.get("enable_right_narrow_non_cjk_filter", True):
            min_x_ratio = float(basic.get("right_narrow_min_x_ratio", 0.88))
            max_width_ratio = float(basic.get("right_narrow_max_width_ratio", 0.12))
            if (
                item["min_x"] >= image_width * min_x_ratio
                and item["width"] <= image_width * max_width_ratio
                and not contains_cjk(text)
                and not contains_emoji(text)
            ):
                return "right_narrow_non_cjk"
        if basic.get("enable_non_cjk_non_emoji_filter", True) and not contains_cjk(text) and not contains_emoji(text):
            return "non_cjk"
        return None

    def sender_for(item: dict[str, Any]) -> str:
        method = str(sender_detection.get("method", "position") or "position")
        sender2_min_x_ratio = float(sender_detection.get("sender2_min_x_ratio", 0.52))
        if method == "position":
            return "sender2" if item["min_x"] > image_width * sender2_min_x_ratio else "sender1"
        saturation_threshold = float(sender_detection.get("color_saturation_threshold", 0.65))
        if method == "color" and item.get("background_saturation", 0.0) >= saturation_threshold:
            return "sender2"
        if bool(sender_detection.get("enable_color_fallback", False)) and item.get("background_saturation", 0.0) >= saturation_threshold:
            return "sender2"
        if item["min_x"] > image_width * sender2_min_x_ratio:
            return "sender2"
        return "sender1"

    candidates: list[dict[str, Any]] = []
    for item in rule_items:
        reason = invalid_reason(item)
        if reason:
            item["debug_stage"] = "filtered_before_merge"
            item["debug_reason"] = reason
            continue
        item["sender"] = sender_for(item)
        item["debug_stage"] = "candidate_kept"
        candidates.append(item)

    filtered_groups: list[list[dict[str, Any]]] = []
    cutoff = False
    for group in _group_by_line(candidates, line_gap):
        text = _line_text(group, line_gap)
        if cutoff:
            continue
        if find_keyword(text, line_and_below):
            cutoff = True
            continue
        filtered_groups.append(group)

    merged: list[dict[str, Any]] = []
    for item in [part for group in filtered_groups for part in group]:
        if merged:
            previous = merged[-1]
            vertical_gap = item["min_y"] - previous["max_y"]
            horizontal_overlap = min(previous["max_x"], item["max_x"]) - max(previous["min_x"], item["min_x"])
            previous_center_x = (previous["min_x"] + previous["max_x"]) / 2
            center_gap = abs(previous_center_x - item["center_x"])
            same_bubble_column = horizontal_overlap > -image_width * 0.08 or center_gap <= image_width * 0.24
            if previous["sender"] == item["sender"] and vertical_gap <= name_gap and same_bubble_column:
                previous["segments"].append((item["min_y"], item["min_x"], item["text"]))
                previous["max_y"] = max(previous["max_y"], item["max_y"])
                previous["max_x"] = max(previous["max_x"], item["max_x"])
                previous["min_x"] = min(previous["min_x"], item["min_x"])
                continue
        merged.append(
            {
                "sender": item["sender"],
                "segments": [(item["min_y"], item["min_x"], item["text"])],
                "min_x": item["min_x"],
                "max_x": item["max_x"],
                "min_y": item["min_y"],
                "max_y": item["max_y"],
            }
        )

    dialogues: list[dict[str, str]] = []
    for item in merged:
        content = "".join(_order_segments(item["segments"], line_gap)).strip()
        if content and not find_keyword(content, line_excluded):
            dialogues.append({"sender": item["sender"], "content": content})

    name_candidates = [
        item["text"]
        for item in rule_items
        if item["center_y"] <= image_height * 0.12
        and 2 <= len(item["text"]) <= 16
        and contains_cjk(item["text"])
        and not looks_like_time_text(item["text"])
    ]
    return {"user_name": name_candidates[0] if name_candidates else "", "dialogues": dialogues}


def _box_overlap_ratio(left: dict[str, Any], right: dict[str, Any]) -> float:
    overlap_width = min(left["max_x"], right["max_x"]) - max(left["min_x"], right["min_x"])
    overlap_height = min(left["max_y"], right["max_y"]) - max(left["min_y"], right["min_y"])
    if overlap_width <= 0 or overlap_height <= 0:
        return 0.0
    overlap_area = overlap_width * overlap_height
    left_area = max(1, left["width"] * left["height"])
    right_area = max(1, right["width"] * right["height"])
    return overlap_area / min(left_area, right_area)


def _prefer_rule_item(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_conf = float(left.get("confidence") or 0.0)
    right_conf = float(right.get("confidence") or 0.0)
    if abs(left_conf - right_conf) >= 0.02:
        return left if left_conf > right_conf else right
    return left if len(left["text"]) >= len(right["text"]) else right


def _dedupe_rule_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda part: (part["min_x"], part["min_y"], -part["width"])):
        matched_index: int | None = None
        for index, current in enumerate(deduped):
            if _box_overlap_ratio(item, current) >= 0.65 and text_similarity(item["text"], current["text"]) >= 0.82:
                matched_index = index
                break
        if matched_index is None:
            deduped.append(item)
        else:
            deduped[matched_index] = _prefer_rule_item(deduped[matched_index], item)
    return sorted(deduped, key=lambda part: (part["min_x"], part["min_y"]))


def _list_line_text(items: list[dict[str, Any]], line_gap: int) -> str:
    segments = [(item["min_y"], item["min_x"], item["text"]) for item in _dedupe_rule_items(items)]
    return "".join(_order_segments(segments, line_gap)).strip()


def _looks_like_avatar_item(item: dict[str, Any], image_width: int, avatar_max_x_ratio: float) -> bool:
    return item["center_x"] <= image_width * avatar_max_x_ratio and len(item["text"].strip()) <= 2


def _looks_like_bottom_nav(text: str, keywords: tuple[str, ...]) -> bool:
    compact = normalize_ocr_text(text).replace(" ", "")
    if not compact or len(keywords) < 2:
        return False
    return sum(1 for keyword in keywords if keyword and keyword in compact) >= 2


def _build_bottom_nav_result(
    rule_items: list[dict[str, Any]],
    image_size: tuple[int, int],
    rules: dict[str, Any],
) -> list[dict[str, Any]]:
    image_width, image_height = image_size
    bottom_keywords = _keywords(rules.get("bottom_nav_keywords"))
    bottom_top_ratio = float(rules.get("bottom_nav_top_ratio", 0.88))
    bottom_accept_any_text = bool(rules.get("bottom_nav_accept_any_text", False))
    top_nav_keywords = _keywords(rules.get("top_nav_keywords"))
    top_nav_top_ratio = float(rules.get("top_nav_top_ratio", 0.0))
    top_nav_bottom_ratio = float(rules.get("top_nav_bottom_ratio", 0.0))

    def _nav_candidates(
        top_ratio: float,
        bottom_ratio: float,
        keywords: tuple[str, ...],
        prefer_bottom: bool,
    ) -> list[dict[str, Any]]:
        if not keywords or bottom_ratio <= top_ratio:
            return []
        candidates = [
            item
            for item in rule_items
            if image_height * top_ratio <= item["center_y"] <= image_height * bottom_ratio
            and 1 <= len(item["text"]) <= 4
            and contains_cjk(item["text"])
            and (item["text"] in keywords or (prefer_bottom and bottom_accept_any_text))
        ]
        if candidates:
            anchor = (
                max(item["center_y"] for item in candidates)
                if prefer_bottom
                else min(item["center_y"] for item in candidates)
            )
            candidates = [
                item
                for item in candidates
                if abs(item["center_y"] - anchor) <= max(72, image_height * 0.035)
            ]
        return _dedupe_rule_items(candidates)

    candidates = _nav_candidates(bottom_top_ratio, 1.0, bottom_keywords, prefer_bottom=True)
    candidates.extend(
        _nav_candidates(top_nav_top_ratio, top_nav_bottom_ratio, top_nav_keywords, prefer_bottom=False)
    )
    candidates = _dedupe_rule_items(candidates)
    candidates.sort(
        key=lambda item: (
            0
            if top_nav_top_ratio <= item["center_y"] / image_height <= top_nav_bottom_ratio
            and item["text"] in top_nav_keywords
            else 1,
            item["center_x"],
        )
    )
    return [
        {
            "name": item["text"],
            "bounds": _round_bounds([
                item["min_x"] / image_width,
                item["min_y"] / image_height,
                item["max_x"] / image_width,
                item["max_y"] / image_height,
            ]),
        }
        for item in candidates[:9]
    ]


def _strip_unread_prefix(text: str) -> tuple[int | None, str]:
    cleaned = normalize_ocr_text(text)
    if cleaned.startswith("["):
        close_index = cleaned.find("]")
        if close_index != -1:
            return 1, cleaned[close_index + 1 :].strip()
    compact = cleaned.replace(" ", "")
    compact = compact.translate(str.maketrans({"?": "+", "?": "+", "?": ".", "?": ".", "?": "F", "?": "f"}))
    if re.fullmatch(r"99(?:[+Ff.])?", compact):
        return 99, ""
    match = re.match(r"^([1-9]\d?)(?=[\u4e00-\u9fffA-Za-z])", compact)
    if not match:
        return None, cleaned
    unread_count = int(match.group(1))
    if unread_count > 99:
        return None, cleaned
    return unread_count, compact[len(match.group(1)) :]

def _extract_list_name(text: str) -> str:
    _unread, cleaned = _strip_unread_prefix(text)
    cleaned = cleaned.strip("[](){} ")
    cleaned = re.sub(r"\d{1,2}[:.]\d{2}$", "", cleaned).strip(" :|-_")
    leading_cjk = re.match(r"^[\u4e00-\u9fff]{1,12}", cleaned)
    if leading_cjk:
        return leading_cjk.group(0).strip(" :|-_")
    leading_ascii = re.match(r"^[A-Za-z0-9_.]{1,12}", cleaned)
    if leading_ascii:
        return leading_ascii.group(0).strip(" :|-_")
    return cleaned[:24]


def _clean_list_content(text: str, name: str) -> str:
    _unread, cleaned = _strip_unread_prefix(text)
    cleaned = normalize_ocr_text(cleaned).strip("[](){} ")
    if name and cleaned.startswith(name):
        cleaned = cleaned[len(name) :].lstrip(" :|-_")
    cleaned = re.sub(r"^\d{1,2}[:.]\d{2}", "", cleaned).lstrip(" :|-_")
    return cleaned


def _unread_count_from_text(text: str, max_unread_count: int = 99) -> int | None:
    compact = normalize_ocr_text(text).replace(" ", "")
    compact = compact.translate(str.maketrans({"?": "+", "?": "+", "?": ".", "?": ".", "?": "F", "?": "f"}))
    compact = compact.translate(str.maketrans({"①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5"}))
    if re.fullmatch(r"99(?:[+Ff.])?", compact):
        return min(99, max_unread_count)
    match = re.fullmatch(r"([1-9]\d?)(?:[+Ff.])?", compact)
    if not match:
        return None
    value = int(match.group(1))
    return value if 0 <= value <= max_unread_count else None


def _looks_like_unread_badge(item: dict[str, Any], image_width: int, image_height: int, right_meta_min: float) -> bool:
    if item["min_x"] < right_meta_min:
        return False
    if item["width"] > image_width * 0.22 or item["height"] > image_height * 0.05:
        return False
    text = item["text"]
    return bool(text) and not looks_like_time_text(text) and _unread_count_from_text(text) is not None


def _detect_red_badge_bounds(
    image_array: np.ndarray,
    row_bounds: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    rules: dict[str, Any],
) -> tuple[int, int, int, int] | None:
    row_left, row_top, row_right, row_bottom = row_bounds
    badge_position = str(rules.get("badge_position", "right")).strip().lower()
    if badge_position == "left":
        search_left = max(0, int(image_width * float(rules.get("left_badge_min_ratio", 0.02))))
        search_right = min(image_width, int(image_width * float(rules.get("left_badge_max_ratio", 0.18))))
    else:
        search_left = max(row_left, int(image_width * float(rules.get("right_badge_min_ratio", 0.82))))
        search_right = min(image_width, row_right)
    search_box = (
        search_left,
        max(0, row_top),
        max(search_left + 1, search_right),
        min(image_height, row_bottom),
    )
    crop = image_array[search_box[1] : search_box[3], search_box[0] : search_box[2]]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([14, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([165, 80, 80]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_box: tuple[int, int, int, int] | None = None
    best_score = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        if area < max(30.0, image_width * 0.04):
            continue
        if width < 6 or height < 6 or width > image_width * 0.09 or height > image_width * 0.09:
            continue
        ratio = width / max(1, height)
        if ratio > 1.8 or ratio < 0.55:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.30:
            continue
        score = area * circularity
        if score > best_score:
            best_score = score
            best_box = (
                search_box[0] + x,
                search_box[1] + y,
                search_box[0] + x + width,
                search_box[1] + y + height,
            )
    return best_box


def _read_badge_number_from_crop(
    source_image: ImageInput,
    badge_bounds: tuple[int, int, int, int],
    image_size: tuple[int, int],
    base_ocr_config: dict[str, Any],
    max_unread_count: int,
    rules: dict[str, Any],
) -> int | None:
    image_width, image_height = image_size
    left, top, right, bottom = badge_bounds
    pad = int(rules.get("badge_crop_padding", 6))
    region = [
        max(0.0, (left - pad) / image_width),
        max(0.0, (top - pad) / image_height),
        min(1.0, (right - left + pad * 2) / image_width),
        min(1.0, (bottom - top + pad * 2) / image_height),
    ]
    config = dict(base_ocr_config)
    config["region"] = region
    config["preprocess"] = "original"
    config["scale"] = max(1, int(rules.get("badge_ocr_scale", 6) or 6))
    config["rotate_to_portrait"] = False
    best_value: int | None = None
    best_confidence = -1.0
    for item in ocr(source_image, config):
        value = _unread_count_from_text(str(item.get("text", "")), max_unread_count)
        if value is not None:
            confidence = float(item.get("confidence") or 0.0)
            if confidence > best_confidence:
                best_value = value
                best_confidence = confidence
    return best_value


def _read_unread_number_from_row_strip(
    source_image: ImageInput,
    row_bounds: tuple[int, int, int, int],
    image_size: tuple[int, int],
    base_ocr_config: dict[str, Any],
    max_unread_count: int,
    bottom_nav_top: int,
    rules: dict[str, Any],
) -> int | None:
    _image_width, image_height = image_size
    _row_left, row_top, _row_right, row_bottom = row_bounds
    right_meta_min_ratio = float(rules.get("right_meta_min_ratio", 0.78))
    strip_left = max(float(rules.get("row_strip_left_min_ratio", 0.84)), right_meta_min_ratio + float(rules.get("row_strip_extra_left_ratio", 0.06)))
    strip_width = min(float(rules.get("row_strip_width_ratio", 0.12)), max(0.0, 1.0 - strip_left))
    strip_top = max(0.0, row_top / image_height)
    row_height = row_bottom - row_top
    strip_bottom_px = min(
        row_bottom,
        bottom_nav_top - int(rules.get("row_strip_bottom_nav_margin", 4)),
        row_top
        + max(
            int(rules.get("row_strip_min_height_px", 24)),
            min(
                int(row_height * float(rules.get("row_strip_height_ratio", 0.62))),
                int(image_height * float(rules.get("row_strip_max_height_ratio", 0.12))),
            ),
        ),
    )
    strip_bottom = strip_bottom_px / image_height
    if strip_bottom <= strip_top or strip_width <= 0.0:
        return None
    config = dict(base_ocr_config)
    config["region"] = [strip_left, strip_top, strip_width, strip_bottom - strip_top]
    config["preprocess"] = "original"
    config["scale"] = max(1, int(rules.get("row_badge_ocr_scale", 4) or 4))
    config["rotate_to_portrait"] = False
    best_value: int | None = None
    best_confidence = -1.0
    for item in ocr(source_image, config):
        value = _unread_count_from_text(str(item.get("text", "")), max_unread_count)
        if value is None:
            continue
        confidence = float(item.get("confidence") or 0.0)
        if confidence < float(rules.get("row_badge_min_confidence", 0.80)):
            continue
        if confidence > best_confidence:
            best_value = value
            best_confidence = confidence
    if best_value is not None:
        return best_value
    return None


def _looks_like_list_content(text: str, image_width: int, width: int) -> bool:
    compact = normalize_ocr_text(text).replace(" ", "")
    if not compact or looks_like_time_text(compact):
        return False
    if re.search(r"[，。！？!?,:：.]", compact):
        return True
    return len(compact) >= 7 and width >= image_width * 0.24


def _looks_like_list_name(text: str) -> bool:
    compact = normalize_ocr_text(text).replace(" ", "")
    if not compact or looks_like_time_text(compact):
        return False
    if contains_cjk(compact):
        return True
    return re.fullmatch(r"[A-Za-z0-9_.-]{1,16}", compact) is not None


def _build_list_line(group: list[dict[str, Any]], image_width: int, line_gap: int, rules: dict[str, Any]) -> dict[str, Any]:
    avatar_max_x_ratio = float(rules.get("avatar_max_x_ratio", 0.18))
    text_right_max = image_width * float(rules.get("text_right_max_ratio", 0.78))
    right_meta_min = image_width * float(rules.get("right_meta_min_ratio", 0.78))
    main_items = [
        item
        for item in group
        if item["min_x"] < text_right_max and not _looks_like_avatar_item(item, image_width, avatar_max_x_ratio)
    ]
    right_items = [item for item in group if item["min_x"] >= right_meta_min]
    main_items = _dedupe_rule_items(main_items)
    right_items = _dedupe_rule_items(right_items)
    return {
        "group": group,
        "main_items": main_items,
        "right_items": right_items,
        "left_text": _list_line_text(main_items, line_gap),
        "right_text": _list_line_text(right_items, line_gap),
        "min_x": min(item["min_x"] for item in group),
        "max_x": max(item["max_x"] for item in group),
        "min_y": min(item["min_y"] for item in group),
        "max_y": max(item["max_y"] for item in group),
        "center_y": sum(item["center_y"] for item in group) / len(group),
        "left_width": (
            max((item["max_x"] for item in main_items), default=0)
            - min((item["min_x"] for item in main_items), default=0)
        ),
    }


def _is_list_header_line(line: dict[str, Any], image_width: int, rules: dict[str, Any]) -> bool:
    text = line["left_text"]
    if not text or looks_like_time_text(text) or not line["main_items"]:
        return False
    first_x = min(item["min_x"] for item in line["main_items"])
    if first_x < image_width * float(rules.get("text_left_min_ratio", 0.16)):
        return False
    name = _extract_list_name(text)
    if not name or not _looks_like_list_name(name):
        return False
    if _looks_like_list_content(name, image_width, int(line["left_width"])):
        return False
    return 1 <= len(name.replace(" ", "")) <= 16


def build_list_result(
    rule_items: list[dict[str, Any]],
    image_size: tuple[int, int],
    rules: dict[str, Any] | None = None,
    source_image: ImageInput | None = None,
    ocr_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rules = _deep_merge(DEFAULT_LIST_RULES, rules or {})
    image_width, image_height = image_size
    line_gap = int(rules.get("line_gap", 24))
    min_confidence = float(rules.get("min_confidence", 0.30))
    min_height = image_height * float(rules.get("min_box_height_ratio", 0.02))
    top = image_height * float(rules.get("main_top_ratio", 0.12))
    bottom = image_height * float(rules.get("main_bottom_ratio", 0.92))
    content_gap_threshold = int(
        max(
            float(rules.get("content_gap_min", 38)),
            min(
                float(rules.get("content_gap_max", 62)),
                image_height * float(rules.get("content_gap_ratio", 0.022)),
            ),
        )
    )
    bottom_nav_top = image_height * float(rules.get("bottom_nav_top_ratio", 0.88))
    system_keywords = _keywords(rules.get("system_keywords"))
    excluded_keywords = _keywords(rules.get("excluded_keywords"))
    content_excluded_keywords = _keywords(rules.get("content_excluded_keywords"))
    bottom_nav_keywords = _keywords(rules.get("bottom_nav_keywords"))
    max_unread_count = int(rules.get("max_unread_count", 99))
    right_meta_min = image_width * float(rules.get("right_meta_min_ratio", 0.78))
    top_nav_top = image_height * float(rules.get("top_nav_top_ratio", 0.0))
    top_nav_bottom = image_height * float(rules.get("top_nav_bottom_ratio", 0.0))
    bottom_nav = _build_bottom_nav_result(rule_items, image_size, rules)
    bottom_nav_names = [item["name"] for item in bottom_nav]
    image_array = np.array(load_image(source_image).convert("RGB")) if source_image is not None else None

    candidates = _dedupe_rule_items(
        [
            item
            for item in rule_items
            if item["confidence"] >= min_confidence
            and (
                item["height"] >= min_height
                or _looks_like_unread_badge(item, image_width, image_height, right_meta_min)
            )
            and top <= item["center_y"] <= bottom
            and item["min_x"] < image_width * 0.96
        ]
    )

    lines: list[dict[str, Any]] = []
    for group in _group_by_line(candidates, line_gap):
        line = _build_list_line(group, image_width, line_gap, rules)
        text = line["left_text"]
        if not text:
            continue
        is_top_nav_line = top_nav_bottom > top_nav_top and top_nav_top <= line["center_y"] <= top_nav_bottom
        if _looks_like_bottom_nav(text, tuple(bottom_nav_names)) and (
            line["center_y"] >= bottom_nav_top or is_top_nav_line
        ):
            continue
        if line["center_y"] < image_height * 0.14 and (
            find_keyword(text, system_keywords) or find_keyword(text, excluded_keywords)
        ):
            continue
        lines.append(line)

    raw_rows: list[dict[str, Any]] = []
    current_row: dict[str, Any] | None = None
    for line in lines:
        is_header = _is_list_header_line(line, image_width, rules)
        if current_row is not None:
            vertical_gap = line["min_y"] - current_row["last_max_y"]
            if vertical_gap <= content_gap_threshold:
                if is_header and current_row["content_lines"]:
                    raw_rows.append(current_row)
                    current_row = None
                else:
                    if line["left_text"] and not looks_like_time_text(line["left_text"]):
                        current_row["content_lines"].append(line)
                        current_row["last_max_y"] = max(current_row["last_max_y"], line["max_y"])
                        current_row["max_x"] = max(current_row["max_x"], line["max_x"])
                    continue
            if current_row is None:
                pass
            else:
                raw_rows.append(current_row)
                current_row = None
        if is_header:
            current_row = {
                "header_line": line,
                "content_lines": [],
                "last_max_y": line["max_y"],
                "max_x": line["max_x"],
            }
            continue
        if raw_rows and line["min_y"] - raw_rows[-1]["last_max_y"] <= content_gap_threshold:
            raw_rows[-1]["content_lines"].append(line)
            raw_rows[-1]["last_max_y"] = max(raw_rows[-1]["last_max_y"], line["max_y"])
            raw_rows[-1]["max_x"] = max(raw_rows[-1]["max_x"], line["max_x"])

    if current_row is not None:
        raw_rows.append(current_row)

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows):
        header_line = row["header_line"]
        if header_line["center_y"] >= bottom_nav_top:
            continue
        unread_count, header_text = _strip_unread_prefix(header_line["left_text"])
        name = _extract_list_name(header_text)
        if not name or find_keyword(name, excluded_keywords):
            continue
        content_parts = [
            _clean_list_content(line["left_text"], name)
            for line in row["content_lines"]
            if line["left_text"] and not looks_like_time_text(line["left_text"])
        ]
        content = "".join(part for part in content_parts if part).strip()
        if name and content.startswith(name):
            content = content[len(name) :].lstrip(" :|-_")
        if find_keyword(content, content_excluded_keywords):
            continue
        row_lines = [header_line, *row["content_lines"]]
        next_top = raw_rows[index + 1]["header_line"]["min_y"] if index + 1 < len(raw_rows) else int(image_height * 0.93)
        row_left = max(0, min(line["min_x"] for line in row_lines) - 16)
        row_top = max(0, header_line["min_y"] - 12)
        row_right = min(image_width, max(max(line["max_x"] for line in row_lines) + 16, image_width - 32))
        detection_row_bottom = min(image_height, max(row["last_max_y"] + 16, next_top - 12))
        content_bottom = row["last_max_y"] + 12
        minimum_bottom = header_line["max_y"] + 20
        next_limit = next_top - 8
        row_bottom = min(image_height, max(minimum_bottom, min(content_bottom, next_limit)))
        favorability = None
        favorability_candidates: list[int] = []
        unread_candidates: list[int] = []
        badge_position = str(rules.get("badge_position", "right")).strip().lower()
        for line in row_lines:
            for right_item in line["right_items"]:
                if looks_like_time_text(right_item["text"]):
                    continue
                value = _unread_count_from_text(right_item["text"], max_unread_count)
                if value is not None:
                    if badge_position == "left":
                        favorability_candidates.append(value)
                    else:
                        unread_candidates.append(value)
        if badge_position == "left" and favorability_candidates:
            favorability = favorability_candidates[0]
        if badge_position == "left" and favorability is None:
            search_padding = max(8, int(image_height * 0.008))
            search_top = max(0, row_top - search_padding)
            search_bottom = min(image_height, detection_row_bottom + search_padding)
            for item in candidates:
                if item["min_x"] < right_meta_min:
                    continue
                if not (search_top <= item["center_y"] <= search_bottom):
                    continue
                if looks_like_time_text(item["text"]):
                    continue
                value = _unread_count_from_text(item["text"], max_unread_count)
                if value is not None:
                    favorability_candidates.append(value)
            if favorability_candidates:
                favorability = favorability_candidates[0]
        if badge_position != "left" and not unread_candidates:
            search_padding = max(8, int(image_height * 0.008))
            search_top = max(0, row_top - search_padding)
            search_bottom = min(image_height, detection_row_bottom + search_padding)
            for item in candidates:
                if not _looks_like_unread_badge(item, image_width, image_height, right_meta_min):
                    continue
                if not (search_top <= item["center_y"] <= search_bottom):
                    continue
                value = _unread_count_from_text(item["text"], max_unread_count)
                if value is not None:
                    unread_candidates.append(value)
        if badge_position != "left" and not unread_candidates and image_array is not None and ocr_config is not None:
            badge_bounds = _detect_red_badge_bounds(
                image_array,
                (row_left, row_top, row_right, detection_row_bottom),
                image_width,
                image_height,
                rules,
            )
            if badge_bounds is not None:
                value = _read_badge_number_from_crop(
                    source_image,
                    badge_bounds,
                    image_size,
                    ocr_config,
                    max_unread_count,
                    rules,
                )
                if value is not None:
                    unread_candidates.append(value)
        if badge_position == "left" and image_array is not None and ocr_config is not None:
            badge_bounds = _detect_red_badge_bounds(
                image_array,
                (row_left, row_top, row_right, detection_row_bottom),
                image_width,
                image_height,
                rules,
            )
            if badge_bounds is not None:
                value = _read_badge_number_from_crop(
                    source_image,
                    badge_bounds,
                    image_size,
                    ocr_config,
                    max_unread_count,
                    rules,
                )
                if value is not None:
                    unread_candidates.append(value)
            if not unread_candidates:
                unread_candidates.append(int(rules.get("badge_default_unread_count", 1)))
        if badge_position != "left" and not unread_candidates and ocr_config is not None and source_image is not None:
            value = _read_unread_number_from_row_strip(
                source_image,
                (row_left, row_top, row_right, detection_row_bottom),
                image_size,
                ocr_config,
                max_unread_count,
                int(bottom_nav_top),
                rules,
            )
            if value is not None:
                unread_candidates.append(value)
        if unread_count is None and unread_candidates:
            unread_count = unread_candidates[0]
        if unread_count is not None and not (0 <= unread_count <= max_unread_count):
            unread_count = None
        rows.append(
            {
                "name": name,
                "favorability": favorability,
                "unread_count": unread_count,
                "content": content,
                "bounds": _round_bounds([
                    row_left / image_width,
                    row_top / image_height,
                    row_right / image_width,
                    row_bottom / image_height,
                ]),
            }
        )
    return {"message_list": rows, "bottom_nav": bottom_nav}


def _merge_ocr_items(primary: list[OcrItem], secondary: list[OcrItem]) -> list[OcrItem]:
    merged = [dict(item) for item in primary]
    for candidate in secondary:
        matched_index: int | None = None
        for index, current in enumerate(merged):
            same_text = str(candidate["text"]) == str(current["text"])
            close_x = abs(float(candidate["x"]) - float(current["x"])) <= 0.03
            close_y = abs(float(candidate["y"]) - float(current["y"])) <= 0.03
            if same_text and close_x and close_y:
                matched_index = index
                break
        if matched_index is None:
            merged.append(dict(candidate))
            continue
        current_conf = merged[matched_index].get("confidence") or 0.0
        candidate_conf = candidate.get("confidence") or 0.0
        if candidate_conf > current_conf:
            merged[matched_index] = dict(candidate)
    merged.sort(key=lambda item: (item["y"], item["x"]))
    return merged


def _mode_config(config: dict[str, Any], mode: str) -> dict[str, Any]:
    configured_modes = config.get("ocr_modes", {}) if isinstance(config.get("ocr_modes", {}), dict) else {}
    configured = configured_modes.get(mode, {}) if isinstance(configured_modes.get(mode, {}), dict) else {}
    return _deep_merge(DEFAULT_OCR_MODES.get(mode, {}), configured)


def run_ocr_mode(
    image: ImageInput,
    config: dict[str, Any],
    mode: str,
    *,
    region: Any | None = None,
) -> dict[str, Any]:
    if mode not in {"dialogue", "list"}:
        raise ValueError("mode must be dialogue or list")
    base_ocr = config.get("ocr", {}) if isinstance(config.get("ocr", {}), dict) else {}
    mode_config = _mode_config(config, mode)
    preprocess = mode_config.get("preprocess", {}) if isinstance(mode_config.get("preprocess", {}), dict) else {}
    portrait_image = ensure_portrait_image(image)
    primary_ocr_config = dict(base_ocr)
    primary_ocr_config["region"] = region
    primary_ocr_config["preprocess"] = "original"
    primary_ocr_config["scale"] = preprocess.get("scale", 2)
    primary_ocr_config["rotate_to_portrait"] = False
    primary_items = ocr(portrait_image, primary_ocr_config)

    raw_items = primary_items
    rule_items, normalized_items, image_size = _to_rule_items(portrait_image, raw_items, region=region)
    rules = mode_config.get("rules", {}) if isinstance(mode_config.get("rules", {}), dict) else {}
    if mode == "dialogue":
        result = build_dialogue_result(rule_items, image_size, rules)
    else:
        result = build_list_result(
            rule_items,
            image_size,
            rules,
            source_image=portrait_image,
            ocr_config=primary_ocr_config,
        )
    return {
        "mode": mode,
        "items": rule_items,
        "ocr_items": normalized_items,
        "result": result,
    }
