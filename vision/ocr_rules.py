from __future__ import annotations

import copy
import difflib
import re
from typing import Any

import cv2
import numpy as np

from .ocr import OcrItem, ocr
from .screen import ImageInput, ensure_portrait_image, load_image


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
        r"\d{1,3}澶╁墠",
    )
    return any(re.fullmatch(pattern, compact) for pattern in patterns) or compact in {"鍒氬垰", "鏄ㄥぉ", "鍓嶅ぉ"}


def looks_like_debug_text(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(token in lowered for token in ("dx:", "dy:", "xv:", "newyv", "prs:", "size:", "kb/s"))


def looks_like_reward_text(text: str) -> bool:
    compact = str(text or "").replace(" ", "")
    return bool(re.search(r"\+?\s*0\.\d+", compact))


def find_keyword(text: str, keywords: tuple[str, ...]) -> str | None:
    for keyword in keywords:
        if keyword and keyword in text:
            return keyword
    return None


def _sample_background_color_metrics(
    image_array: np.ndarray, min_x: int, max_x: int, min_y: int, max_y: int
) -> tuple[float, float | None, float, float, float]:
    pad_x = max(10, min(28, max(4, (max_x - min_x) // 3)))
    pad_y = max(8, min(20, max(4, (max_y - min_y) // 2)))
    crop = image_array[
        max(0, min_y - pad_y) : min(image_array.shape[0], max_y + pad_y),
        max(0, min_x - pad_x) : min(image_array.shape[1], max_x + pad_x),
    ].astype(np.float32)
    if crop.size == 0:
        return 0.0, None, 0.0, 0.0, 0.0
    flat = crop.reshape(-1, 3)
    rgb_max = flat.max(axis=1)
    rgb_min = flat.min(axis=1)
    saturation = np.divide(rgb_max - rgb_min, rgb_max, out=np.zeros_like(rgb_max), where=rgb_max > 0)
    value = rgb_max / 255.0
    usable = saturation[value > 0.45]
    if usable.size == 0:
        usable = saturation
    background_saturation = float(np.median(usable))
    rgb_usable = flat[value > 0.20]
    if rgb_usable.size == 0:
        rgb_usable = flat
    median_r, median_g, median_b = np.median(rgb_usable, axis=0)
    blue_green_delta = float(median_b - median_g)
    blue_red_ratio = float(median_b / max(1.0, median_r))
    hsv = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    hsv_saturation = hsv[:, 1] / 255.0
    hsv_value = hsv[:, 2] / 255.0
    color_mask = (hsv_value > 0.20) & (hsv_saturation > 0.10)
    if np.count_nonzero(color_mask) >= 16:
        background_hue = float(np.median(hsv[color_mask, 0]))
        background_value = float(np.median(hsv_value[color_mask]))
        return background_saturation, background_hue, background_value, blue_green_delta, blue_red_ratio
    return background_saturation, None, float(np.median(hsv_value)), blue_green_delta, blue_red_ratio


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
        (
            background_saturation,
            background_hue,
            background_value,
            background_blue_green_delta,
            background_blue_red_ratio,
        ) = _sample_background_color_metrics(
            image_array, min_x, max_x, min_y, max_y
        )
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
                "background_saturation": round(background_saturation, 4),
                "background_hue": round(background_hue, 2) if background_hue is not None else None,
                "background_value": round(background_value, 4),
                "background_blue_green_delta": round(background_blue_green_delta, 2),
                "background_blue_red_ratio": round(background_blue_red_ratio, 4),
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
    preprocess = mode_config["preprocess"]
    image_width, image_height = image_size
    if not bool(preprocess["rotate_to_portrait"]) or image_width <= image_height:
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


def _item_bounds(item: dict[str, Any], image_width: int, image_height: int) -> list[float]:
    return _round_bounds(
        [
            item["min_x"] / image_width,
            item["min_y"] / image_height,
            item["max_x"] / image_width,
            item["max_y"] / image_height,
        ]
    )


def _debug_excluded_entry(
    *,
    entry_type: str,
    reason: str,
    text: str,
    image_width: int,
    image_height: int,
    raw_text: str | None = None,
    bounds: list[float] | None = None,
    matched_keyword: str | None = None,
    confidence: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "type": entry_type,
        "reason": reason,
        "text": text,
    }
    if raw_text is not None:
        entry["raw_text"] = raw_text
    if bounds is not None:
        entry["bounds"] = bounds
    if matched_keyword:
        entry["matched_keyword"] = matched_keyword
    if confidence is not None:
        entry["confidence"] = round(float(confidence), 4)
    if extra:
        entry.update(extra)
    return entry


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
    rules = dict(rules or {})
    image_width, image_height = image_size
    line_gap = int(rules["line_gap"])
    if "name_gap_ratio" in rules:
        name_gap = int(round(image_height * float(rules["name_gap_ratio"])))
    else:
        name_gap_value = float(rules["name_gap"])
        name_gap = int(round(image_height * name_gap_value)) if name_gap_value < 1 else int(round(name_gap_value))
    min_confidence = float(rules["min_confidence"])
    min_height = image_height * float(rules["min_box_height_ratio"])
    sender_detection = (
        rules["sender_detection"] if isinstance(rules["sender_detection"], dict) else {}
    )
    basic = rules["basic_filter"] if isinstance(rules["basic_filter"], dict) else {}
    result_excluded = _keywords(rules["ocr_result_excluded_keywords"])
    line_excluded = _keywords(rules["line_excluded_keywords"])
    line_and_below = _keywords(rules["line_and_below_keywords"])
    excluded_entries: list[dict[str, Any]] = []

    def invalid_reason(item: dict[str, Any]) -> str | None:
        text = item["text"]
        if item["confidence"] < min_confidence:
            return "low_confidence"
        if item["height"] < min_height:
            return "box_too_short"
        if basic["enable_center_y_region_filter"]:
            if item["center_y"] < image_height * float(basic["center_y_min_ratio"]):
                return "above_dialogue_region"
            if item["center_y"] > image_height * float(basic["center_y_max_ratio"]):
                return "below_dialogue_region"
        if basic["enable_time_text_filter"] and looks_like_time_text(text):
            return "time_text"
        if basic["enable_debug_text_filter"] and looks_like_debug_text(text):
            return "debug_text"
        if basic["enable_reward_text_filter"] and looks_like_reward_text(text):
            return "reward_text"
        if find_keyword(text, result_excluded):
            return "ocr_keyword"
        if basic["enable_max_x_filter"] and item["max_x"] >= image_width * float(basic["max_x_ratio"]):
            return "too_far_right"
        if basic["enable_max_width_filter"] and item["width"] > image_width * float(basic["max_width_ratio"]):
            return "too_wide"
        if basic["enable_low_conf_single_char_filter"]:
            threshold = float(basic["single_char_confidence_threshold"])
            if len(text) == 1 and item["confidence"] < threshold and not contains_emoji(text):
                return "low_confidence_single_char"
        if basic["enable_right_narrow_non_cjk_filter"]:
            min_x_ratio = float(basic["right_narrow_min_x_ratio"])
            max_width_ratio = float(basic["right_narrow_max_width_ratio"])
            if (
                item["min_x"] >= image_width * min_x_ratio
                and item["width"] <= image_width * max_width_ratio
                and not contains_cjk(text)
                and not contains_emoji(text)
            ):
                return "right_narrow_non_cjk"
        if basic["enable_non_cjk_non_emoji_filter"] and not contains_cjk(text) and not contains_emoji(text):
            return "non_cjk"
        return None

    def sender_for(item: dict[str, Any]) -> str:
        method = str(sender_detection["method"])
        sender2_min_x_ratio = float(sender_detection["sender2_min_x_ratio"])
        center_position_sender = "sender2" if item["center_x"] > image_width * sender2_min_x_ratio else "sender1"
        if method == "position":
            return "sender2" if item["min_x"] > image_width * sender2_min_x_ratio else "sender1"
        if method == "center_position":
            return center_position_sender
        if method == "bubble_color":
            hue = item.get("background_hue")
            saturation = float(item.get("background_saturation") or 0.0)
            value = float(item.get("background_value") or 0.0)
            min_saturation = float(sender_detection.get("bubble_color_min_saturation", 0.12))
            min_value = float(sender_detection.get("bubble_color_min_value", 0.20))
            hue_min = float(sender_detection.get("bubble_color_sender2_hue_min", 125))
            hue_max = float(sender_detection.get("bubble_color_sender2_hue_max", 170))
            blue_green_min = float(sender_detection.get("bubble_color_sender2_blue_green_min", 65))
            blue_red_ratio_min = float(sender_detection.get("bubble_color_sender2_blue_red_ratio_min", 1.60))
            blue_min_saturation = float(sender_detection.get("bubble_color_sender2_blue_min_saturation", 0.30))
            if hue is not None and saturation >= min_saturation and value >= min_value:
                is_sender2_hue = hue_min <= float(hue) <= hue_max
                blue_green_delta = float(item.get("background_blue_green_delta") or 0.0)
                blue_red_ratio = float(item.get("background_blue_red_ratio") or 0.0)
                is_sender2_blue = (
                    saturation >= blue_min_saturation
                    and blue_green_delta >= blue_green_min
                    and blue_red_ratio >= blue_red_ratio_min
                )
                return "sender2" if is_sender2_hue or is_sender2_blue else "sender1"
            return center_position_sender
        saturation_threshold = float(sender_detection["color_saturation_threshold"])
        if method == "color" and item.get("background_saturation", 0.0) >= saturation_threshold:
            return "sender2"
        if bool(sender_detection["enable_color_fallback"]) and item.get("background_saturation", 0.0) >= saturation_threshold:
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
            matched_keyword = find_keyword(item["text"], result_excluded) if reason == "ocr_keyword" else None
            excluded_entries.append(
                _debug_excluded_entry(
                    entry_type="item",
                    reason=reason,
                    text=item["text"],
                    raw_text=item.get("raw_text"),
                    image_width=image_width,
                    image_height=image_height,
                    bounds=_item_bounds(item, image_width, image_height),
                    matched_keyword=matched_keyword,
                    confidence=item.get("confidence"),
                )
            )
            continue
        item["sender"] = sender_for(item)
        item["debug_stage"] = "candidate_kept"
        candidates.append(item)

    filtered_groups: list[list[dict[str, Any]]] = []
    cutoff = False
    for group in _group_by_line(candidates, line_gap):
        text = _line_text(group, line_gap)
        if cutoff:
            excluded_entries.append(
                _debug_excluded_entry(
                    entry_type="line",
                    reason="after_line_and_below_cutoff",
                    text=text,
                    image_width=image_width,
                    image_height=image_height,
                    bounds=_round_bounds(
                        [
                            min(item["min_x"] for item in group) / image_width,
                            min(item["min_y"] for item in group) / image_height,
                            max(item["max_x"] for item in group) / image_width,
                            max(item["max_y"] for item in group) / image_height,
                        ]
                    ),
                )
            )
            continue
        matched_line_and_below = find_keyword(text, line_and_below)
        if matched_line_and_below:
            cutoff = True
            excluded_entries.append(
                _debug_excluded_entry(
                    entry_type="line",
                    reason="line_and_below_keyword",
                    text=text,
                    image_width=image_width,
                    image_height=image_height,
                    bounds=_round_bounds(
                        [
                            min(item["min_x"] for item in group) / image_width,
                            min(item["min_y"] for item in group) / image_height,
                            max(item["max_x"] for item in group) / image_width,
                            max(item["max_y"] for item in group) / image_height,
                        ]
                    ),
                    matched_keyword=matched_line_and_below,
                )
            )
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
        matched_line_excluded = find_keyword(content, line_excluded) if content else None
        if content and not matched_line_excluded:
            dialogues.append({"sender": item["sender"], "content": content})
            continue
        if content:
            excluded_entries.append(
                _debug_excluded_entry(
                    entry_type="dialogue",
                    reason="line_excluded_keyword",
                    text=content,
                    image_width=image_width,
                    image_height=image_height,
                    bounds=_round_bounds(
                        [
                            item["min_x"] / image_width,
                            item["min_y"] / image_height,
                            item["max_x"] / image_width,
                            item["max_y"] / image_height,
                        ]
                    ),
                    matched_keyword=matched_line_excluded,
                    extra={"sender": item["sender"]},
                )
            )

    name_candidates = [
        item["text"]
        for item in rule_items
        if item["center_y"] <= image_height * 0.12
        and 2 <= len(item["text"]) <= 16
        and contains_cjk(item["text"])
        and not looks_like_time_text(item["text"])
    ]
    return {
        "user_name": name_candidates[0] if name_candidates else "",
        "dialogues": dialogues,
        "debug": {"excluded": excluded_entries},
    }


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
    bottom_keywords = _keywords(rules["bottom_nav_keywords"])
    bottom_top_ratio = float(rules["bottom_nav_top_ratio"])
    bottom_accept_any_text = bool(rules["bottom_nav_accept_any_text"])
    top_nav_keywords = _keywords(rules["top_nav_keywords"])
    top_nav_top_ratio = float(rules["top_nav_top_ratio"])
    top_nav_bottom_ratio = float(rules["top_nav_bottom_ratio"])

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
            bracket_text = cleaned[1:close_index].replace(" ", "")
            bracket_text = bracket_text.translate(str.maketrans({"?": "+", "?": "+", "?": ".", "?": ".", "?": "F", "?": "f"}))
            if re.fullmatch(r"99(?:[+Ff.])?", bracket_text):
                return 99, cleaned[close_index + 1 :].strip()
            if re.fullmatch(r"[1-9]\d?", bracket_text):
                return int(bracket_text), cleaned[close_index + 1 :].strip()
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


def _strip_list_name_suffixes(name: str, rules: dict[str, Any]) -> str:
    cleaned = normalize_ocr_text(name).replace(" ", "")
    for suffix in _keywords(rules["name_suffix_keywords"]):
        suffix_text = normalize_ocr_text(suffix).replace(" ", "")
        if suffix_text and cleaned.endswith(suffix_text) and len(cleaned) > len(suffix_text):
            return cleaned[: -len(suffix_text)].strip(" :|-_")
    return name


def _clean_list_content(text: str, name: str) -> str:
    _unread, cleaned = _strip_unread_prefix(text)
    cleaned = normalize_ocr_text(cleaned).strip("[](){} ")
    if name and cleaned.startswith(name):
        cleaned = cleaned[len(name) :].lstrip(" :|-_")
    cleaned = re.sub(r"^\d{1,2}[:.]\d{2}", "", cleaned).lstrip(" :|-_")
    return cleaned


def _unread_count_from_text(text: str, max_unread_count: int) -> int | None:
    compact = normalize_ocr_text(text).replace(" ", "")
    compact = compact.translate(str.maketrans({"\uff0b": "+", "\ufe62": "+", "\u00b7": ".", "\u3002": ".", "\uff26": "F", "\uff46": "f"}))
    compact = compact.translate(str.maketrans({"\u2460": "1", "\u2461": "2", "\u2462": "3", "\u2463": "4", "\u2464": "5"}))
    if re.fullmatch(r"99(?:[+Ff.])?", compact):
        return min(99, max_unread_count)
    match = re.fullmatch(r"([1-9]\d?)(?:[+Ff.])?", compact)
    if not match:
        return None
    value = int(match.group(1))
    return value if 0 <= value <= max_unread_count else None


def _looks_like_unread_badge(
    item: dict[str, Any],
    image_width: int,
    image_height: int,
    right_meta_min: float,
    max_unread_count: int,
) -> bool:
    if item["min_x"] < right_meta_min:
        return False
    if item["width"] > image_width * 0.22 or item["height"] > image_height * 0.05:
        return False
    text = item["text"]
    return bool(text) and not looks_like_time_text(text) and _unread_count_from_text(text, max_unread_count) is not None


def _looks_like_left_ocr_unread_badge(
    item: dict[str, Any],
    image_width: int,
    image_height: int,
    rules: dict[str, Any],
    max_unread_count: int,
) -> bool:
    if item["center_x"] < image_width * float(rules["left_badge_min_ratio"]):
        return False
    if item["center_x"] > image_width * float(rules["left_badge_max_ratio"]):
        return False
    if item["width"] > image_width * 0.08 or item["height"] > image_height * 0.05:
        return False
    return _unread_count_from_text(item["text"], max_unread_count) is not None


def _detect_red_badge_bounds(
    image_array: np.ndarray,
    row_bounds: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    rules: dict[str, Any],
) -> tuple[int, int, int, int] | None:
    row_left, row_top, row_right, row_bottom = row_bounds
    badge_position = str(rules["badge_position"]).strip().lower()
    if badge_position == "left":
        search_left = max(0, int(image_width * float(rules["left_badge_min_ratio"])))
        search_right = min(image_width, int(image_width * float(rules["left_badge_max_ratio"])))
        search_top = max(0, row_top - int(image_height * float(rules["left_badge_extra_top_ratio"])))
        search_bottom = min(image_height, row_bottom)
        search_height_ratio = rules.get("left_red_badge_search_height_ratio")
        if search_height_ratio is not None:
            search_height_px = max(
                int(rules.get("left_red_badge_search_min_height_px", 0)),
                int(round(image_height * float(search_height_ratio))),
            )
            if search_height_px > 0:
                search_bottom = min(search_bottom, search_top + search_height_px)
    else:
        search_left = max(row_left, int(image_width * float(rules["right_badge_min_ratio"])))
        search_right = min(image_width, row_right)
        search_top = max(0, row_top)
        search_bottom = min(image_height, row_bottom)
    search_box = (
        search_left,
        search_top,
        max(search_left + 1, search_right),
        max(search_top + 1, search_bottom),
    )
    crop = image_array[search_box[1] : search_box[3], search_box[0] : search_box[2]]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    red_saturation_min = int(rules["red_badge_saturation_min"])
    red_value_min = int(rules["red_badge_value_min"])
    min_area = float(rules.get("red_badge_min_area", max(30.0, image_width * 0.04)))
    min_width = int(rules.get("red_badge_min_width", 6))
    min_height = int(rules.get("red_badge_min_height", 6))
    max_width_ratio = float(rules.get("red_badge_max_width_ratio", 0.09))
    max_height_ratio = float(rules.get("red_badge_max_height_ratio", 0.09))
    min_aspect_ratio = float(rules.get("red_badge_min_aspect_ratio", 0.55))
    max_aspect_ratio = float(rules.get("red_badge_max_aspect_ratio", 1.8))
    min_circularity = float(rules.get("red_badge_min_circularity", 0.30))
    mask1 = cv2.inRange(hsv, np.array([0, red_saturation_min, red_value_min]), np.array([14, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([165, red_saturation_min, red_value_min]), np.array([180, 255, 255]))
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
        if area < min_area:
            continue
        if width < min_width or height < min_height or width > image_width * max_width_ratio or height > image_width * max_height_ratio:
            continue
        ratio = width / max(1, height)
        if ratio > max_aspect_ratio or ratio < min_aspect_ratio:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < min_circularity:
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
    pad = int(rules["badge_crop_padding"])
    region = [
        max(0.0, (left - pad) / image_width),
        max(0.0, (top - pad) / image_height),
        min(1.0, (right - left + pad * 2) / image_width),
        min(1.0, (bottom - top + pad * 2) / image_height),
    ]
    config = dict(base_ocr_config)
    config["region"] = region
    config["preprocess"] = "original"
    config["scale"] = max(1, int(rules["badge_ocr_scale"]))
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


def _unread_count_from_badge_text(text: str, max_unread_count: int) -> int | None:
    compact = normalize_ocr_text(text).replace(" ", "")
    compact = compact.translate(str.maketrans({"一": "1", "丨": "1", "l": "1", "I": "1", "|": "1"}))
    return _unread_count_from_text(compact, max_unread_count)


def _read_left_badge_number_from_row_crop(
    source_image: ImageInput,
    row_bounds: tuple[int, int, int, int],
    image_size: tuple[int, int],
    base_ocr_config: dict[str, Any],
    max_unread_count: int,
    rules: dict[str, Any],
) -> int | None:
    def read_from_region(
        *,
        crop_left: float,
        crop_width: float,
        crop_top_offset: float,
        crop_height: float,
        scale: int,
        min_confidence: float,
    ) -> int | None:
        crop_top = max(0.0, (row_top / image_height) - crop_top_offset)
        normalized_width = min(crop_width, max(0.0, 1.0 - crop_left))
        normalized_height = min(crop_height, max(0.0, 1.0 - crop_top))
        if normalized_width <= 0.0 or normalized_height <= 0.0:
            return None
        config = dict(base_ocr_config)
        config["region"] = [crop_left, crop_top, normalized_width, normalized_height]
        config["preprocess"] = "original"
        config["scale"] = max(1, scale)
        config["rotate_to_portrait"] = False
        best_value: int | None = None
        best_confidence = -1.0
        for item in ocr(source_image, config):
            value = _unread_count_from_badge_text(str(item.get("text", "")), max_unread_count)
            if value is None:
                continue
            confidence = float(item.get("confidence") or 0.0)
            if confidence < min_confidence:
                continue
            if confidence > best_confidence:
                best_value = value
                best_confidence = confidence
        return best_value

    image_width, image_height = image_size
    _row_left, row_top, _row_right, _row_bottom = row_bounds
    primary_value = read_from_region(
        crop_left=float(rules.get("left_badge_ocr_left_ratio", rules["left_badge_min_ratio"])),
        crop_width=float(rules.get("left_badge_ocr_width_ratio", 0.08)),
        crop_top_offset=float(rules.get("left_badge_ocr_top_offset_ratio", 0.012)),
        crop_height=float(rules.get("left_badge_ocr_height_ratio", 0.045)),
        scale=int(rules.get("left_badge_ocr_scale", rules["badge_ocr_scale"])),
        min_confidence=float(rules.get("left_badge_ocr_min_confidence", 0.50)),
    )
    if primary_value is not None:
        return primary_value
    return read_from_region(
        crop_left=float(rules.get("left_badge_retry_left_ratio", rules.get("left_badge_ocr_left_ratio", rules["left_badge_min_ratio"]))),
        crop_width=float(rules.get("left_badge_retry_width_ratio", rules.get("left_badge_ocr_width_ratio", 0.08))),
        crop_top_offset=float(rules.get("left_badge_retry_top_offset_ratio", rules.get("left_badge_ocr_top_offset_ratio", 0.012))),
        crop_height=float(rules.get("left_badge_retry_height_ratio", rules.get("left_badge_ocr_height_ratio", 0.045))),
        scale=int(rules.get("left_badge_retry_scale", rules.get("left_badge_ocr_scale", rules["badge_ocr_scale"]))),
        min_confidence=float(rules.get("left_badge_retry_min_confidence", rules.get("left_badge_ocr_min_confidence", 0.50))),
    )


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
    right_meta_min_ratio = float(rules["right_meta_min_ratio"])
    strip_left = max(
        float(rules["row_strip_left_min_ratio"]),
        right_meta_min_ratio + float(rules["row_strip_extra_left_ratio"]),
    )
    strip_width = min(float(rules["row_strip_width_ratio"]), max(0.0, 1.0 - strip_left))
    strip_top = max(0.0, row_top / image_height)
    row_height = row_bottom - row_top
    strip_bottom_px = min(
        row_bottom,
        bottom_nav_top - int(rules["row_strip_bottom_nav_margin"]),
        row_top
        + max(
            int(rules["row_strip_min_height_px"]),
            min(
                int(row_height * float(rules["row_strip_height_ratio"])),
                int(image_height * float(rules["row_strip_max_height_ratio"])),
            ),
        ),
    )
    strip_bottom = strip_bottom_px / image_height
    if strip_bottom <= strip_top or strip_width <= 0.0:
        return None
    config = dict(base_ocr_config)
    config["region"] = [strip_left, strip_top, strip_width, strip_bottom - strip_top]
    config["preprocess"] = "original"
    config["scale"] = max(1, int(rules["row_badge_ocr_scale"]))
    config["rotate_to_portrait"] = False
    best_value: int | None = None
    best_confidence = -1.0
    for item in ocr(source_image, config):
        value = _unread_count_from_text(str(item.get("text", "")), max_unread_count)
        if value is None:
            continue
        confidence = float(item.get("confidence") or 0.0)
        if confidence < float(rules["row_badge_min_confidence"]):
            continue
        if confidence > best_confidence:
            best_value = value
            best_confidence = confidence
    if best_value is not None:
        return best_value
    return None


def _looks_like_list_content(text: str, image_width: int, width: int, rules: dict[str, Any]) -> bool:
    compact = normalize_ocr_text(text).replace(" ", "")
    if not compact or looks_like_time_text(compact):
        return False
    if re.search(r"[锛屻€傦紒锛??,:锛?]", compact):
        return True
    min_length = int(rules["content_like_name_min_length"])
    min_width_ratio = float(rules["content_like_name_min_width_ratio"])
    return len(compact) >= min_length and width >= image_width * min_width_ratio


def _looks_like_list_name(text: str) -> bool:
    compact = normalize_ocr_text(text).replace(" ", "")
    if not compact or looks_like_time_text(compact):
        return False
    if contains_cjk(compact):
        return True
    return re.fullmatch(r"[A-Za-z0-9_.-]{1,16}", compact) is not None


def _build_list_line(group: list[dict[str, Any]], image_width: int, line_gap: int, rules: dict[str, Any]) -> dict[str, Any]:
    avatar_max_x_ratio = float(rules["avatar_max_x_ratio"])
    text_right_max = image_width * float(rules["text_right_max_ratio"])
    right_meta_min = image_width * float(rules["right_meta_min_ratio"])
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
    if first_x < image_width * float(rules["text_left_min_ratio"]):
        return False
    name = _strip_list_name_suffixes(_extract_list_name(text), rules)
    if not name or not _looks_like_list_name(name):
        return False
    if _looks_like_list_content(name, image_width, int(line["left_width"]), rules):
        return False
    return 1 <= len(name.replace(" ", "")) <= 16


def build_list_result(
    rule_items: list[dict[str, Any]],
    image_size: tuple[int, int],
    rules: dict[str, Any] | None = None,
    source_image: ImageInput | None = None,
    ocr_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rules = dict(rules or {})
    image_width, image_height = image_size
    line_gap = int(rules["line_gap"])
    min_confidence = float(rules["min_confidence"])
    min_height = image_height * float(rules["min_box_height_ratio"])
    top = image_height * float(rules["main_top_ratio"])
    bottom = image_height * float(rules["main_bottom_ratio"])
    content_gap_threshold = int(
        max(
            float(rules["content_gap_min"]),
            min(
                float(rules["content_gap_max"]),
                image_height * float(rules["content_gap_ratio"]),
            ),
        )
    )
    bottom_nav_top = image_height * float(rules["bottom_nav_top_ratio"])
    system_keywords = _keywords(rules["system_keywords"])
    excluded_keywords = _keywords(rules["excluded_keywords"])
    content_excluded_keywords = _keywords(rules["content_excluded_keywords"])
    unread_count_excluded_keywords = _keywords(rules.get("unread_count_excluded_keywords", ()))
    bottom_nav_keywords = _keywords(rules["bottom_nav_keywords"])
    max_unread_count = int(rules["max_unread_count"])
    right_meta_min = image_width * float(rules["right_meta_min_ratio"])
    top_nav_top = image_height * float(rules["top_nav_top_ratio"])
    top_nav_bottom = image_height * float(rules["top_nav_bottom_ratio"])
    enable_red_badge_detection = bool(rules.get("enable_red_badge_detection", True))
    bottom_nav = _build_bottom_nav_result(rule_items, image_size, rules)
    bottom_nav_names = [item["name"] for item in bottom_nav]
    image_array = np.array(load_image(source_image).convert("RGB")) if source_image is not None else None
    excluded_entries: list[dict[str, Any]] = []

    pre_candidates: list[dict[str, Any]] = []
    for item in rule_items:
        reason: str | None = None
        if item["confidence"] < min_confidence:
            reason = "low_confidence"
        elif (
            item["height"] < min_height
            and not _looks_like_unread_badge(item, image_width, image_height, right_meta_min, max_unread_count)
        ):
            reason = "box_too_short"
        elif not (top <= item["center_y"] <= bottom):
            reason = "outside_list_region"
        elif item["min_x"] >= image_width * 0.96:
            reason = "too_far_right"
        if reason:
            excluded_entries.append(
                _debug_excluded_entry(
                    entry_type="item",
                    reason=reason,
                    text=item["text"],
                    raw_text=item.get("raw_text"),
                    image_width=image_width,
                    image_height=image_height,
                    bounds=_item_bounds(item, image_width, image_height),
                    confidence=item.get("confidence"),
                )
            )
            continue
        pre_candidates.append(item)
    candidates = _dedupe_rule_items(pre_candidates)

    lines: list[dict[str, Any]] = []
    for group in _group_by_line(candidates, line_gap):
        line = _build_list_line(group, image_width, line_gap, rules)
        text = line["left_text"]
        if not text:
            excluded_entries.append(
                _debug_excluded_entry(
                    entry_type="line",
                    reason="empty_left_text",
                    text="",
                    image_width=image_width,
                    image_height=image_height,
                    bounds=_round_bounds(
                        [
                            line["min_x"] / image_width,
                            line["min_y"] / image_height,
                            line["max_x"] / image_width,
                            line["max_y"] / image_height,
                        ]
                    ),
                )
            )
            continue
        is_top_nav_line = top_nav_bottom > top_nav_top and top_nav_top <= line["center_y"] <= top_nav_bottom
        if _looks_like_bottom_nav(text, tuple(bottom_nav_names)) and (
            line["center_y"] >= bottom_nav_top or is_top_nav_line
        ):
            excluded_entries.append(
                _debug_excluded_entry(
                    entry_type="line",
                    reason="bottom_nav_line",
                    text=text,
                    image_width=image_width,
                    image_height=image_height,
                    bounds=_round_bounds(
                        [
                            line["min_x"] / image_width,
                            line["min_y"] / image_height,
                            line["max_x"] / image_width,
                            line["max_y"] / image_height,
                        ]
                    ),
                )
            )
            continue
        matched_system_keyword = find_keyword(text, system_keywords)
        matched_excluded_keyword = find_keyword(text, excluded_keywords)
        if line["center_y"] < image_height * 0.14 and (matched_system_keyword or matched_excluded_keyword):
            reason = "system_keyword_line" if matched_system_keyword else "excluded_keyword_line"
            excluded_entries.append(
                _debug_excluded_entry(
                    entry_type="line",
                    reason=reason,
                    text=text,
                    image_width=image_width,
                    image_height=image_height,
                    bounds=_round_bounds(
                        [
                            line["min_x"] / image_width,
                            line["min_y"] / image_height,
                            line["max_x"] / image_width,
                            line["max_y"] / image_height,
                        ]
                    ),
                    matched_keyword=matched_system_keyword or matched_excluded_keyword,
                )
            )
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
            excluded_entries.append(
                _debug_excluded_entry(
                    entry_type="row",
                    reason="row_below_bottom_nav",
                    text=header_line["left_text"],
                    image_width=image_width,
                    image_height=image_height,
                    bounds=_round_bounds(
                        [
                            header_line["min_x"] / image_width,
                            header_line["min_y"] / image_height,
                            row["max_x"] / image_width,
                            row["last_max_y"] / image_height,
                        ]
                    ),
                )
            )
            continue
        unread_count, header_text = _strip_unread_prefix(header_line["left_text"])
        name = _strip_list_name_suffixes(_extract_list_name(header_text), rules)
        matched_name_excluded = find_keyword(name, excluded_keywords) if name else None
        if not name or matched_name_excluded:
            excluded_entries.append(
                _debug_excluded_entry(
                    entry_type="row",
                    reason="invalid_name" if not name else "excluded_name_keyword",
                    text=header_line["left_text"],
                    image_width=image_width,
                    image_height=image_height,
                    bounds=_round_bounds(
                        [
                            header_line["min_x"] / image_width,
                            header_line["min_y"] / image_height,
                            row["max_x"] / image_width,
                            row["last_max_y"] / image_height,
                        ]
                    ),
                    matched_keyword=matched_name_excluded,
                )
            )
            continue
        content_parts = [
            _clean_list_content(line["left_text"], name)
            for line in row["content_lines"]
            if line["left_text"] and not looks_like_time_text(line["left_text"])
        ]
        content = "".join(part for part in content_parts if part).strip()
        if name and content.startswith(name):
            content = content[len(name) :].lstrip(" :|-_")
        matched_content_excluded = find_keyword(content, content_excluded_keywords)
        if matched_content_excluded:
            excluded_entries.append(
                _debug_excluded_entry(
                    entry_type="row",
                    reason="content_excluded_keyword",
                    text="".join(part for part in [name, content] if part),
                    image_width=image_width,
                    image_height=image_height,
                    bounds=_round_bounds(
                        [
                            header_line["min_x"] / image_width,
                            header_line["min_y"] / image_height,
                            row["max_x"] / image_width,
                            row["last_max_y"] / image_height,
                        ]
                    ),
                    matched_keyword=matched_content_excluded,
                )
            )
            continue
        row_text = "".join([name, content, *[line["left_text"] for line in row["content_lines"]]])
        suppress_unread_count = bool(find_keyword(row_text, unread_count_excluded_keywords))
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
        badge_position = str(rules["badge_position"]).strip().lower()
        left_red_badge_force_default = bool(rules.get("left_red_badge_force_default_unread_count", False))
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
                if not _looks_like_unread_badge(item, image_width, image_height, right_meta_min, max_unread_count):
                    continue
                if not (search_top <= item["center_y"] <= search_bottom):
                    continue
                value = _unread_count_from_text(item["text"], max_unread_count)
                if value is not None:
                    unread_candidates.append(value)
        left_red_badge_bounds: tuple[int, int, int, int] | None = None
        if (
            badge_position == "left"
            and not suppress_unread_count
            and enable_red_badge_detection
            and image_array is not None
        ):
            left_red_badge_bounds = _detect_red_badge_bounds(
                image_array,
                (row_left, row_top, row_right, detection_row_bottom),
                image_width,
                image_height,
                rules,
            )
            if left_red_badge_bounds is not None and left_red_badge_force_default:
                default_unread_count = int(rules["badge_default_unread_count"])
                if default_unread_count > 0:
                    unread_candidates.append(default_unread_count)
        if (
            badge_position != "left"
            and not unread_candidates
            and enable_red_badge_detection
            and image_array is not None
            and ocr_config is not None
        ):
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
        if badge_position == "left" and not unread_candidates and not suppress_unread_count:
            search_top = max(0, row_top - int(image_height * float(rules["left_badge_extra_top_ratio"])))
            search_bottom = min(image_height, detection_row_bottom)
            for item in candidates:
                if not (search_top <= item["center_y"] <= search_bottom):
                    continue
                if not _looks_like_left_ocr_unread_badge(item, image_width, image_height, rules, max_unread_count):
                    continue
                value = _unread_count_from_text(item["text"], max_unread_count)
                if value is not None:
                    unread_candidates.append(value)
                    break
        if (
            badge_position == "left"
            and not unread_candidates
            and not suppress_unread_count
            and not (left_red_badge_force_default and left_red_badge_bounds is not None)
            and ocr_config is not None
            and source_image is not None
        ):
            value = _read_left_badge_number_from_row_crop(
                source_image,
                (row_left, row_top, row_right, detection_row_bottom),
                image_size,
                ocr_config,
                max_unread_count,
                rules,
            )
            if value is not None:
                unread_candidates.append(value)
        if (
            badge_position == "left"
            and not suppress_unread_count
            and enable_red_badge_detection
            and image_array is not None
            and ocr_config is not None
        ):
            badge_bounds = left_red_badge_bounds or _detect_red_badge_bounds(
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
                elif not unread_candidates:
                    default_unread_count = int(rules["badge_default_unread_count"])
                    if default_unread_count > 0:
                        unread_candidates.append(default_unread_count)
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
        if suppress_unread_count:
            unread_count = None
            unread_candidates = []
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
    return {"message_list": rows, "bottom_nav": bottom_nav, "debug": {"excluded": excluded_entries}}


def _points_value_from_text(text: str, keywords: tuple[str, ...]) -> str | None:
    compact = normalize_ocr_text(text).replace(" ", "").replace(",", "")
    if not compact:
        return None
    numeric = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)", compact)
    if numeric:
        return numeric.group(1)
    for keyword in keywords:
        normalized_keyword = normalize_ocr_text(keyword).replace(" ", "").replace(",", "")
        if not normalized_keyword or normalized_keyword not in compact:
            continue
        head = compact.split(normalized_keyword, 1)[0]
        prefix_match = re.search(r"([0-9]+(?:\.[0-9]+)?)$", head)
        if prefix_match and not looks_like_time_text(compact):
            return prefix_match.group(1)
        tail = compact.split(normalized_keyword, 1)[1]
        match = re.search(r"[:：]?([0-9]+(?:\.[0-9]+)?)", tail)
        if match:
            return match.group(1)
    return None


def build_points_result(
    rule_items: list[dict[str, Any]],
    image_size: tuple[int, int],
    rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rules = dict(rules or {})
    image_width, image_height = image_size
    line_gap = int(rules["line_gap"])
    min_confidence = float(rules["min_confidence"])
    min_height = image_height * float(rules["min_box_height_ratio"])
    keywords = _keywords(rules["keywords"])

    candidates = [
        item
        for item in rule_items
        if item["confidence"] >= min_confidence and item["height"] >= min_height
    ]
    groups = _group_by_line(candidates, line_gap)

    for group in groups:
        text = _line_text(group, line_gap)
        if not find_keyword(text, keywords):
            continue
        keyword_center_x = sum(item["center_x"] for item in group) / len(group)
        keyword_center_y = sum(item["center_y"] for item in group) / len(group)
        keyword_min_x = min(item["min_x"] for item in group)
        keyword_max_x = max(item["max_x"] for item in group)
        search_items = list(group)
        horizontal_padding = image_width * float(rules["neighbor_x_ratio"])
        near_items = [
            item
            for item in candidates
            if abs(item["center_y"] - keyword_center_y) <= image_height * float(rules["neighbor_y_ratio"])
            and item["max_x"] >= keyword_min_x - horizontal_padding
            and item["min_x"] <= keyword_max_x + horizontal_padding
            and item not in group
        ]
        search_items.extend(near_items)
        ranked_values: list[tuple[float, float, str]] = []
        for item in search_items:
            value = _points_value_from_text(item["text"], keywords)
            if value is None:
                continue
            item_has_keyword = find_keyword(item["text"], keywords) is not None
            distance = abs(item["center_y"] - keyword_center_y) + abs(item["center_x"] - keyword_center_x) * 0.5
            ranked_values.append((0.0 if item_has_keyword else 1.0, distance, -float(item.get("confidence") or 0.0), value))
        value = min(ranked_values)[3] if ranked_values else None
        raw_text = "".join([text, *[item["text"] for item in sorted(near_items, key=lambda part: part["min_x"])]])
        bounds = _round_bounds([
            min(item["min_x"] for item in group) / image_width,
            min(item["min_y"] for item in group) / image_height,
            max(item["max_x"] for item in group) / image_width,
            max(item["max_y"] for item in group) / image_height,
        ])
        return {"points": value or "", "balance": value or "", "raw_text": raw_text, "bounds": bounds}

    raw_text = "".join(_line_text(group, line_gap) for group in groups)
    return {"points": "", "balance": "", "raw_text": raw_text, "bounds": None}


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
    configured_modes = config["ocr_modes"]
    if not isinstance(configured_modes, dict):
        raise ValueError("ocr_modes must be a dict")
    configured = configured_modes[mode]
    if not isinstance(configured, dict):
        raise ValueError(f"ocr_modes.{mode} must be a dict")
    return copy.deepcopy(configured)


def run_ocr_mode(
    image: ImageInput,
    config: dict[str, Any],
    mode: str,
    *,
    region: Any | None = None,
) -> dict[str, Any]:
    if mode not in {"dialogue", "list", "points"}:
        raise ValueError("mode must be dialogue, list, or points")
    base_ocr = config["ocr"]
    if not isinstance(base_ocr, dict):
        raise ValueError("ocr must be a dict")
    mode_config = _mode_config(config, mode)
    preprocess = mode_config["preprocess"]
    if not isinstance(preprocess, dict):
        raise ValueError(f"ocr_modes.{mode}.preprocess must be a dict")
    portrait_image = ensure_portrait_image(image)
    primary_ocr_config = dict(base_ocr)
    primary_ocr_config["region"] = region
    primary_ocr_config["preprocess"] = "original"
    primary_ocr_config["scale"] = preprocess["scale"]
    primary_ocr_config["rotate_to_portrait"] = False
    primary_items = ocr(portrait_image, primary_ocr_config)

    raw_items = primary_items
    rule_items, normalized_items, image_size = _to_rule_items(portrait_image, raw_items, region=region)
    rules = mode_config["rules"]
    if not isinstance(rules, dict):
        raise ValueError(f"ocr_modes.{mode}.rules must be a dict")
    if mode == "dialogue":
        result = build_dialogue_result(rule_items, image_size, rules)
    elif mode == "list":
        result = build_list_result(
            rule_items,
            image_size,
            rules,
            source_image=portrait_image,
            ocr_config=primary_ocr_config,
        )
    else:
        result = build_points_result(rule_items, image_size, rules)
    return {
        "mode": mode,
        "items": rule_items,
        "ocr_items": normalized_items,
        "result": result,
    }


