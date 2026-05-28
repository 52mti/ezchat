from __future__ import annotations

from PIL import Image, ImageDraw

from vision.matcher import match


def _pattern_image() -> tuple[Image.Image, Image.Image]:
    source = Image.new("RGB", (80, 80), "black")
    draw = ImageDraw.Draw(source)
    draw.rectangle((30, 25, 49, 44), fill="white")
    draw.line((30, 25, 49, 44), fill="black", width=2)
    draw.line((49, 25, 30, 44), fill="black", width=2)
    template = source.crop((30, 25, 50, 45))
    return source, template


def test_match_returns_normalized_coordinates():
    source, template = _pattern_image()

    result = match(source, template, threshold=0.8)

    assert result is not None
    assert abs(result["x"] - 0.375) < 0.01
    assert abs(result["y"] - 0.3125) < 0.01
    assert abs(result["w"] - 0.25) < 0.01
    assert abs(result["h"] - 0.25) < 0.01
    assert result["score"] >= 0.8


def test_match_returns_none_below_threshold():
    source, _ = _pattern_image()
    missing = Image.new("RGB", (20, 20), "white")

    result = match(source, missing, threshold=0.99)

    assert result is None
