from __future__ import annotations

import json
from pathlib import Path

from apps.xinyu.config import DEFAULT_CONFIG


def test_xinyu_template_descriptions_keep_photo_fallback_constrained():
    path = Path(DEFAULT_CONFIG["template_description_config"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    photo = payload["templates"]["button.photo"]

    assert photo["reference_template"] == "btn_照片按钮.jpg"
    assert photo["fallback"]["enabled"] is True
    policy = payload["model_observation_policy"]
    assert policy["output_coordinate_space"] == "full_image_normalized_999_point"
    assert policy["coordinate_scale"] == 999
    assert policy["must_stay_within_configured_match_region"] is True
    assert policy["response_protocol"] == "autoglm_phone_action"
    assert policy["model_point_is_candidate"] is True
    assert policy["local_pixel_recheck_required"] is True
    assert policy["auto_click_from_model_output"] is False
    assert 'do(action="Tap", element=[x,y])' in payload["model_prompt_template"]
    assert "bbox" not in payload["model_prompt_template"]
