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
    assert payload["model_observation_policy"]["must_stay_within_configured_match_region"] is True
    assert payload["model_observation_policy"]["local_pixel_recheck_required"] is True
    assert payload["model_observation_policy"]["auto_click_from_model_output"] is False
