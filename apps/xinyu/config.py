from apps.defaults import DEFAULT_PERSONA_CONFIG, DEFAULT_RUNTIME_CONFIG

from .ocr_config import OCR_MODES


DEFAULT_CONFIG = {
    "ocr": {
        "lang": "ch",
        "runtime_enabled": True,
        "backend": "auto",
        "model_tier": "lite",
        "use_gpu": True,
        "text_detection_model_name": "PP-OCRv5_mobile_det",
        "text_recognition_model_name": "PP-OCRv5_mobile_rec",
        "det_db_thresh": 0.3,
        "rec_thresh": 0.5,
        "regions": {
            "message_list": [0.0, 0.1, 1.0, 0.9],
            "chat_area": [0.0, 0.1, 1.0, 0.8],
        },
    },
    "ocr_modes": OCR_MODES,
    "templates": {
        "buttons": "apps/xinyu/templates/buttons",
        "screens": "apps/xinyu/templates/screens",
    },
    "match_threshold": 0.7,
    "popup_match_threshold": 0.8,
    "template_match_regions": {
        "button.back": [0.0, 0.911, 0.993, 0.089],
        "button.input_box": [0.651, 0.737, 0.349, 0.188],
        "button.microphone": [0.012, 0.555, 0.984, 0.168],
        "button.new_greeting": [0.011, 0.687, 0.98, 0.171],
        "button.send": [0.63, 0.305, 0.367, 0.376],
        "screen.chat_list": [0.0, 0.023, 0.489, 0.127],
        "screen.chat_window": [0.007, 0.476, 0.261, 0.523],
        "screen.my": [0.468, 0.686, 0.488, 0.187],
    },
    "template_match_thresholds": {
        "screen.chat_window": 0.7,
    },
    **DEFAULT_RUNTIME_CONFIG,
    "idle_poll_interval_seconds": 5,
    "step_timeout": 10,
    "enable_runtime_coordinate_cache": True,
    "enable_bottom_nav_anchor_reuse": True,
    "enable_keep_awake_message_anchor": True,
    "enable_compact_list_ocr_region": True,
    "enable_compact_dialogue_ocr_region": True,
    "enable_unread_tab_band_cache": True,
    "list_compact_top_margin": 0.02,
    "list_compact_bottom_margin": 0.02,
    "list_compact_min_height": 0.2,
    "dialogue_message_input_box_top_margin": 0.04,
    "dialogue_compact_min_height": 0.2,
    "unread_tab_band_vertical_scale": 0.6,
    "persona": dict(DEFAULT_PERSONA_CONFIG),
    "circuit_breaker_threshold": 5,
    "recover_failure_limit": 3,
}
