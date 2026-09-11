from services.defaults import DEFAULT_PERSONA_CONFIG, DEFAULT_RUNTIME_CONFIG

from .ocr_config import LLM_DIALOGUE_CONFIG, OCR_MODES


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
    "dialogue_engine": "llm_vision",
    "dialogue_llm": LLM_DIALOGUE_CONFIG,
    "templates": {
        "buttons": "apps/huanyou/templates/buttons",
        "screens": "apps/huanyou/templates/screens",
    },
    "match_threshold": 0.7,
    "popup_match_threshold": 0.8,
    "enable_enhanced_template_match": True,
    "template_match_region_expand_ratio": 0.03,
    "template_match_scales": [0.92, 0.96, 1.0, 1.04, 1.08],
    "template_match_channels": ["gray_norm", "gray", "edge"],
    "template_match_thresholds": {
        "button.next_message": 0.7,
        "screen.chat_window": 0.7,
    },
    **DEFAULT_RUNTIME_CONFIG,
    "idle_poll_interval_seconds": 20,
    "keep_awake_interval_seconds": 20,
    "enable_runtime_coordinate_cache": True,
    "enable_screenshot_reuse": True,
    "enable_screenshot_detection_reuse": True,
    "enable_screenshot_ocr_reuse": True,
    "enable_compact_list_ocr_region": True,
    "enable_dialogue_name_region": True,
    "enable_dialogue_message_region": True,
    "enable_next_new_greeting_fast_path": True,
    "enable_bottom_message_recovery": True,
    "ocr_reuse_image_mean_threshold": 0.5,
    "ocr_reuse_image_pixel_threshold": 5,
    "ocr_reuse_image_ratio_threshold": 0.001,
    "step_timeout": 10,
    "persona": dict(DEFAULT_PERSONA_CONFIG),
    "circuit_breaker_threshold": 5,
    "recover_failure_limit": 3,
}
