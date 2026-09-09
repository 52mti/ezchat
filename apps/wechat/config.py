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
        "buttons": "apps/wechat/templates/buttons",
        "screens": "apps/wechat/templates/screens",
    },
    "match_threshold": 0.7,
    "popup_match_threshold": 0.8,
    "enable_enhanced_template_match": True,
    "template_match_region_expand_ratio": 0.03,
    "template_match_scales": [0.92, 0.96, 1.0, 1.04, 1.08],
    "template_match_channels": ["gray_norm", "gray", "edge"],
    **DEFAULT_RUNTIME_CONFIG,
    "step_timeout": 10,
    "persona": dict(DEFAULT_PERSONA_CONFIG),
    "circuit_breaker_threshold": 5,
    "recover_failure_limit": 3,
}
