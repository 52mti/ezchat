from __future__ import annotations


# “欢游” 的 OCR 规则也单独放这里。
# 现在先保留和微信一致的基础结构，后续如果这个 App 的界面和列表规则不同，
# 直接改这里，不影响别的 App。


# 下面这些关键词会在 vision/ocr_rules.py 里生效：
# - `ocr_result_excluded_keywords`：对话页单个 OCR 文字块命中就直接丢掉。
# - `line_excluded_keywords`：对话页合并成一句后，命中就整条丢掉。
# - `line_and_below_keywords`：对话页命中后，当前行以及下面的内容都不再算聊天内容。
# - `system_keywords`：列表页顶部区域命中时，按系统/入口行排除。
# - `excluded_keywords`：列表页联系人名字命中时，整行丢掉。
# - `content_excluded_keywords`：列表页内容摘要命中时，整行丢掉。
# 修改这里以后，当前 App 的 OCR 后处理规则会随之改变，不会影响其他 App。
DIALOGUE_OCR_RESULT_EXCLUDED_KEYWORDS = [
    "记录点滴",
    "折扣商店",
    "活动倒计时",
    "建议完成真人认证",
    "为了真实交友体验",
    "输入你想说的话",
    "绑定关系",
    "开启日记",
    "服务号合集",
    "聊天广场",
    "家族群聊",
    "加入家族",
    "去聊天",
    "免费匹配机会",
    "点击领取",
    "前往",
    "赠礼物",
]

DIALOGUE_LINE_EXCLUDED_KEYWORDS = [
    "回复即可开启缘分之旅",
    "已认证",
    "上传头像",
    "搭讪礼物",
    "金币奖励",
    "主动给对方",
    "本人真实",
    "丘比特牵线成功",
    "真人认证",
    "开始游戏",
    "你们的共同点",
    "常用语",
]

DIALOGUE_LINE_AND_BELOW_KEYWORDS = [
    "输入你想说的话",
    "推荐灵感",
    "真实性及体验",
]

# 列表页规则会在 run_ocr_mode(..., "list") 之后进入 build_list_result()。
LIST_SYSTEM_KEYWORDS = [
    "聊天",
    "群聊",
    "心动",
    "开启",
    "记录点滴",
    "折扣商店",
    "活动倒计时",
    "建议完成真人认证",
    "为了真实交友体验",
    "前往",
    "输入你想说的话",
    "输入",
    "绑定关系",
    "开启日记",
    "日记",
    "官方",
    "服务号",
    "服务号合集",
    "聊天广场",
    "家族群聊",
    "加入家族",
    "去聊天",
    "免费匹配机会",
    "点击领取",
]

LIST_EXCLUDED_KEYWORDS = [
    "家族群聊",
    "聊天广场",
    "服务号合集",
    "官方群",
    "官方",
    "点击领取",
    "免费匹配机会",
    "加入家族",
    "去聊天",
]

LIST_CONTENT_EXCLUDED_KEYWORDS = ["推荐灵感"]


DIALOGUE_OCR_MODE = {
    "preprocess": {
        "mode": "original",
        "scale": 2,
        "rotate_to_portrait": True,
    },
    "rules": {
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
        "ocr_result_excluded_keywords": DIALOGUE_OCR_RESULT_EXCLUDED_KEYWORDS,
        "line_excluded_keywords": DIALOGUE_LINE_EXCLUDED_KEYWORDS,
        "line_and_below_keywords": DIALOGUE_LINE_AND_BELOW_KEYWORDS,
        "basic_filter": {
            "enable_time_text_filter": True,
            "enable_debug_text_filter": True,
            "enable_reward_text_filter": True,
            "enable_center_y_region_filter": True,
            "center_y_min_ratio": 0.18,
            "center_y_max_ratio": 0.88,
            "enable_max_x_filter": True,
            "max_x_ratio": 0.90,
            "enable_max_width_filter": True,
            "max_width_ratio": 0.80,
            "enable_low_conf_single_char_filter": True,
            "single_char_confidence_threshold": 0.90,
            "enable_right_narrow_non_cjk_filter": True,
            "right_narrow_min_x_ratio": 0.88,
            "right_narrow_max_width_ratio": 0.12,
            "enable_non_cjk_non_emoji_filter": True,
        },
    },
}


LIST_OCR_MODE = {
    "preprocess": {
        "mode": "original",
        "scale": 2,
        "rotate_to_portrait": True,
    },
    "rules": {
        "min_confidence": 0.30,
        "min_box_height_ratio": 0.02,
        "line_gap": 24,
        "main_top_ratio": 0.12,
        "main_bottom_ratio": 0.92,
        "max_unread_count": 99,
        "text_left_min_ratio": 0.16,
        "text_right_max_ratio": 0.78,
        "right_meta_min_ratio": 0.78,
        "avatar_max_x_ratio": 0.18,
        "content_gap_ratio": 0.022,
        "content_gap_min": 38,
        "content_gap_max": 62,
        "bottom_nav_top_ratio": 0.88,
        "badge_crop_padding": 6,
        "badge_ocr_scale": 6,
        "row_badge_ocr_scale": 4,
        "row_badge_min_confidence": 0.80,
        "row_strip_left_min_ratio": 0.84,
        "row_strip_extra_left_ratio": 0.06,
        "row_strip_width_ratio": 0.12,
        "row_strip_height_ratio": 0.62,
        "row_strip_max_height_ratio": 0.12,
        "row_strip_min_height_px": 24,
        "row_strip_bottom_nav_margin": 4,
        "system_keywords": LIST_SYSTEM_KEYWORDS,
        "excluded_keywords": LIST_EXCLUDED_KEYWORDS,
        "content_excluded_keywords": LIST_CONTENT_EXCLUDED_KEYWORDS,
    },
}


DEFAULT_OCR_MODES = {
    "dialogue": DIALOGUE_OCR_MODE,
    "list": LIST_OCR_MODE,
}
