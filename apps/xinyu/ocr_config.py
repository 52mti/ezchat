from __future__ import annotations


# 心遇的 OCR 规则单独配置，不和微信共用。
# 结构分成两个模式：
# - dialogue：对话详情页
# - list：列表页
# 这样以后你改某个 App 的规则时，只需要改这个文件。


# 下面这些关键词会在 vision/ocr_rules.py 里生效：
# - `ocr_result_excluded_keywords`：对话页单个 OCR 文字块命中就直接丢掉。
# - `line_excluded_keywords`：对话页合并成一句后，命中就整条丢掉。
# - `line_and_below_keywords`：对话页命中后，当前行以及下面的内容都不再算聊天内容。
# - `system_keywords`：列表页顶部区域命中时，按系统/入口行排除。
# - `excluded_keywords`：列表页联系人名字命中时，整行丢掉。
# - `content_excluded_keywords`：列表页内容摘要命中时，整行丢掉。
# 你后面可以直接在这里改，run_ocr_mode() 会读取到新的配置。
# 下面先按 zhaopiaochuli/dialogue_config.ini 填入，后续可按心遇界面单独删改。
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
    "仙女棒",
    "常用语",
]

DIALOGUE_LINE_AND_BELOW_KEYWORDS = [
    "输入你想说的话",
    "推荐灵感",
    "真实性及体验",
]

# 下面先按 zhaopiaochuli/list_config.ini 填入，并保留心遇顶部筛选项。
# 这里的规则会在 run_ocr_mode(..., "list") 后进入 build_list_result()。
LIST_SYSTEM_KEYWORDS = ["全部消息", "全部信息", "未回复", "在线", "距离"]

LIST_EXCLUDED_KEYWORDS = []

LIST_CONTENT_EXCLUDED_KEYWORDS = []


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
        # 心遇列表左侧常出现“匹配度”，右侧是未读红点或数值。
        "max_unread_count": 99,
        "badge_position": "left",
        "badge_default_unread_count": 1,
        "text_left_min_ratio": 0.16,
        "text_right_max_ratio": 0.78,
        "right_meta_min_ratio": 0.78,
        "avatar_max_x_ratio": 0.18,
        "content_gap_ratio": 0.022,
        "content_gap_min": 38,
        "content_gap_max": 62,
        "bottom_nav_top_ratio": 0.80,
        "bottom_nav_accept_any_text": True,
        "badge_crop_padding": 6,
        "left_badge_min_ratio": 0.02,
        "left_badge_max_ratio": 0.24,
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
        # 列表页底部导航和顶部“全部消息 / 全部信息”等入口。
        "bottom_nav_keywords": [
            "喜欢",
            "心动",
            "家族广场",
            "消息",
            "我的",
            "全部消息",
            "全部信息",
            "未回复",
            "在线",
            "距离",
        ],
        "top_nav_keywords": ["全部消息", "全部信息", "未回复", "在线", "距离"],
        "top_nav_top_ratio": 0.10,
        "top_nav_bottom_ratio": 0.20,
        "system_keywords": LIST_SYSTEM_KEYWORDS,
        "excluded_keywords": LIST_EXCLUDED_KEYWORDS,
        "content_excluded_keywords": LIST_CONTENT_EXCLUDED_KEYWORDS,
    },
}


DEFAULT_OCR_MODES = {
    "dialogue": DIALOGUE_OCR_MODE,
    "list": LIST_OCR_MODE,
}
