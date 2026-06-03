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
    "收到搭讪礼物",
    "收到搭汕礼物",
    "收到搭训礼物",
    "汉堡包",
    "回赠礼物",
    "+0.19金币",
    "丘比特发现",
    "缘分哦",
    "神秘礼",
    "小技巧",
    "您没有及时回复",
    "被退回",
    "主动给对方",
    "金币",
    "多回复几句",
    "回复率更高",
    "爱的",
    "电话粥",
    "红包",
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
    "礼物",
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
    "收到搭讪礼物",
    "收到搭汕礼物",
    "收到搭训礼物",
    "丘比特发现",
    "回消息还可收到",
    "金币礼物",
    "回复率更高",
]

DIALOGUE_LINE_AND_BELOW_KEYWORDS = [
    "输入你想说的话",
    "推荐灵感",
    "真实性及体验",
]

# 列表页规则会在 run_ocr_mode(..., "list") 之后进入 build_list_result()。
LIST_SYSTEM_KEYWORDS = [
    "消息",
    "关系",
    "优质用户",
    "回归奖励",
    "登录就送现金",
    "谁偷偷看过你",
    "快去看看",
    "聊天",
    "群聊",
    "群聊派对",
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
    "家族狂欢",
    "人在玩",
    "人在这里嗨",
    "加入家族",
    "去聊天",
    "免费匹配机会",
    "点击领取",
]

LIST_EXCLUDED_KEYWORDS = [
    "Ta",
    "在线匹配",
    "在线配对",
    "优质用户",
    "回归奖励",
    "登录就送现金",
    "谁偷偷看过你",
    "群聊派对",
    "家族狂欢",
    "家族群聊",
    "聊天广场",
    "服务号合集",
    "官方群",
    "官方",
    "点击领取",
    "免费匹配机会",
    "加入家族",
    "去聊天",
    "小助手",
    "温馨提示",
]

LIST_CONTENT_EXCLUDED_KEYWORDS = [
    "推荐灵感",
    "快去看看",
    "人在这里嗨",
    "人在玩",
    "登录就送现金",
]

LIST_UNREAD_COUNT_EXCLUDED_KEYWORDS = [
    "温馨提示",
    "聊天卡",
]

LIST_BOTTOM_NAV_KEYWORDS = ["交友", "广场", "娱乐", "消息", "我的"]

LIST_TOP_NAV_KEYWORDS = ["消息", "关系"]

LIST_NAME_SUFFIX_KEYWORDS = ["附近"]

POINTS_KEYWORDS = ["我的收益", "收益", "金币"]


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
        "name_gap_ratio": 0.037,
        "sender_detection": {
            "method": "bubble_color",
            "sender2_min_x_ratio": 0.52,
            "enable_color_fallback": False,
            "color_saturation_threshold": 0.30,
            "bubble_color_min_saturation": 0.12,
            "bubble_color_min_value": 0.20,
            "bubble_color_sender2_hue_min": 110,
            "bubble_color_sender2_hue_max": 150,
            "bubble_color_sender2_blue_green_min": 65,
            "bubble_color_sender2_blue_red_ratio_min": 1.60,
            "bubble_color_sender2_blue_min_saturation": 0.30,
        },
        "target_sender": "sender1",
        "user_name_top_ratio": 0.09,
        "user_name_center_x_min_ratio": 0.3,
        "user_name_center_x_max_ratio": 0.7,
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
        "main_top_ratio": 0.10,
        "main_bottom_ratio": 0.88,
        "max_unread_count": 99,
        "badge_position": "left",
        "badge_default_unread_count": 1,
        "enable_red_badge_detection": True,
        "left_red_badge_force_default_unread_count": True,
        "text_left_min_ratio": 0.16,
        "text_right_max_ratio": 0.78,
        "right_meta_min_ratio": 0.78,
        "avatar_max_x_ratio": 0.18,
        "content_gap_ratio": 0.022,
        "content_gap_min": 38,
        "content_gap_max": 62,
        "bottom_nav_top_ratio": 0.88,
        "bottom_nav_accept_any_text": False,
        "bottom_nav_keywords": LIST_BOTTOM_NAV_KEYWORDS,
        "top_nav_keywords": [],
        "top_nav_top_ratio": 0.0,
        "top_nav_bottom_ratio": 0.0,
        "left_badge_min_ratio": 0.02,
        "left_badge_max_ratio": 0.24,
        "left_badge_extra_top_ratio": 0.03,
        "red_badge_saturation_min": 60,
        "red_badge_value_min": 60,
        "red_badge_min_area": 120,
        "red_badge_min_width": 16,
        "red_badge_min_height": 16,
        "left_red_badge_search_height_ratio": 0.07,
        "left_red_badge_search_min_height_px": 52,
        "content_like_name_min_length": 10,
        "content_like_name_min_width_ratio": 0.24,
        "badge_crop_padding": 6,
        "badge_ocr_scale": 6,
        "left_badge_ocr_scale": 8,
        "left_badge_ocr_min_confidence": 0.50,
        "left_badge_ocr_left_ratio": 0.14,
        "left_badge_ocr_width_ratio": 0.08,
        "left_badge_ocr_top_offset_ratio": 0.012,
        "left_badge_ocr_height_ratio": 0.045,
        "left_badge_retry_left_ratio": 0.1,
        "left_badge_retry_width_ratio": 0.12,
        "left_badge_retry_top_offset_ratio": 0.02,
        "left_badge_retry_height_ratio": 0.06,
        "left_badge_retry_scale": 12,
        "left_badge_retry_min_confidence": 0.5,
        "row_badge_ocr_scale": 4,
        "row_badge_min_confidence": 0.80,
        "row_strip_left_min_ratio": 0.84,
        "row_strip_extra_left_ratio": 0.06,
        "row_strip_width_ratio": 0.12,
        "row_strip_height_ratio": 0.62,
        "row_strip_max_height_ratio": 0.12,
        "row_strip_min_height_px": 24,
        "row_strip_bottom_nav_margin": 4,
        "name_suffix_keywords": LIST_NAME_SUFFIX_KEYWORDS,
        "system_keywords": LIST_SYSTEM_KEYWORDS,
        "excluded_keywords": LIST_EXCLUDED_KEYWORDS,
        "content_excluded_keywords": LIST_CONTENT_EXCLUDED_KEYWORDS,
        "unread_count_excluded_keywords": LIST_UNREAD_COUNT_EXCLUDED_KEYWORDS,
    },
}


POINTS_OCR_MODE = {
    "preprocess": {
        "mode": "original",
        "scale": 2,
        "rotate_to_portrait": True,
    },
    "rules": {
        "min_confidence": 0.30,
        "min_box_height_ratio": 0.01,
        "line_gap": 24,
        "keywords": POINTS_KEYWORDS,
        "neighbor_y_ratio": 0.08,
        "neighbor_x_ratio": 0.65,
    },
}


OCR_MODES = {
    "dialogue": DIALOGUE_OCR_MODE,
    "list": LIST_OCR_MODE,
    "points": POINTS_OCR_MODE,
}
