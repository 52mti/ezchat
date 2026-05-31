from __future__ import annotations


# 这个文件只负责“微信”这个 App 的 OCR 规则。
# 运行时入口在 vision/ocr_rules.py 的 run_ocr_mode()：
# - dialogue：进入聊天详情页后，识别对话气泡，输出 user_name/dialogues。
# - list：停留在消息列表页时，识别联系人、未读数、内容摘要和底部导航。
# 以后如果新增 App，不要共用这里的规则，应该在新 App 自己的 ocr_config.py 里单独写。


# 下面这些关键词会在 vision/ocr_rules.py 里生效：
# - `ocr_result_excluded_keywords`：对话页单个 OCR 文字块命中就直接丢掉。
# - `line_excluded_keywords`：对话页合并成一句后，命中就整条丢掉。
# - `line_and_below_keywords`：对话页命中后，当前行以及下面的内容都不再算聊天内容。
# - `system_keywords`：列表页顶部区域命中时，按系统/入口行排除。
# - `excluded_keywords`：列表页联系人名字命中时，整行丢掉。
# - `content_excluded_keywords`：列表页内容摘要命中时，整行丢掉。
# 你后面只要改这里，run_ocr_mode() 读到的结果就会跟着变。
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

LIST_NAME_SUFFIX_KEYWORDS = []


# 对话模式配置，结构参考 zhaopiaochuli/dialogue_config.ini。
# 这里的 rules 会在 run_ocr_mode(..., "dialogue") 时进入 build_dialogue_result()。
DIALOGUE_OCR_MODE = {
    "preprocess": {
        # 机械臂截图会先统一转成竖屏；OCR 和返回坐标都按竖屏归一化坐标理解。
        "mode": "original",
        "scale": 2,
        "rotate_to_portrait": True,
    },
    "rules": {
        "min_confidence": 0.30,
        "min_box_height_ratio": 0.02,
        "line_gap": 24,
        "name_gap": 70,
        # sender1/sender2 目前按文字块位置判断：靠左是对方，靠右是自己。
        "sender_detection": {
            "method": "position",
            "sender2_min_x_ratio": 0.52,
            "enable_color_fallback": False,
            "color_saturation_threshold": 0.65,
        },
        # OCR 原始结果里只要包含这些词，就直接丢掉。
        "ocr_result_excluded_keywords": DIALOGUE_OCR_RESULT_EXCLUDED_KEYWORDS,
        # 合并成一句话后，只要包含这些词，这整句话就丢掉。
        "line_excluded_keywords": DIALOGUE_LINE_EXCLUDED_KEYWORDS,
        # 识别到这些词以后，当前行和它下面的内容都不再当作聊天内容。
        "line_and_below_keywords": DIALOGUE_LINE_AND_BELOW_KEYWORDS,
        "basic_filter": {
            # 常见基础过滤开关：时间、调试文字、奖励提示、输入框区域等。
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
            # 参考 dialogue_config.ini：右侧很窄、又不是中文/表情的文字，多半是噪声。
            "enable_right_narrow_non_cjk_filter": True,
            "right_narrow_min_x_ratio": 0.88,
            "right_narrow_max_width_ratio": 0.12,
            "enable_non_cjk_non_emoji_filter": True,
        },
    },
}


# 列表模式配置，结构参考 zhaopiaochuli/list_config.ini。
# 这里的 rules 会在 run_ocr_mode(..., "list") 时进入 build_list_result()。
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
        "badge_position": "right",
        "badge_default_unread_count": 1,
        # 微信列表：头像在左，名字/摘要在中间，未读数通常在右侧。
        "text_left_min_ratio": 0.16,
        "text_right_max_ratio": 0.78,
        "right_meta_min_ratio": 0.78,
        "avatar_max_x_ratio": 0.18,
        "content_gap_ratio": 0.022,
        "content_gap_min": 38,
        "content_gap_max": 62,
        "bottom_nav_top_ratio": 0.88,
        "bottom_nav_accept_any_text": False,
        # 未读角标识别失败时，会按行右侧小区域再单独 OCR 一次。
        "badge_crop_padding": 6,
        "right_badge_min_ratio": 0.82,
        "red_badge_saturation_min": 80,
        "red_badge_value_min": 80,
        "content_like_name_min_length": 10,
        "content_like_name_min_width_ratio": 0.24,
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
        # 底部导航只用于识别页面和防熄屏跳转，不会进入 message_list。
        "bottom_nav_keywords": ["消息", "通讯录", "发现", "视频号", "我"],
        "top_nav_keywords": [],
        "top_nav_top_ratio": 0.0,
        "top_nav_bottom_ratio": 0.0,
        "name_suffix_keywords": LIST_NAME_SUFFIX_KEYWORDS,
        # system_keywords：用于排除搜索框、底部导航、系统入口等非联系人行。
        "system_keywords": LIST_SYSTEM_KEYWORDS,
        # excluded_keywords：名字命中这些词时，这一整行不会作为联系人。
        "excluded_keywords": LIST_EXCLUDED_KEYWORDS,
        # content_excluded_keywords：摘要内容命中这些词时，这一行不会作为可回复消息。
        "content_excluded_keywords": LIST_CONTENT_EXCLUDED_KEYWORDS,
    },
}


OCR_MODES = {
    "dialogue": DIALOGUE_OCR_MODE,
    "list": LIST_OCR_MODE,
}
