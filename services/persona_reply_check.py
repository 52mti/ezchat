"""生成后程序检查（docs/13 4.6②，规则族移植自 skill_agent procheck 并适配）。

对候选回复做代码层硬检查，返回命中原因列表（空 = 通过）：
- 格式：单行（多句用 | 分隔）、分句 ≤5、中文 4~60 字、无 emoji、纯中文（禁字母）
- 黑名单：平台名/平台缩写/"平台"二字、做不到的承诺（纯文字陪伴）
- 风险表达：改天/下次类、离场/睡眠类（用户主动收尾时放宽）、位置线索、未来联系承诺
- 防复读：与最近回复按 | 子句 bigram ≥0.9 判重
"""
from __future__ import annotations

import re
from typing import Iterable

# ---------------- 相似度 ----------------
def _bigrams(s: str) -> set[str]:
    s = re.sub(r"\s+", "", s)
    return {s[i : i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else set()


def bigram_overlap(a: str, b: str) -> float:
    ba, bb = _bigrams(a), _bigrams(b)
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / max(len(ba), len(bb))


def near_same(a: str, b: str, threshold: float = 0.88) -> bool:
    """打回卡死检测：两次被拒文本归一化后近似相同（docs/13 4.6）。"""
    aa = normalize_reply(strip_narration(a or ""))
    bb = normalize_reply(strip_narration(b or ""))
    if not aa or not bb:
        return False
    if aa == bb:
        return True
    return bigram_overlap(aa, bb) >= threshold


# ---------------- 文本清洗 ----------------
def strip_narration(text: str) -> str:
    """去操作旁白：括号心理戏 / 正在读取… / 注：…"""
    if not text:
        return text
    s = text
    s = re.sub(r"（[^）]*）", "", s)
    s = re.sub(r"\([^\)]*\)", "", s)
    s = re.sub(r"[^\n]*正在(查看|读取|写入|执行|标记|调用|加载|检查)[^\n]*", "", s)
    s = re.sub(r"[^\n]*(查看|调用|读取)(知识库|对应文件|\S*\.md)[^\n]*", "", s)
    s = re.sub(r"注[：:].+", "", s)
    return s.strip()


def normalize_reply(text: str) -> str:
    """轻量规整输出：去 emoji、平台名替换、换行折叠为 |。"""
    if not text:
        return text
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    s = EMOJI_RE.sub("", s)
    for name in sorted(PLATFORM_NAMES, key=len, reverse=True):
        s = s.replace(name, "那个")
    s = re.sub(r"加微|微我|加抖", "那个", s)
    s = re.sub(r"\n+", " | ", s)
    s = re.sub(r"\s*\|\s*", "|", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip(" \n\t|")


def user_initiated_close(user_texts: Iterable[str]) -> bool:
    """用户是否主动收尾（说晚安/先睡/不吵你等）。命中时离场/睡眠类检查放宽。"""
    text = " ".join(user_texts or [])
    return bool(re.search(r"困了|先睡|睡了|晚安|早点休息|不吵你|你先忙|你忙吧|回去休息|别太晚|我先撤|我先走", text))


# ---------------- 黑名单 ----------------
PLATFORM_NAMES = (
    "微信", "微信号", "加微",
    "抖音", "小红书", "微博", "快手", "探探", "陌陌", "soul", "ins",
    "支付宝", "淘宝", "天猫", "京东", "拼多多", "闲鱼", "美团",
    "知乎", "豆瓣", "贴吧", "视频号", "哔哩哔哩", "b站", "B站",
    "QQ", "qq", "Q群",
    "世纪佳缘", "百合网", "珍爱网",
)

# 承诺了"做不到的事"——纯文字线上陪伴
UNDOABLE_ACTIONS = (
    "打游戏", "开黑", "上分", "双排", "组队", "带你飞",
    "给你唱", "唱给你", "唱歌给",
    "给你做饭", "做给你吃", "给你煮", "煮给你",
    "帮你写作业", "帮你写代码", "替你写作业", "辅导你写", "教你写",
)

EMOJI_RE = re.compile(
    "[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0000FE0F\U0000200D]"
)
# 平台缩写：微/抖 当名词用（后接特定字）才判，避开 微笑/微风/发抖/抖动 等正常词
ABBR_RE = re.compile(r"微(?=[只不也号上是了的加我没给])|加微|微我|抖(?=[上号是也的加没])|加抖")
FUTURE_RE = re.compile(r"改天|下次|回头|明天再说|明天吧|明天有空再|有空再来|顺路过去|顺路来|过去看看你|下回来|自己来听|想见我就直说|到时候再说|以后再说|有机会再")
LEAVING_RE = re.compile(r"我先回了|我真得走了|我也该走了|先回家|回家了|正打算锁门|准备锁门|差不多要关门|打烊|收摊|下班了|早点回去歇着|你赶紧回家歇着|我去擦擦柜台|先回了啊|我先去忙|我先去|先忙啦|不跟你聊了|不理你了|挂了吧|挂电话|去吧|睡吧|好梦|晚安|吃完再聊|等你回来|早点歇着|忙你的去吧|别急着散场|别把话聊死")
SLEEPY_RE = re.compile(r"躺着了|躺床上|快睡了|先睡了|去睡了|歇着了|困了|装睡|闭眼|枕头")
LOCATION_HINT_RE = re.compile(r"(旁边|附近|对面|隔壁).{0,8}(店|馆|门|楼|路)|几步路|门口就是|楼下就是")
FUTURE_CONTACT_RE = re.compile(r"到了我.*说|到家.*说一声|忙完.*说一声|回头.*联系|晚点.*找你|等会.*聊")


def dedup_pool_from_text(assistant_text: str) -> list[str]:
    """最近 assistant 回复按 | 拆子句池（防复读）。"""
    pool: list[str] = []
    for seg in str(assistant_text or "").split("|"):
        seg = seg.strip()
        if seg:
            pool.append(seg)
    return pool


def _format_reasons(raw_text: str) -> list[str]:
    reasons: list[str] = []
    if re.search(r"[\r\n]", raw_text):
        reasons.append("回复里出现了换行，必须单行输出，句子之间只用 | 分隔，禁止换行。")
    hit = next((p for p in PLATFORM_NAMES if p in raw_text), None)
    if hit:
        reasons.append(f'回复里出现了平台名"{hit}"，角色输出绝对不能出现任何平台名（拒绝时也别说出名字，用"那个""那儿"代替）。')
    abbr = ABBR_RE.search(raw_text)
    if abbr:
        reasons.append(f'回复里出现了平台缩写"{abbr.group(0)}"（微=微信/抖=抖音），不得用缩写提平台。')
    if "平台" in raw_text:
        reasons.append('回复里出现了"平台"二字，角色是真实人物，绝对不能流露出自己在某个App/平台上的感觉（用"这儿""这里"代替）。')
    if EMOJI_RE.search(raw_text):
        reasons.append("回复里出现了 emoji/符号，角色台词只用纯文字，禁止 emoji。")
    if re.search(r"[A-Za-z]", raw_text):
        reasons.append("回复里出现了英文字母，本角色台词必须为纯中文，不允许任何英文单词或字母（含品牌名、缩写、语气词）。")
    return reasons


def _content_reasons(cleaned: str, user_close: bool) -> list[str]:
    reasons: list[str] = []
    act = next((p for p in UNDOABLE_ACTIONS if p in cleaned), None)
    if act:
        reasons.append(f'回复里承诺了做不到的事（"{act}"），她是纯文字线上陪伴，不能打游戏/唱歌/做饭/替写作业等。')
    if FUTURE_RE.search(cleaned):
        reasons.append("回复里出现了改天/下次/顺路过去这类未来可见表达，容易被审核判成暗示见面。")
    if LEAVING_RE.search(cleaned) and not user_close:
        reasons.append("回复里出现了关门/锁门/回家/走了这类离场表达，角色不该主动结束对话。")
    if SLEEPY_RE.search(cleaned) and not user_close:
        reasons.append("回复里出现了躺着了/快睡了这类睡眠或下线状态，角色应保持一直在线。")
    if LOCATION_HINT_RE.search(cleaned):
        reasons.append("回复里出现了周边环境或可跟到人的位置线索，容易被审核判成可定位信息。")
    if FUTURE_CONTACT_RE.search(cleaned):
        reasons.append("回复里出现了到家再说/忙完再说/回头联系这类未来互动承诺，容易被审核判成越界。")
    return reasons


def _length_reasons(cleaned: str, max_segments: int, min_chars: int, max_chars: int) -> list[str]:
    reasons: list[str] = []
    segs = [s for s in re.split(r"\|", cleaned) if s.strip()]
    if not segs:
        reasons.append("回复为空，请输出角色台词。")
        return reasons
    if len(segs) > max_segments:
        reasons.append(f"分句太多（{len(segs)} 句），最多 {max_segments} 句，请合并精简。")
    total_cn = len(re.findall(r"[一-鿿]", cleaned))
    if total_cn < min_chars:
        reasons.append(f"总字数太少（{total_cn} 字），整条至少 {min_chars} 个中文字。")
    if total_cn > max_chars:
        reasons.append(f"总字数太多（{total_cn} 字），整条最多 {max_chars} 个中文字。")
    return reasons


def _dedup_reasons(cleaned: str, dedup_pool: list[str] | None) -> list[str]:
    reasons: list[str] = []
    segs = [s for s in re.split(r"\|", cleaned) if s.strip()]
    for prev in dedup_pool or []:
        hit = False
        for seg in segs:
            if bigram_overlap(seg, prev) >= 0.9:
                reasons.append(f"和上一条回复高度重复（'{prev[:14]}'），请换一种完全不同的说法。")
                hit = True
                break
        if hit:
            break
    return reasons


def check_reply(
    text: str,
    *,
    dedup_pool: list[str] | None = None,
    user_messages: list[str] | None = None,
    max_segments: int = 5,
    min_chars: int = 4,
    max_chars: int = 60,
) -> list[str]:
    """完整程序检查：返回全部命中原因，空列表 = 通过。

    格式/黑名单/风险检查在剥旁白后的原始文本上做（normalize 会把平台名替换成
    "那个"，检查若在归一化文本上做将永远命中不了）；长度/分句/判重用归一化文本。
    """
    raw = strip_narration(text or "")
    if not raw.strip():
        return ["回复为空或全是操作旁白，请直接输出角色台词本身。"]
    cleaned = normalize_reply(raw)
    user_close = user_initiated_close(user_messages or [])
    reasons = _format_reasons(raw)
    reasons += _content_reasons(raw, user_close)
    reasons += _length_reasons(cleaned, max_segments, min_chars, max_chars)
    reasons += _dedup_reasons(cleaned, dedup_pool)
    return reasons


def check_greeting(text: str, *, max_chars: int = 40) -> list[str]:
    """招呼改写产物的格式/黑名单子集（docs/13 4.4）。"""
    raw = str(text or "").strip()
    if not raw:
        return ["招呼语为空。"]
    reasons = _format_reasons(raw)
    if len(re.findall(r"[一-鿿]", raw)) > max_chars:
        reasons.append(f"招呼语太长（超过 {max_chars} 个中文字）。")
    return reasons
