"""人设对话编排层（docs/13 三/四节）。

对外三个入口：
- generate_reply：消息 → DeepSeek 生成（程序检查 + AI 审核打回环）→ 候选回复
- generate_greeting：pingtai 底稿 → （可选）AI 按人设改写 + 防重复负例 → 招呼语
- record_sent：App 实际发出后统一落库（普通/招呼/平台新招呼占位），并后台触发滚动摘要

人设与会话解耦：settings 来自根 config.json 的 device.app_overrides[<app>]（由接线方注入
read_settings 回调，测试传静态 dict），人设文本来自客户方案包 JSON > apps/<app>/ 随包默认。
"""
from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .deepseek_client import DeepSeekClient
from .persona_prompt import (
    append_rejection,
    beijing_now_text,
    build_chat_messages,
    build_greeting_messages,
    build_judge_messages,
    build_summary_messages,
)
from .persona_reply_check import (
    check_greeting,
    check_reply,
    dedup_pool_from_text,
    near_same,
    normalize_reply,
    strip_narration,
)
from .persona_session_store import (
    KIND_CHAT,
    KIND_GREETING,
    KIND_SYNTHETIC_GREETING,
    PersonaSessionStore,
)

DEFAULT_SETTINGS: dict[str, Any] = {
    # 客户域设置默认值（docs/13 五/六节）：实际值由接线方从 config.json 注入
    "model": "",
    "persona_profile_json": "",
    "greetings_txt": "",
    "judge_enabled": True,
    "greeting_ai_enabled": True,
    "history_turns": 6,
    "summary_interval": 10,
}

PROG_CAP = 2
JUDGE_CAP = 2
MAX_GEN_ATTEMPTS = max(3, 1 + PROG_CAP + JUDGE_CAP + 2)
JUDGE_TURNS = 2
EMPTY_REPLY_FALLBACK = "（这轮没接住话，换个话题吧）"
SYNTHETIC_GREETING_PLACEHOLDER = "（已通过平台功能向对方打了招呼）"

DEFAULT_GREETINGS_PATH = Path("runtime/greetings/pingtai.txt")


class PersonaChatError(RuntimeError):
    pass


@dataclass
class PersonaReply:
    text: str
    session_id: str
    contact_name: str = ""
    stats: dict[str, Any] = field(default_factory=dict)


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    s = str(text or "").strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```"):
            s = "\n".join(lines[1:-1]).strip()
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(s[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def _parse_verdict(text: str) -> dict[str, Any]:
    """审核结论解析；任何失败一律放行（fail-open，docs/13 4.6③）。"""
    obj = _extract_first_json_object(text)
    if obj is None:
        return {"ok": True, "reason": ""}
    return {"ok": bool(obj.get("ok")), "reason": str(obj.get("reason") or "")}


class PersonaChat:
    def __init__(
        self,
        *,
        client: DeepSeekClient,
        store: PersonaSessionStore,
        workspace_root: str | Path | None = None,
        read_settings: Callable[[], dict[str, Any]] | None = None,
        logger: Callable[[str, str, dict[str, Any] | None], None] | None = None,
    ) -> None:
        import os

        self.client = client
        self.store = store
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()
        configured = str(os.environ.get("JIXIEBI_PACKAGE_ROOT", "") or "").strip()
        self.package_root = Path(configured).resolve() if configured else self.workspace_root
        self._read_settings = read_settings or (lambda: {})
        self._logger = logger
        self._profile_cache: dict[str, tuple[int, dict[str, str]]] = {}
        self._summary_locks_guard = threading.Lock()
        self._summary_locks: dict[str, threading.Lock] = {}

    # ---------- 基础工具 ----------
    def _log(self, level: str, message: str, payload: dict[str, Any] | None = None) -> None:
        if self._logger:
            try:
                self._logger(level, message, payload)
            except Exception:
                pass

    def settings(self) -> dict[str, Any]:
        merged = dict(DEFAULT_SETTINGS)
        try:
            overrides = self._read_settings() or {}
        except Exception:
            overrides = {}
        if isinstance(overrides, dict):
            for key in DEFAULT_SETTINGS:
                if key in overrides and overrides[key] is not None:
                    merged[key] = overrides[key]
        return merged

    def _load_profile(self, path_text: str) -> dict[str, str]:
        path_text = str(path_text or "").strip()
        if not path_text:
            return {}
        path = Path(path_text)
        if not path.is_absolute():
            path = (self.workspace_root / path).resolve()
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            self._log("warning", "人设方案包 JSON 不可读，回退随包默认", {"path": str(path)})
            return {}
        cached = self._profile_cache.get(str(path))
        if cached and cached[0] == mtime:
            return cached[1]
        try:
            # utf-8-sig：容错客户用记事本编辑产生的 BOM
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            self._log("warning", "人设方案包 JSON 解析失败，回退随包默认", {"path": str(path), "error": str(exc)})
            return {}
        profile = {key: str(data.get(key) or "").strip() for key in ("基础设定", "人物信息", "边界条件", "审核标准")}
        self._profile_cache[str(path)] = (mtime, profile)
        return profile

    def _read_persona_default(self, app_name: str) -> dict[str, str]:
        """随包默认人设：apps/<app>/persona_default.json（四字段空白模板，客户未导入时兜底）。"""
        path = self.package_root / "apps" / str(app_name or "") / "persona_default.json"
        try:
            # utf-8-sig：容错 BOM；解析失败必须留痕，不能静默回退空人设
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            self._log("warning", "随包默认 persona_default.json 解析失败，人设将为空", {"path": str(path), "error": str(exc)})
            return {}
        if not isinstance(data, dict):
            return {}
        return {key: str(data.get(key) or "").strip() for key in ("基础设定", "人物信息", "边界条件", "审核标准")}

    def _effective_texts(self, app_name: str, settings: dict[str, Any]) -> dict[str, str]:
        """生效优先级：方案包 JSON > apps/<app>/persona_default.json（docs/13 六）。"""
        defaults = self._read_persona_default(app_name)
        profile = self._load_profile(settings.get("persona_profile_json", ""))
        return {
            "base_prompt": profile.get("基础设定") or defaults.get("基础设定", ""),
            "profile_text": profile.get("人物信息", "") or defaults.get("人物信息", ""),
            "boundary": profile.get("边界条件") or defaults.get("边界条件", ""),
            "judge_prompt": profile.get("审核标准") or defaults.get("审核标准", ""),
        }

    def _greetings_path(self, app_name: str, settings: dict[str, Any]) -> Path | None:
        configured = str(settings.get("greetings_txt", "") or "").strip()
        if configured:
            path = Path(configured)
            return path if path.is_absolute() else (self.workspace_root / path).resolve()
        default = self.workspace_root / DEFAULT_GREETINGS_PATH
        if default.is_file():
            return default
        bundled = self.package_root / "apps" / str(app_name or "") / "greetings_default.txt"
        return bundled if bundled.is_file() else None

    def _pick_greeting_draft(self, path: Path | None) -> str:
        if path is None:
            raise PersonaChatError("招呼语库不存在（未配置路径且无 pingtai.txt / 随包默认）")
        try:
            lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        except OSError as exc:
            raise PersonaChatError(f"招呼语库读取失败：{path}") from exc
        lines = [line for line in lines if line and not line.startswith("#")]
        if not lines:
            raise PersonaChatError(f"招呼语库为空：{path}")
        return random.choice(lines)

    def _resolve_session(
        self,
        device_ip: str,
        app_name: str,
        contact_name: str,
        continuity: dict[str, Any] | None,
    ) -> dict[str, Any]:
        session = self.store.find_or_create_session(device_ip, app_name, contact_name)
        turns = self.store.recent_turns(device_ip, app_name, session["session_id"], n=2)
        if turns or not isinstance(continuity, dict):
            return session
        # 新会话 + 屏幕气泡连续性信息：尝试找回旧会话并换绑昵称（docs/13 4.2）
        matched = self.store.find_continuity_session(
            device_ip,
            app_name,
            outgoing_texts=continuity.get("sender2"),
            incoming_texts=continuity.get("sender1"),
        )
        if not matched:
            return session
        matched_session = matched["session"]
        self._log(
            "info",
            "OCR 昵称漂移恢复：连续性命中，换绑旧会话",
            {"old_contact": matched_session.get("contact_name"), "new_contact": contact_name, "score": matched.get("score")},
        )
        return self.store.rebind_contact(device_ip, app_name, matched_session["session_id"], contact_name) or session

    # ---------- 主入口：消息回复 ----------
    def generate_reply(
        self,
        *,
        device_ip: str,
        app_name: str,
        contact_name: str,
        message: str,
        continuity: dict[str, Any] | None = None,
    ) -> PersonaReply:
        settings = self.settings()
        texts = self._effective_texts(app_name, settings)
        if not texts["base_prompt"] and not texts["profile_text"]:
            raise PersonaChatError("人设为空：未配置方案包 JSON，且随包默认 persona_default.json 为空")

        session = self._resolve_session(device_ip, app_name, contact_name, continuity)
        session_id = str(session["session_id"])
        history_turns = self.store.recent_turns(device_ip, app_name, session_id, n=2 * max(0, int(settings["history_turns"])))
        summary_row = self.store.get_summary(device_ip, app_name, session_id)
        summary = str(summary_row["summary"]) if summary_row else ""
        now_text = beijing_now_text()
        last_assistant = next((t["content"] for t in reversed(history_turns) if t["role"] == "assistant"), "")
        dedup_pool = dedup_pool_from_text(last_assistant)

        messages = build_chat_messages(
            base_prompt=texts["base_prompt"],
            profile_text=texts["profile_text"],
            summary=summary,
            history_turns=history_turns,
            query=str(message or ""),
            boundary=texts["boundary"],
            now_text=now_text,
        )
        model = str(settings.get("model") or "")

        prog_retries = 0
        judge_retries = 0
        attempts = 0
        last_rejected = ""
        text = ""
        cache_hit_total = 0
        started = time.time()
        while attempts < MAX_GEN_ATTEMPTS:
            attempts += 1
            try:
                reply = self.client.chat(messages, **({"model": model} if model else {}), max_tokens=512)
            except Exception as exc:
                raise PersonaChatError(f"DeepSeek 生成失败：{exc}") from exc
            text = reply.text
            cache_hit_total += int(reply.usage.get("prompt_cache_hit_tokens") or 0)

            reasons = check_reply(text, dedup_pool=dedup_pool, user_messages=[message])
            if reasons:
                if prog_retries < PROG_CAP and not near_same(text, last_rejected):
                    prog_retries += 1
                    last_rejected = text
                    self._log("info", "程序检查打回", {"attempt": attempts, "reasons": reasons[:3]})
                    append_rejection(messages, draft_text=text, reason="；".join(reasons), label="系统打回")
                    continue
                self._log("warning", "程序检查预算耗尽，带病输出", {"reasons": reasons[:2]})
                break

            if settings.get("judge_enabled", True):
                judge_messages = build_judge_messages(
                    judge_prompt=texts["judge_prompt"],
                    history_turns=history_turns,
                    query=message,
                    reply_text=text,
                    now_text=now_text,
                    judge_turns=JUDGE_TURNS,
                )
                try:
                    judge_reply = self.client.chat(
                        judge_messages,
                        **({"model": model} if model else {}),
                        temperature=0.3,
                        max_tokens=200,
                    )
                    verdict = _parse_verdict(judge_reply.text)
                except Exception as exc:
                    self._log("warning", "AI 审核调用失败，本轮放行", {"error": str(exc)})
                    verdict = {"ok": True, "reason": ""}
                if not verdict.get("ok"):
                    reason = str(verdict.get("reason") or "").strip() or "回复质量不达标"
                    if judge_retries < JUDGE_CAP and not near_same(text, last_rejected):
                        judge_retries += 1
                        last_rejected = text
                        self._log("info", "AI 审核打回", {"attempt": attempts, "reason": reason[:120]})
                        append_rejection(messages, draft_text=text, reason=reason, label="审核打回")
                        continue
                    self._log("warning", "审核预算耗尽，带病输出", {"reason": reason[:120]})
                    break
            break

        final_text = normalize_reply(strip_narration(text or ""))
        if not final_text:
            final_text = EMPTY_REPLY_FALLBACK
        stats = {
            "prog_retries": prog_retries,
            "judge_retries": judge_retries,
            "attempts": attempts,
            "cache_hit_tokens": cache_hit_total,
            "elapsed_s": round(time.time() - started, 2),
            "session_id": session_id,
        }
        return PersonaReply(text=final_text, session_id=session_id, contact_name=str(session.get("contact_name") or ""), stats=stats)

    # ---------- 落库与摘要 ----------
    def record_sent(
        self,
        *,
        device_ip: str,
        app_name: str,
        contact_name: str,
        user_text: str = "",
        assistant_text: str = "",
        kind: str = KIND_CHAT,
    ) -> None:
        """App 实际发出后的统一落库入口（docs/13 4.3 第 6 步）。"""
        if not str(user_text or "").strip() and not str(assistant_text or "").strip():
            return
        session = self.store.find_or_create_session(device_ip, app_name, contact_name)
        session_id = str(session["session_id"])
        self.store.record_exchange(
            device_ip,
            app_name,
            session_id,
            user_text=user_text,
            assistant_text=assistant_text,
            kind=kind,
        )
        self._maybe_summarize(device_ip=device_ip, app_name=app_name, session_id=session_id)

    def record_synthetic_greeting(self, *, device_ip: str, app_name: str, contact_name: str) -> None:
        """平台"新招呼"按钮代发后的占位落库（docs/13 4.4）。"""
        self.record_sent(
            device_ip=device_ip,
            app_name=app_name,
            contact_name=contact_name,
            assistant_text=SYNTHETIC_GREETING_PLACEHOLDER,
            kind=KIND_SYNTHETIC_GREETING,
        )

    def _session_summary_lock(self, session_id: str) -> threading.Lock:
        with self._summary_locks_guard:
            lock = self._summary_locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._summary_locks[session_id] = lock
            return lock

    def _maybe_summarize(self, *, device_ip: str, app_name: str, session_id: str) -> None:
        thread = threading.Thread(
            target=self._summarize_job,
            args=(device_ip, app_name, session_id),
            daemon=True,
            name=f"persona-summary-{session_id[:8]}",
        )
        thread.start()

    def _summarize_job(self, device_ip: str, app_name: str, session_id: str) -> None:
        lock = self._session_summary_lock(session_id)
        if not lock.acquire(blocking=False):
            return  # 上一轮摘要还在跑，下个触发点自然会重试
        try:
            settings = self.settings()
            aged, aged_max, _watermark = self.store.aged_turns_for_summary(
                device_ip,
                app_name,
                session_id,
                window_n=max(0, int(settings["history_turns"])),
                min_aged=max(1, int(settings["summary_interval"])),
            )
            if not aged:
                return
            summary_row = self.store.get_summary(device_ip, app_name, session_id)
            previous = str(summary_row["summary"]) if summary_row else ""
            messages = build_summary_messages(previous_summary=previous, aged_turns=aged)
            model = str(settings.get("model") or "")
            try:
                reply = self.client.chat(
                    messages,
                    **({"model": model} if model else {}),
                    temperature=0.3,
                    max_tokens=800,
                )
            except Exception as exc:
                self._log("warning", "记忆提炼调用失败，下个触发点重试", {"session_id": session_id, "error": str(exc)})
                return
            summary_text = str(reply.text or "").strip()
            if not summary_text or len(summary_text) > 4000:
                self._log("warning", "记忆提炼产物异常，放弃写入", {"session_id": session_id, "length": len(summary_text)})
                return
            self.store.save_summary(device_ip, app_name, session_id, summary_text, aged_max)
            self._log("info", "前情摘要已更新", {"session_id": session_id, "up_to_seq": aged_max, "chars": len(summary_text)})
        finally:
            lock.release()

    # ---------- 招呼 ----------
    def generate_greeting(self, *, device_ip: str, app_name: str, contact_name: str) -> str:
        """招呼语：底稿 →（可选）AI 按人设改写（防重复负例 + 程序检查 + 回退）。"""
        settings = self.settings()
        draft = self._pick_greeting_draft(self._greetings_path(app_name, settings))
        if not settings.get("greeting_ai_enabled", True):
            return draft
        texts = self._effective_texts(app_name, settings)
        if not texts["base_prompt"] and not texts["profile_text"]:
            return draft  # 人设为空时无从改写，直接用底稿
        negatives = self.store.recent_greetings(device_ip, app_name, n=5)
        model = str(settings.get("model") or "")
        for _attempt in range(2):  # 检查不过重试一次（docs/13 4.4 第 3 步）
            try:
                reply = self.client.chat(
                    build_greeting_messages(
                        base_prompt=texts["base_prompt"],
                        profile_text=texts["profile_text"],
                        draft=draft,
                        negative_examples=negatives,
                    ),
                    **({"model": model} if model else {}),
                    max_tokens=64,
                    timeout=15.0,
                )
            except Exception as exc:
                self._log("warning", "招呼语 AI 改写失败，回退底稿", {"error": str(exc)})
                return draft
            candidate = normalize_reply(strip_narration(reply.text or ""))
            if not check_greeting(candidate):
                return candidate
        self._log("info", "招呼语改写两次未过检查，回退底稿", {})
        return draft
