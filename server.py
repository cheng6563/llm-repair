#!/usr/bin/env python3
"""LLM API 透明反代 + 请求响应日志记录工具

用法:
    python llm_proxy.py [--port PORT]

寻址：上游目标由请求路径首段给出（无回退，缺失则返回 400）。
  - 编码完整 URL：
      ANTHROPIC_BASE_URL=http://127.0.0.1:38787/https%3A%2F%2Fai.usmercari.com%2F
      → /https%3A%2F%2Fai.usmercari.com%2F/v1/messages 反代到 https://ai.usmercari.com/v1/messages
  - 只填域名（自动补 https）：
      ANTHROPIC_BASE_URL=http://127.0.0.1:38787/ai.usmercari.com
      → /ai.usmercari.com/v1/messages 反代到 https://ai.usmercari.com/v1/messages
  - 带额外路径前缀（编码进 URL，或明文跟在域名后均可）：
      ANTHROPIC_BASE_URL=http://127.0.0.1:38787/ai.usmercari.com/api/v3
      → /ai.usmercari.com/api/v3/v1/messages 反代到 https://ai.usmercari.com/api/v3/v1/messages

日志目录: 当前脚本目录下的 logs/（可用 --log-dir 覆盖）
  YYYY-MM-DD/HH/<req_id>.json
      每个请求一个独立文件，按 日期/小时 分子目录。内容是单个完整的
      请求+响应对象（缩进美化，可直接打开阅读）。
      浏览示例：
        cat 2026-06-06/14/20260606_142315_123456.json | jq .
        grep -rl '"host":"ai.usmercari.com"' 2026-06-06/
  YYYY-MM-DD.tar.gz
      过去日期的目录整体打包归档；今天的目录保持明文散文件。
      浏览示例：
        tar -xzf 2026-06-05.tar.gz -O 2026-06-05/14/xxx.json | jq .
  proxy.log
      滚动控制台日志（RotatingFileHandler，单文件 10MB × 3）
  --max-log-size 控制目录总占用（GB），默认 1，超出按 mtime 升序清理最旧归档

代理设置: 设置环境变量 HTTPS_PROXY=http://127.0.0.1:30808 即可走本地代理

==============================================================================
usage 归一化（_SSEUsageNormalizer）—— 重要备忘，勿随意删
==============================================================================
问题：部分三方 Claude API（如某些 B 渠道）的 SSE 响应中，message_start 与 message_delta
      两个事件对 usage 字段的报告存在严重不一致：
        message_start.usage  = {input:5719, cache_read:0, cache_create:49933}  ← 完整
        message_delta.usage  = {input:1850, cache_read:0, cache_create:16148}  ← 残缺
                                + cache_creation:{ephemeral_5m_input_tokens:16148}
      其中 message_delta 只汇报了”5 分钟短期缓存（ephemeral_5m）”那层 ~16K，
      把其余 ~39K 的真实上下文彻底省略。

      Claude Code 的 statusline 在 message_start.cache_read=0 时会 fallback 到
      message_delta 合计值，于是显示 ~16K 而非真实的 ~55K，用户误以为上下文丢失。

      实测证据（同一 55K 会话切换 B API 后）：
        message_start total = 55 652 tokens（完整上下文，正确）
        message_delta total =  17 998 tokens（ephemeral_5m，错误）
        statusline 显示     =      18K → 16K（每轮递减，实为 5min 缓存过期重建）

根治：在 _SSEUsageNormalizer._handle 里仅改写 Claude 的 message_delta：
  - message_start：只做静默记录（self._start_usage），不改写任何字段。
    ★ 不改 start 是刻意的：CC 用 start.cache_read 判断是否触发自动压缩；
      若把 cache_read 从 0 改为 ~123K，会导致 CC 错误触发压缩，即使模型
      支持 1M 上下文也会中途打断会话。保持 start.cache_read=0 → 不压缩。
  - message_delta：用 message_start 保存的真实总量覆盖 delta.cache_read，
    并删除 cache_creation.ephemeral_5m_input_tokens 等私有扩展字段。
    结果：delta.cache_read = message_start 真实总量，statusline 显示准确。
    ★ 这里可以改：CC 在 start.cache_read=0 时 fallback 读 delta 合计值来
      更新 statusline；delta 里放大数字只影响显示，不影响压缩判断。

副作用：
  - 对 A API（cache_read 已 >0）：start 不变，delta.cache_read 从 ~18K 升为
    ~65K，与 start 一致，statusline 无变化（CC 优先读 start.cache_read）。
  - 对 B API：start 不变（cache_read=0），delta.cache_read 从 ~16K 升为真实
    总量，statusline 从 ~16K 恢复为 ~55K；压缩逻辑不受影响。
  - 日志 resp.json 的 usage_normalized 记录本次是否触发过归一化。
"""
import argparse
import codecs
import json
import re
import logging
import logging.handlers
import shutil
import sys
import tarfile
import threading
import time
import urllib.parse
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

# ---------- CLI 参数 ----------
_DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "logs"

parser = argparse.ArgumentParser(
    description="LLM 反代日志工具（上游由请求路径首段的编码 URL/域名指定）"
)
parser.add_argument("--port", type=int, default=38787, help="本地监听端口")
parser.add_argument(
    "--log-dir",
    default=str(_DEFAULT_LOG_DIR),
    help=f"日志目录，默认为当前脚本目录下的 logs：{_DEFAULT_LOG_DIR}",
)
parser.add_argument(
    "--max-log-size",
    type=float,
    default=1.0,
    help="日志目录最大大小（GB），超出后按 mtime 升序清理最旧的会话归档，默认 1",
)
parser.add_argument(
    "--fix-usage",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="归一化 SSE usage 字段（修复 B API 的 statusline 上下文计数偏小问题），默认开启",
)
args, _ = parser.parse_known_args()

PORT = args.port
LOG_DIR = Path(args.log_dir)
LOG_DIR.mkdir(parents=True, exist_ok=True)
FIX_USAGE = args.fix_usage
MAX_LOG_BYTES = int(args.max_log_size * 1024 * 1024 * 1024)

# ---------- 日志 ----------
# proxy.log 用滚动文件由 logging 自管，与会话归档分开，避免被清理逻辑误删
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.handlers.RotatingFileHandler(
            LOG_DIR / "proxy.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("proxy")

# ---------- 过滤 hop-by-hop 头 ----------
HOP_BY_HOP = {
    "transfer-encoding", "connection", "keep-alive",
    "te", "trailers", "upgrade", "proxy-connection",
}


def filter_resp_headers(headers: httpx.Headers) -> dict:
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


def _resolve_target(request: Request) -> str:
    """解析上游目标 URL：目标必须由请求路径首段给出，无回退。

    支持三方 API 的额外路径前缀——前缀无论编码进首段、还是以明文斜杠跟在
    域名后，最终都按 base + 剩余路径 拼接，结果一致：
      - 编码完整 URL（可含前缀）：
          /https%3A%2F%2Fai.usmercari.com%2Fapi%2Fv3/v1/messages
            → https://ai.usmercari.com/api/v3/v1/messages
      - 只填域名（自动补 https）：
          /ai.usmercari.com/v1/messages        → https://ai.usmercari.com/v1/messages
      - 域名 + 明文前缀：
          /ai.usmercari.com/api/v3/v1/messages → https://ai.usmercari.com/api/v3/v1/messages
    首段既不是 http(s) URL 也不像主机名 → 抛 ValueError（由调用方回 400）。
    query string 原样透传。用 raw_path（未解码）以免 %2F 被提前解码而错位。
    """
    raw = request.scope.get("raw_path")
    raw_path = raw.decode("latin-1") if raw else request.url.path
    p = raw_path[1:] if raw_path.startswith("/") else raw_path

    enc_first, _, rest = p.partition("/")
    decoded = urllib.parse.unquote(enc_first)
    low = decoded.lower()
    if low.startswith(("http://", "https://")):
        base = decoded.rstrip("/")
    else:
        host = decoded.split("/", 1)[0].split(":", 1)[0]
        if host == "localhost" or "." in host:
            base = ("https://" + decoded).rstrip("/")
        else:
            raise ValueError(
                "缺少上游目标：请把目标 URL 或域名编码进路径首段，例如 "
                "/https%3A%2F%2Fai.usmercari.com%2F/v1/messages 或 "
                "/ai.usmercari.com/v1/messages"
            )

    url = base + (("/" + rest) if rest else "")
    if request.url.query:
        url += f"?{request.url.query}"
    return url


# ---------- 全局 httpx 客户端 ----------
http_client: httpx.AsyncClient = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(
        verify=False,
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=5.0),
        follow_redirects=True,
    )
    log.info(f"反代启动 http://0.0.0.0:{PORT}（上游由请求路径首段的编码 URL/域名指定）")
    log.info(
        f"日志目录: {LOG_DIR.resolve()}（上限 {MAX_LOG_BYTES / 1024 / 1024:.0f} MB，"
        f"超出按 mtime 自动清理）"
    )
    # 启动时先做一次清理，避免上次崩溃后残留把目录撑爆
    threading.Thread(target=_cleanup_logs, daemon=True).start()
    yield
    await http_client.aclose()


app = FastAPI(lifespan=lifespan)


# ---------- 主路由 ----------
@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def proxy(request: Request, path: str):
    try:
        url = _resolve_target(request)
    except ValueError as e:
        log.warning(f"目标解析失败: {e}")
        return Response(status_code=400)

    # 去掉会干扰 httpx 的头
    req_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in {"host", "content-length"}
    }
    body = await request.body()
    started_at = datetime.now()
    req_id = started_at.strftime("%Y%m%d_%H%M%S_%f")

    # 请求体格式化（尝试解析成结构化 JSON，否则保留字符串）
    try:
        body_text = body.decode("utf-8", errors="replace") if body else None
        if body_text:
            try:
                body_text = json.loads(body_text)
            except Exception:
                pass
    except Exception:
        body_text = "<binary>"

    parsed = urllib.parse.urlparse(url)
    model = _extract_model(body_text)
    fix_usage = FIX_USAGE and _model_wants_usage_fix(model)
    session = {
        "id": req_id,
        "ts": started_at.isoformat(timespec="milliseconds"),
        "method": request.method,
        "url": url,
        "host": parsed.hostname,
        "path": parsed.path,
        "model": model,
        "fix_usage_enabled": fix_usage,
        "request": {"headers": dict(req_headers), "body": body_text},
    }
    log.info(f"[{req_id}] → {request.method} {url}")

    # --- 发起上游请求（流式） ---
    upstream_req = http_client.build_request(
        method=request.method, url=url, headers=req_headers, content=body
    )
    upstream_resp = await http_client.send(upstream_req, stream=True)

    resp_headers = filter_resp_headers(upstream_resp.headers)
    content_type = upstream_resp.headers.get("content-type", "")
    is_sse = "text/event-stream" in content_type

    log.info(
        f"[{req_id}] ← {upstream_resp.status_code} "
        f"{'[SSE]' if is_sse else ''} {content_type}"
    )

    if is_sse:
        # 流式：边转发边缓存，流结束后写一份会话日志
        chunks: list[bytes] = []

        async def stream_gen():
            normalizer = _SSEUsageNormalizer(enabled=fix_usage)
            completed = False
            try:
                async for chunk in upstream_resp.aiter_bytes():
                    chunks.append(chunk)              # 原始留存，供日志/排查
                    out = normalizer.feed(chunk)      # 必要时归一化 Claude usage
                    if out:
                        yield out.encode("utf-8")
                # 仅在正常读完时吐残留——此处 yield 安全。
                tail = normalizer.flush()
                if tail:
                    yield tail.encode("utf-8")
                completed = True
            finally:
                # 关键：绝不在 finally 里 yield。客户端中途断开时（Agent 频繁并发/
                # 弃用子流，很常见），生成器会被关闭，在挂起的 yield 处抛
                # GeneratorExit/CancelledError；若此处再 yield，会触发 RuntimeError
                # 并中断 finally 剩余代码，导致 _write_session 跑不到、日志静默丢失。
                # 故这里只做（同步、不可中断的）落盘，再关上游连接。
                try:
                    full = b"".join(chunks).decode("utf-8", errors="replace")
                    session["status"] = upstream_resp.status_code
                    session["streaming"] = True
                    session["completed"] = completed  # False=客户端中途断开
                    session["usage_normalized"] = normalizer.usage_normalized
                    session["response"] = {
                        "headers": dict(upstream_resp.headers),
                        "events": _parse_sse_body(full),
                    }
                    _write_session(session)
                    _maybe_cleanup_logs()
                    log.info(
                        f"[{req_id}] SSE {'完成' if completed else '中断'}，"
                        f"{len(full)} 字符"
                        + ("，usage 已归一化" if normalizer.usage_normalized else "")
                    )
                except Exception as e:
                    log.warning(f"[{req_id}] SSE 会话记录失败: {e}")
                try:
                    await upstream_resp.aclose()
                except Exception:
                    pass

        return StreamingResponse(
            stream_gen(),
            status_code=upstream_resp.status_code,
            headers=resp_headers,
            media_type=content_type,
        )
    else:
        content = await upstream_resp.aread()
        await upstream_resp.aclose()

        try:
            body_decoded = content.decode("utf-8", errors="replace")
            try:
                body_decoded = json.loads(body_decoded)
            except Exception:
                pass
        except Exception:
            body_decoded = f"<binary {len(content)} bytes>"

        session["status"] = upstream_resp.status_code
        session["streaming"] = False
        session["response"] = {
            "headers": dict(upstream_resp.headers),
            "body": body_decoded,
        }
        _write_session(session)
        _maybe_cleanup_logs()
        log.info(f"[{req_id}] 完成，{len(content)} bytes")

        return Response(
            content=content,
            status_code=upstream_resp.status_code,
            headers=resp_headers,
            media_type=content_type,
        )


# ---------- 会话归档（每请求一文件，按 日期/小时 分目录；历史按天打 tar.gz） ----------
_log_lock = threading.Lock()
_DAY_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _today_name() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _session_file_path(req_id: str) -> Path:
    """单个请求的日志路径：LOG_DIR/YYYY-MM-DD/HH/<req_id>.json。

    日期与小时取自 req_id 前缀（与生成时一致），故同一请求始终落在同一文件。
    """
    day = f"{req_id[0:4]}-{req_id[4:6]}-{req_id[6:8]}"
    hour = req_id[9:11]
    return LOG_DIR / day / hour / f"{req_id}.json"


def _write_session(session: dict) -> None:
    """把一次完整请求+响应写成独立 JSON 文件（缩进美化，可直接打开阅读）。

    req_id 含微秒，几乎不会撞名；万一同微秒并发，追加序号避免互相覆盖。
    """
    path = _session_file_path(session["id"])
    try:
        with _log_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():  # 极端并发兜底：同微秒不覆盖
                path = path.with_name(f"{path.stem}_{id(session) & 0xffff:04x}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(session, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"日志写入失败 {path}: {e}")


def _archive_old_days() -> None:
    """把过去日期的目录整体打包成 <day>.tar.gz 后删除原目录；今天的目录不动。

    打包时若目标已存在（如跨日有迟到请求重建目录），用序号另存，绝不覆盖历史。
    """
    today = _today_name()
    for p in LOG_DIR.iterdir():
        if not p.is_dir() or not _DAY_DIR_RE.match(p.name) or p.name == today:
            continue
        gz = LOG_DIR / f"{p.name}.tar.gz"
        n = 1
        while gz.exists():
            gz = LOG_DIR / f"{p.name}.{n}.tar.gz"
            n += 1
        try:
            with tarfile.open(gz, "w:gz") as tar:
                tar.add(p, arcname=p.name)
            shutil.rmtree(p)
            log.info(f"日志归档: {p.name}/ → {gz.name}")
        except Exception as e:
            log.warning(f"归档失败 {p}: {e}")
            gz.unlink(missing_ok=True)  # 半成品清掉，下轮重来


# ---------- 惰性清理：启动一次 + 间隔触发，避免每次请求都 stat 整目录 ----------
_cleanup_lock = threading.Lock()
_last_cleanup_ts = 0.0
_CLEANUP_MIN_INTERVAL = 600.0  # 秒，两次清理至少相隔这么久


def _maybe_cleanup_logs() -> None:
    """惰性触发：距上次清理够久才真正做一次，否则立即返回（几乎零开销）。"""
    global _last_cleanup_ts
    now = time.monotonic()
    if now - _last_cleanup_ts < _CLEANUP_MIN_INTERVAL:
        return
    _last_cleanup_ts = now
    threading.Thread(target=_cleanup_logs, daemon=True).start()


def _cleanup_logs() -> None:
    """先把历史日期目录打包，再按 mtime 升序删最旧的文件直到总占用 ≤ 上限。

    不动今天的目录（在写）和 proxy.log*（由 logging 自管）。删空小时目录收尾。
    """
    if not _cleanup_lock.acquire(blocking=False):
        return
    try:
        _archive_old_days()

        today = _today_name()
        today_dir = LOG_DIR / today
        files: list[tuple[float, int, Path]] = []
        total = 0
        for p in LOG_DIR.rglob("*"):
            if not p.is_file() or p.name.startswith("proxy.log"):
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            files.append((st.st_mtime, st.st_size, p))
            total += st.st_size

        if total <= MAX_LOG_BYTES:
            return

        files.sort(key=lambda x: x[0])
        removed_n, removed_bytes = 0, 0
        for _, size, p in files:
            if total <= MAX_LOG_BYTES:
                break
            if today_dir in p.parents:  # 今天的活动文件不动
                continue
            try:
                p.unlink()
                total -= size
                removed_bytes += size
                removed_n += 1
            except OSError:
                pass

        # 删空目录收尾（自底向上，跳过今天的目录）
        for d in sorted(LOG_DIR.rglob("*"), key=lambda x: len(x.parts), reverse=True):
            if d.is_dir() and d != today_dir and today_dir not in d.parents:
                try:
                    d.rmdir()  # 仅当为空才成功
                except OSError:
                    pass

        if removed_n:
            log.info(
                f"日志清理: 删除 {removed_n} 个旧文件，释放 "
                f"{removed_bytes / 1024 / 1024:.1f} MB，当前 "
                f"{total / 1024 / 1024:.1f} MB / 上限 "
                f"{MAX_LOG_BYTES / 1024 / 1024:.0f} MB"
            )
    finally:
        _cleanup_lock.release()


def _parse_sse_body(raw: str) -> list:
    """把 SSE 文本解析成结构化列表，方便阅读"""
    events = []
    for line in raw.splitlines():
        if line.startswith("data:"):
            data = line[5:].strip()
            if data in ("[DONE]", ""):
                events.append(data)
                continue
            try:
                events.append(json.loads(data))
            except Exception:
                events.append(data)
        elif line.startswith("event:") or line.startswith("id:"):
            events.append(line)
    return events


# ---------- Claude usage 归一化 ----------
# 仅对 Claude 系模型启用（B API 冷缓存场景特有）。需扩展时往这里加关键字。
_USAGE_FIX_MODEL_KEYWORDS = ("claude",)

# 解析失败（请求体非标准 JSON）时，从原文里精确抠出 model 字段值兜底判定。
_MODEL_FIELD_RE = re.compile(r'"model"\s*:\s*"([^"]*)"')


def _extract_model(body_text) -> str:
    """从请求体取 model 名：已解析的 dict 直接读，否则用正则从原文抠。取不到返回 ""。"""
    if isinstance(body_text, dict):
        model = body_text.get("model")
        return model if isinstance(model, str) else ""
    if isinstance(body_text, str):
        m = _MODEL_FIELD_RE.search(body_text)
        if m:
            return m.group(1)
    return ""


def _model_wants_usage_fix(model: str) -> bool:
    """model 含 _USAGE_FIX_MODEL_KEYWORDS 任一关键字（不区分大小写）才做 usage 归一化。"""
    low = model.lower()
    return any(kw in low for kw in _USAGE_FIX_MODEL_KEYWORDS)


def _sse_event(event_type: str, payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event_type}\ndata: {data}\n\n"


class _SSEUsageNormalizer:
    """增量解析 Anthropic SSE 流，按需归一化 Claude usage，其余原样透传。"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._dec = codecs.getincrementaldecoder("utf-8")()
        self._buf = ""
        self._start_usage: dict | None = None  # message_start 时保存，供 delta 归一化
        self.usage_normalized = False

    def feed(self, chunk: bytes) -> str:
        text = self._dec.decode(chunk)
        if not self.enabled:
            return text
        self._buf += text
        return self._drain(final=False)

    def flush(self) -> str:
        text = self._dec.decode(b"", final=True)
        if not self.enabled:
            return text
        self._buf += text
        return self._drain(final=True)

    def _drain(self, final: bool) -> str:
        out = []
        while True:
            i = self._buf.find("\n\n")
            if i < 0:
                break
            event = self._buf[:i]
            self._buf = self._buf[i + 2:]
            out.append(self._handle(event))
        if final and self._buf:
            out.append(self._buf)  # 残留（不完整事件）原样吐出
            self._buf = ""
        return "".join(out)

    def _handle(self, event: str) -> str:
        data = None
        for line in event.split("\n"):
            if line.startswith("data:"):
                data = line[5:].strip()
        if not data:
            return event + "\n\n"
        try:
            obj = json.loads(data)
        except Exception:
            return event + "\n\n"

        t = obj.get("type")
        # --- usage 归一化（Claude 模型 + 全局开关开启）---
        # message_start 只做静默记录，不改写——避免 cache_read 被改大后触发
        # Claude Code 的自动压缩逻辑（CC 用 start.cache_read 作为压缩阈值判断）。
        # statusline 的修复只靠 message_delta 的归一化（见下），因为 CC 在
        # start.cache_read=0 时会 fallback 到 delta 合计值来更新显示。
        if t == "message_start":
            try:
                usage = obj.get("message", {}).get("usage", {})
                if isinstance(usage, dict):
                    self._start_usage = dict(usage)  # 仅保存，供 delta 归一化使用
            except Exception:
                pass

        if t == "message_delta" and self._start_usage is not None:
            try:
                start_total = (self._start_usage.get("input_tokens", 0)
                               + self._start_usage.get("cache_read_input_tokens", 0)
                               + self._start_usage.get("cache_creation_input_tokens", 0))
                if start_total > 0:
                    delta_usage = obj.get("usage", {})
                    new_usage = dict(delta_usage)
                    new_usage["cache_read_input_tokens"] = start_total
                    new_usage["input_tokens"] = 0
                    new_usage["cache_creation_input_tokens"] = 0
                    new_usage.pop("cache_creation", None)  # 删除 ephemeral_5m 私有字段
                    obj["usage"] = new_usage
                    self.usage_normalized = True
                    return _sse_event("message_delta", obj)
            except Exception:
                pass  # 解析异常原样透传

        return event + "\n\n"  # 其余事件原样透传


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
