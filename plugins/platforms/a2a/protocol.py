"""A2A protocol helpers — Agent Card, JSON-RPC framing, task store, conversation persistence.
Wire shape is A2A v1.0: SCREAMING_SNAKE_CASE states/roles; Parts and StreamResponse events are
discriminated by member presence (no ``kind``/``final``); SSE closure signals the terminal state.
Stdlib only. ``extract_text`` stays tolerant of v0.3 peers."""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from gateway.platforms._shared import coerce_port as _coerce_int
from hermes_constants import get_hermes_home

_log = logging.getLogger(__name__)

PROTOCOL_VERSION = "1.0"

# A2A v1.0 task lifecycle states + message roles.
STATE_SUBMITTED, STATE_WORKING, STATE_INPUT_REQUIRED = "TASK_STATE_SUBMITTED", "TASK_STATE_WORKING", "TASK_STATE_INPUT_REQUIRED"
STATE_COMPLETED, STATE_FAILED = "TASK_STATE_COMPLETED", "TASK_STATE_FAILED"
STATE_CANCELED, STATE_REJECTED = "TASK_STATE_CANCELED", "TASK_STATE_REJECTED"
TERMINAL_STATES = frozenset({STATE_COMPLETED, STATE_FAILED, STATE_CANCELED, STATE_REJECTED})
ROLE_USER, ROLE_AGENT = "ROLE_USER", "ROLE_AGENT"

# A reply starting with this marker is a clarification request -> TASK_STATE_INPUT_REQUIRED (marker stripped).
INPUT_REQUIRED_MARKER = "[INPUT_REQUIRED]"

# JSON-RPC / A2A error codes. -32001..-32003 are A2A spec-defined; custom errors
# live at -32050..-32059 (implementation-defined space, clear of the A2A block).
ERR_PARSE, ERR_INVALID_PARAMS, ERR_METHOD_NOT_FOUND = -32700, -32602, -32601
ERR_TASK_NOT_FOUND, ERR_TASK_NOT_CANCELABLE = -32001, -32002  # A2A spec: TaskNotFoundError / TaskNotCancelableError
ERR_UNAUTHORIZED, ERR_RATE_LIMITED, ERR_UNTRUSTED_PEER = -32050, -32051, -32052

# Anti-loop: max inbound turns per context. A2A_MAX_PINGPONG_TURNS env, capped at 20.
_DEFAULT_MAX_PINGPONG, _HARD_MAX_PINGPONG = 5, 20
_RATE_LIMIT_DEFAULT, _RATE_WINDOW = 60, 60.0  # requests per minute, window seconds


def _env_int(name: str, default: int) -> int:
    return _coerce_int(os.getenv(name, default), default)


def max_pingpong_turns() -> int:
    v = _env_int("A2A_MAX_PINGPONG_TURNS", _DEFAULT_MAX_PINGPONG)
    return max(1, min(v, _HARD_MAX_PINGPONG))


def now_iso() -> str:
    """ISO 8601 UTC timestamp with millisecond precision (A2A v1.0)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def build_agent_card(*, name: str, url: str, description: str, skills: Optional[list[dict]] = None,
                     streaming: bool = False, push_notifications: bool = False, auth_required: bool = False,
                     tenant: str = "") -> dict:
    """A2A v1.0 Agent Card. ``tenant`` is the optional multi-tenancy routing key on
    AgentInterface; when present, clients MUST echo it in request params."""
    iface: dict[str, Any] = {"url": url, "protocolBinding": "JSONRPC", "protocolVersion": PROTOCOL_VERSION, **({"tenant": tenant} if tenant else {})}
    card: dict[str, Any] = {
        "name": name,
        "description": description,
        "url": url,  # convenience for pre-1.0 clients; canonical is supportedInterfaces
        "version": "1.0.0",
        "provider": {"organization": os.getenv("A2A_PROVIDER_ORG", "Hermes Agent"), "url": os.getenv("A2A_PROVIDER_URL", "") or url},
        "supportedInterfaces": [iface],
        "capabilities": {"streaming": streaming, "pushNotifications": push_notifications,
                         "stateTransitionHistory": False, "extendedAgentCard": False},
        "defaultInputModes": ["text/plain"], "defaultOutputModes": ["text/plain"], "skills": skills or [],
    }
    if auth_required:
        card["securitySchemes"] = {"bearer": {"type": "http", "scheme": "bearer"}}
        card["security"] = [{"bearer": []}]
    return card


def skills_from_toolsets(toolsets: "list[str] | dict[str, list[str]] | None") -> list[dict]:
    """A2A skill descriptors from toolset names or a toolset -> tool-names mapping (tool names
    become tags, max 10)."""
    if not isinstance(toolsets, dict):
        toolsets = {ts: [] for ts in set(toolsets or [])}
    skills = [{"id": f"toolset.{name}", "name": name, "description": f"Hermes '{name}' capabilities",
               "tags": [name] + [str(t) for t in (toolsets[name] or [])][:10]} for name in sorted(toolsets)]
    return skills or [{"id": "general", "name": "general", "description": "General-purpose conversational agent", "tags": ["general"]}]


def jsonrpc_result(req_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def jsonrpc_error(req_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def send_message_response(payload: dict) -> dict:
    """v1.0 SendMessageResponse oneof: exactly one of ``task`` / ``message``."""
    if isinstance(payload, dict) and payload.get("status") and payload.get("id"):
        return {"task": payload}
    return {"message": payload}


def unwrap_send_message_response(result: Any) -> Any:
    """Task/Message inside a v1.0 response; legacy bare payloads pass through."""
    if isinstance(result, dict):
        if isinstance(result.get("task"), dict):
            return result["task"]
        if isinstance(result.get("message"), dict):
            return result["message"]
    return result


def stream_task(task: dict) -> dict:
    """v1.0 StreamResponse with a task member."""
    return {"task": task}


def new_task_id() -> str:
    return "task-" + uuid.uuid4().hex[:16]


def new_context_id() -> str:
    return "ctx-" + uuid.uuid4().hex[:16]


def text_part(text: str) -> dict:
    """v1.0 text Part (member-presence discriminated, no ``kind``)."""
    return {"text": text, "mediaType": "text/plain"}


def text_message(role: str, text: str, context_id: str = "") -> dict:
    """A2A v1.0 Message with a single text Part."""
    msg: dict[str, Any] = {"role": role, "parts": [text_part(text)], "messageId": uuid.uuid4().hex}
    if context_id:
        msg["contextId"] = context_id
    return msg


def _file_note(fname: str, body: str, mtype: str) -> str:
    label = f"[file: {fname}]" if fname else "[file]"
    return f"{label} {body}" + (f" ({mtype})" if mtype else "")


def _json_or_str(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(data)


def extract_text(message_or_params: dict) -> str:
    """Concatenated text from an A2A Message / Task-result / params payload. v1.0, v0.3
    (``kind``) and pre-0.3 (``type``) Parts all carry ``text``; file Parts render as
    URL/filename (raw base64 noted, not decoded); data Parts render their JSON."""
    msg = message_or_params.get("message", message_or_params)
    chunks = []
    for part in msg.get("parts", []) if isinstance(msg, dict) else []:
        if not isinstance(part, dict):
            continue
        if isinstance(txt := part.get("text"), str):
            chunks.append(txt)
        elif isinstance(url := part.get("url"), str) and url:
            chunks.append(_file_note(part.get("filename") or part.get("name") or "", url,
                                     part.get("mediaType") or part.get("mimeType") or ""))
        elif isinstance(v03 := part.get("file"), dict) and isinstance(v03.get("fileWithUri"), str):
            chunks.append(_file_note(v03.get("name") or "", v03["fileWithUri"], v03.get("mimeType") or ""))
        elif isinstance(part.get("raw"), str):
            chunks.append(_file_note(part.get("filename") or "", f"{len(part['raw'])} bytes base64-encoded",
                                     part.get("mediaType") or ""))
        elif (data := part.get("data")) is not None:
            chunks.append(f"[data ({part.get('mediaType') or 'application/json'})]\n{_json_or_str(data)}")
    return "\n".join(chunks).strip()


def extract_context_id(params: dict) -> str:
    """v1.0 puts contextId inside the Message; tolerate legacy top-level."""
    msg = params.get("message") or {}
    return (str(msg.get("contextId") or "") if isinstance(msg, dict) else "") or str(params.get("contextId") or "")


def extract_message_id(params: dict) -> str:
    """The peer's own ``message.messageId`` ("" when absent) — informational, never a routing key."""
    msg = params.get("message") or {}
    return str(msg.get("messageId") or "") if isinstance(msg, dict) else ""


# ``MessageEvent.metadata[EVENT_METADATA_KEY]`` is the host-owned A2A namespace on inbound events:
# the authenticated peer, the exact task/context identity, and a bounded copy of the peer's own
# ``message.metadata``. Nothing from the request body is ever placed at the top level of
# ``event.metadata`` (so peers cannot forge ``gateway_session_*`` routing keys), and ``task_id``
# is the exact execution association that final-output / cancel routing keys on.
EVENT_METADATA_KEY = "a2a"
MAX_REQUEST_METADATA_BYTES = 8192


def extract_request_metadata(params: dict) -> Optional[dict]:
    """Bounded copy of the peer's ``message.metadata`` (fallback: top-level ``metadata``).
    ``{}`` when absent; ``None`` when present but not a JSON object or over the size cap — callers
    must reject, never silently drop (a dropped delegation locator would demote a scoped task to an
    unrestricted ordinary conversation)."""
    msg = params.get("message")
    raw = msg.get("metadata") if isinstance(msg, dict) and msg.get("metadata") is not None else params.get("metadata")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        return None
    try:
        if len(json.dumps(raw, ensure_ascii=False).encode("utf-8")) > MAX_REQUEST_METADATA_BYTES:
            return None
    except (TypeError, ValueError):
        return None
    return json.loads(json.dumps(raw))


def inbound_event_metadata(*, peer: str, task_id: str, context_id: str, message_id: str = "",
                           agent_slug: str = "", tenant: str = "", request_metadata: Optional[dict] = None,
                           execution_epoch: int = 1, recovery: Optional[dict] = None) -> dict:
    """``{EVENT_METADATA_KEY: {...}}`` for an inbound MessageEvent. ``peer`` is the credential
    identity from ``do_POST``, never a body field. ``execution_epoch`` is the local recovery
    counter; ``recovery`` (``{"epoch", "reason"}``) is present only on a resumed execution."""
    return {EVENT_METADATA_KEY: {
        "peer": peer, "task_id": task_id, "context_id": context_id, "message_id": message_id,
        "agent": agent_slug, "tenant": tenant, "request_metadata": dict(request_metadata or {}),
        "execution_epoch": int(execution_epoch), **({"recovery": dict(recovery)} if recovery else {}),
    }}


def extract_return_immediately(params: dict) -> bool:
    """v1.0 ``configuration.returnImmediately`` (legacy v0.3 ``blocking: false``)."""
    cfg = params.get("configuration") or {}
    if not isinstance(cfg, dict):
        return False
    if cfg.get("returnImmediately") is not None:
        return bool(cfg.get("returnImmediately"))
    return cfg.get("blocking") is False


# Plugin lifecycle hook for A2A tasks (registered in ``hermes_cli.plugins.VALID_HOOKS``). Kwargs:
# ``phase`` and ``task`` (``TaskStore.lifecycle_view``). Phases: ``recover`` — before any resumed
# model work; a callback returning ``{"action": "deny", "reason": ...}`` or raising DENIES the
# recovery (the task fails, nothing runs). ``cancel_requested`` / ``closed`` are observers (the
# authority-close and fence points for an integration that bound credentials to the execution).
LIFECYCLE_HOOK = "a2a_task_lifecycle"
TASK_STORE_FILENAME = "a2a_tasks.json"


def build_task(task_id: str, context_id: str, state: str, agent_text: str = "", *, created_at: str = "") -> dict:
    """A2A v1.0 Task. ``created_at`` is accepted but NOT serialized: the v1.0 Task proto has no
    createdAt and strict ProtoJSON parsers (a2a-sdk) reject unknown fields."""
    task: dict[str, Any] = {"id": task_id, "contextId": context_id, "status": {"state": state, "timestamp": now_iso()}}
    if agent_text:
        task["status"]["message"] = text_message(ROLE_AGENT, agent_text, context_id)
        if state == STATE_COMPLETED:
            task["artifacts"] = [{"artifactId": uuid.uuid4().hex, "parts": [text_part(agent_text)]}]
    return task


def status_update(task_id: str, context_id: str, state: str, text: str = "") -> dict:
    """v1.0 StreamResponse with a statusUpdate member."""
    status: dict[str, Any] = {"state": state, "timestamp": now_iso()}
    if text:
        status["message"] = text_message(ROLE_AGENT, text, context_id)
    return {"statusUpdate": {"taskId": task_id, "contextId": context_id, "status": status}}


def artifact_update(task_id: str, context_id: str, text: str) -> dict:
    """v1.0 StreamResponse with an artifactUpdate member."""
    artifact = {"artifactId": uuid.uuid4().hex, "parts": [text_part(text)]}
    return {"artifactUpdate": {"taskId": task_id, "contextId": context_id, "artifact": artifact}}


def sse_data(payload: dict, req_id: Any = None) -> str:
    """One StreamResponse as an SSE data frame. §9.4 requires a full JSON-RPC envelope (a2a-sdk
    breaks on bare StreamResponses); ``req_id=None`` is the legacy no-envelope fallback."""
    envelope = jsonrpc_result(req_id, payload) if req_id is not None else payload
    return f"data: {json.dumps(envelope, ensure_ascii=False)}\n\n"


def sse_done() -> str:
    """Stream-closure marker as an SSE *comment* — ``data: {}`` would make JSON-RPC clients parse."""
    return ": done\n\n"


class TurnTracker:
    """Counts inbound turns per context_id; beyond max_pingpong_turns() the adapter rejects."""

    _TTL = 3600  # prune contexts idle longer than 1 hour

    def __init__(self) -> None:
        self._turns: dict[str, tuple[int, float]] = {}  # context_id -> (count, last_seen)
        self._lock = threading.Lock()

    def track(self, context_id: str) -> int:
        """Increment and return the turn count; prunes stale contexts."""
        with self._lock:
            now = time.time()
            self._turns = {cid: v for cid, v in self._turns.items() if now - v[1] <= self._TTL}
            count = self._turns.get(context_id, (0, now))[0] + 1
            self._turns[context_id] = (count, now)
            return count

    def reset(self, context_id: str) -> None:
        with self._lock:
            self._turns.pop(context_id, None)


class RateLimiter:
    """Sliding-window request limiter, one bucket per authenticated identity."""

    def __init__(self) -> None:
        self._buckets: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, identity: str) -> bool:
        with self._lock:
            limit = max(1, _env_int("A2A_RATE_LIMIT", _RATE_LIMIT_DEFAULT))
            now = time.time()
            bucket = self._buckets[identity]
            while bucket and now - bucket[0] > _RATE_WINDOW:
                bucket.popleft()
            if len(bucket) >= limit:
                return False
            bucket.append(now)
            return True


class Metrics:
    """Counters for A2A operations (module singleton ``metrics`` shared by the inbound adapter
    and outbound tools; not persisted)."""

    _COUNTERS = ("inbound_total", "outbound_total", "streams_started", "push_sent", "push_failed",
                 "tasks_completed", "tasks_failed", "anti_loop_triggers", "rate_limit_triggers")

    def __init__(self) -> None:
        for name in self._COUNTERS:
            setattr(self, name, 0)
        self._start_time = time.time()
        self._latencies: deque[float] = deque(maxlen=100)  # last 100 completed inbound tasks

    def record_latency(self, seconds: float) -> None:
        self._latencies.append(seconds)

    def avg_latency(self) -> float:
        return sum(self._latencies) / len(self._latencies) if self._latencies else 0.0

    def snapshot(self) -> dict[str, Any]:
        return {"uptime_seconds": round(time.time() - self._start_time, 1), **{n: getattr(self, n) for n in self._COUNTERS},
                "avg_latency_ms": round(self.avg_latency() * 1000, 1)}


metrics = Metrics()


class TaskStoreUnavailable(RuntimeError):
    """The durable task store could not be read; callers must fail closed, never start empty."""


class TaskStore:
    """A2A tasks, kept after completion for tasks/get. With ``path`` every mutation is persisted
    atomically as ``{"version": 1, "tasks": [...]}`` so a task, its FULL output and its execution
    linkage (session key, local execution epoch, cancel request) survive a gateway restart.
    Records carry agent slug, tenant and the authenticated peer that created them; readers pass a
    scope and get not-found outside it (spec authz rule). An empty scope component means "unscoped"
    and is only for in-process callers — the HTTP layer always passes the authenticated peer.

    Unreadable, corrupt or newer-version state sets ``unavailable`` (a diagnostic reason):
    ``create`` raises :class:`TaskStoreUnavailable` and reads answer not-found. A missing file is a
    fresh install. ``execution_epoch`` is the LOCAL recovery counter (1 at dispatch, +1 per resumed
    execution) and is unrelated to any remote integration's generation counter."""

    STORE_VERSION = 1
    _MAX_TERMINAL = 500
    _DEFAULTS: dict[str, Any] = {
        "reply": "", "push_url": "", "push_config_id": "", "request_metadata": {}, "message_id": "",
        "session_key": "", "execution_epoch": 1, "execution_started_at": None, "cancel_requested_at": None,
    }

    def __init__(self, path: "str | os.PathLike[str] | None" = None) -> None:
        self._tasks: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._watchers: dict[str, list[Future]] = {}
        self._lock = threading.Lock()
        self._path = Path(path) if path else None
        self.unavailable = ""
        if self._path is not None:
            self._load()

    # ---- persistence -------------------------------------------------------------------------

    def _load(self) -> None:
        try:
            raw = self._path.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            return
        except (OSError, UnicodeDecodeError) as e:
            self.unavailable = f"cannot read task store {self._path}: {e}"
            _log.error("A2A: %s — refusing new tasks until it is repaired", self.unavailable)
            return
        try:
            data = json.loads(raw)
            version, tasks = int(data["version"]), data["tasks"]
            if not isinstance(tasks, list):
                raise ValueError("'tasks' is not a list")
        except (ValueError, KeyError, TypeError) as e:  # JSONDecodeError is a ValueError
            self.unavailable = f"corrupt task store {self._path}: {e}"
            _log.error("A2A: %s — refusing new tasks until it is repaired", self.unavailable)
            return
        if version > self.STORE_VERSION:
            self.unavailable = (f"task store {self._path} is version {version} but this build supports "
                                f"{self.STORE_VERSION}; upgrade Hermes or move the file aside")
            _log.error("A2A: %s", self.unavailable)
            return
        for rec in tasks:
            if isinstance(rec, dict) and rec.get("task_id"):
                self._tasks[str(rec["task_id"])] = self._normalize(rec)

    @classmethod
    def _normalize(cls, rec: dict) -> dict:
        """Fill fields a record written by an older build lacks (additive migration)."""
        out = dict(rec)
        for key, default in cls._DEFAULTS.items():
            out.setdefault(key, copy.deepcopy(default))
        out.setdefault("updated_at", out.get("completed_at") or out.get("created_at") or time.time())
        return out

    def _save_locked(self) -> None:
        """Atomic whole-file rewrite. Caller holds the lock. A failed write is logged, never raised:
        the in-memory state stays authoritative for this process."""
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_name(self._path.name + ".tmp")
            tmp.write_text(json.dumps({"version": self.STORE_VERSION, "tasks": list(self._tasks.values())},
                                      ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self._path)
        except (OSError, TypeError, ValueError):
            _log.error("A2A: could not persist task store %s", self._path, exc_info=True)

    def _touch_locked(self, rec: dict) -> None:
        rec["updated_at"] = time.time()
        self._save_locked()

    # ---- scope helpers -----------------------------------------------------------------------

    @staticmethod
    def _in_scope(rec: dict, agent_slug: str = "", tenant: str = "", peer: str = "") -> bool:
        return not ((agent_slug and rec.get("agent_slug", "") != agent_slug) or (tenant and rec.get("tenant", "") != tenant)
                    or (peer and rec.get("peer", "") != peer))

    def _scoped(self, task_id: str, agent_slug: str = "", tenant: str = "", peer: str = "") -> Optional[dict]:
        """Live record if visible in scope. Caller holds the lock."""
        rec = self._tasks.get(task_id)
        return rec if rec and self._in_scope(rec, agent_slug, tenant, peer) else None

    def _push_rec(self, task_id: str, config_id: str = "", agent_slug: str = "", tenant: str = "", peer: str = "") -> Optional[dict]:
        """Scoped record that has a push config (matching ``config_id`` if given). Caller holds the lock."""
        rec = self._scoped(task_id, agent_slug, tenant, peer)
        if rec and rec.get("push_url") and (not config_id or rec.get("push_config_id") == config_id):
            return rec
        return None

    @staticmethod
    def _push_config_view(rec: dict) -> dict:
        return {"configId": rec.get("push_config_id") or "", "taskId": rec["task_id"],
                "createdAt": rec.get("created_iso", ""), "pushNotificationConfig": {"url": rec.get("push_url") or ""}}

    # ---- mutations ---------------------------------------------------------------------------

    def create(self, task_id: str, context_id: str, peer: str, agent_slug: str = "", tenant: str = "") -> dict:
        if self.unavailable:
            raise TaskStoreUnavailable(self.unavailable)
        now = time.time()
        rec = {"task_id": task_id, "context_id": context_id, "peer": peer, "agent_slug": agent_slug or "", "tenant": tenant or "",
               "state": STATE_SUBMITTED, "created_at": now, "created_iso": now_iso(), "updated_at": now,
               **copy.deepcopy(self._DEFAULTS)}
        with self._lock:
            self._tasks[task_id] = rec
            self._save_locked()
        return dict(rec)

    def set_dispatch(self, task_id: str, *, session_key: str, request_metadata: Optional[dict] = None, message_id: str = "") -> None:
        """Record the gateway session the task runs in plus the peer's bounded metadata / messageId."""
        with self._lock:
            if rec := self._tasks.get(task_id):
                rec.update(session_key=session_key, request_metadata=dict(request_metadata or {}), message_id=message_id)
                self._touch_locked(rec)

    def set_state(self, task_id: str, state: str) -> None:
        with self._lock:
            if (rec := self._tasks.get(task_id)) and rec["state"] not in TERMINAL_STATES:
                rec["state"] = state
                self._touch_locked(rec)

    def mark_execution_started(self, task_id: str) -> None:
        """The gateway began (or resumed) model work for this task."""
        with self._lock:
            if (rec := self._tasks.get(task_id)) and rec["state"] not in TERMINAL_STATES:
                rec["execution_started_at"] = time.time()
                self._touch_locked(rec)

    def begin_recovery(self, task_id: str) -> int:
        """Bump the local execution epoch before a resumed execution; returns the new epoch (0 if
        the task is not recoverable)."""
        with self._lock:
            rec = self._tasks.get(task_id)
            if not rec or rec["state"] in TERMINAL_STATES:
                return 0
            rec["execution_epoch"] = int(rec.get("execution_epoch") or 1) + 1
            rec["execution_started_at"] = None
            self._touch_locked(rec)
            return rec["execution_epoch"]

    def request_cancel(self, task_id: str) -> Optional[dict]:
        """Durably record a cancel request (idempotent); the record, or None when terminal/missing."""
        with self._lock:
            rec = self._tasks.get(task_id)
            if not rec or rec["state"] in TERMINAL_STATES:
                return None
            if not rec.get("cancel_requested_at"):
                rec["cancel_requested_at"] = time.time()
                self._touch_locked(rec)
            return dict(rec)

    def set_push_config(self, task_id: str, url: str, agent_slug: str = "", tenant: str = "", peer: str = "") -> Optional[dict]:
        """Attach a push notification config; returns the stored config or None."""
        with self._lock:
            if not (rec := self._scoped(task_id, agent_slug, tenant, peer)):
                return None
            rec["push_url"], rec["push_config_id"] = url, "cfg-" + uuid.uuid4().hex[:12]
            self._touch_locked(rec)
            return self._push_config_view(rec)

    def get_push_config(self, task_id: str, config_id: str = "", agent_slug: str = "", tenant: str = "", peer: str = "") -> Optional[dict]:
        with self._lock:
            return self._push_config_view(rec) if (rec := self._push_rec(task_id, config_id, agent_slug, tenant, peer)) else None

    def list_push_configs(self, task_id: str, agent_slug: str = "", tenant: str = "", peer: str = "") -> list[dict]:
        cfg = self.get_push_config(task_id, "", agent_slug, tenant, peer)
        return [cfg] if cfg else []

    def delete_push_config(self, task_id: str, config_id: str = "", agent_slug: str = "", tenant: str = "", peer: str = "") -> bool:
        with self._lock:
            rec = self._push_rec(task_id, config_id, agent_slug, tenant, peer)
            if rec:
                rec["push_url"] = rec["push_config_id"] = ""
                self._touch_locked(rec)
            return rec is not None

    def pop_push_url(self, task_id: str) -> str:
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec:
                url, rec["push_url"] = rec["push_url"], ""
                self._touch_locked(rec)
            return url if rec else ""

    def complete(self, task_id: str, state: str, reply: str = "") -> Optional[dict]:
        """Transition a task to a terminal state, storing the FULL reply. Idempotent."""
        with self._lock:
            rec = self._tasks.get(task_id)
            if not rec or rec["state"] in TERMINAL_STATES:
                return None
            rec.update(state=state, reply=reply, completed_at=time.time())
            watchers = self._watchers.pop(task_id, [])
            self._trim_locked()
            self._touch_locked(rec)
            out = dict(rec)
        for fut in watchers:
            if not fut.done():
                fut.set_result((state, reply))
        return out

    # ---- reads -------------------------------------------------------------------------------

    def get(self, task_id: str, agent_slug: str = "", tenant: str = "", peer: str = "") -> Optional[dict]:
        with self._lock:
            return dict(rec) if (rec := self._scoped(task_id, agent_slug, tenant, peer)) else None

    def context_peer(self, context_id: str) -> str:
        """Peer that owns ``context_id`` (creator of its oldest stored task), "" when unowned."""
        with self._lock:
            return next((rec["peer"] for rec in self._tasks.values() if rec["context_id"] == context_id), "")

    def in_flight(self, session_key: str = "") -> list[dict]:
        """Non-terminal records (optionally only those dispatched into ``session_key``), oldest first."""
        with self._lock:
            return [dict(r) for r in self._tasks.values()
                    if (not session_key or r.get("session_key") == session_key) and r["state"] not in TERMINAL_STATES]

    def watch(self, task_id: str, agent_slug: str = "", tenant: str = "", peer: str = "") -> Optional[Future]:
        with self._lock:
            if not (rec := self._scoped(task_id, agent_slug, tenant, peer)):
                return None
            fut: Future = Future()
            if rec["state"] in TERMINAL_STATES:
                fut.set_result((rec["state"], rec.get("reply", "")))
            else:
                self._watchers.setdefault(task_id, []).append(fut)
            return fut

    def list(self, context_id: str = "", state: str = "", page_size: int = 50, offset: int = 0,
             agent_slug: str = "", tenant: str = "", with_total: bool = False, peer: str = ""):
        """Filtered task page (newest first) as ``(records, next_offset)``, or
        ``(records, next_offset, total)`` with ``with_total`` (v1.0 ListTasks totalSize)."""
        page_size = max(1, min(int(page_size or 50), 100))
        with self._lock:
            recs = [dict(r) for r in reversed(self._tasks.values())
                    if self._in_scope(r, agent_slug, tenant, peer)
                    and (not context_id or r["context_id"] == context_id) and (not state or r["state"] == state)]
        total = len(recs)
        page = recs[offset:offset + page_size]
        next_offset = offset + page_size if offset + page_size < total else 0
        return (page, next_offset, total) if with_total else (page, next_offset)

    def fail_orphans(self, timeout_seconds: float = 300, *, exclude: set[str] | None = None) -> list[str]:
        """Fail non-terminal tasks idle (no ``updated_at`` touch) longer than ``timeout_seconds``,
        except ``exclude`` (tasks a request owns or whose execution is alive)."""
        excluded = exclude or set()
        with self._lock:
            stale = [tid for tid, rec in self._tasks.items()
                     if tid not in excluded and rec["state"] not in TERMINAL_STATES
                     and time.time() - (rec.get("updated_at") or rec["created_at"]) > timeout_seconds]
        return [tid for tid in stale if self.complete(tid, STATE_FAILED, "[task orphaned — no reply produced]")]

    def _trim_locked(self) -> None:
        terminal = [tid for tid, rec in self._tasks.items() if rec["state"] in TERMINAL_STATES]
        for tid in terminal[:max(0, len(terminal) - self._MAX_TERMINAL)]:
            self._tasks.pop(tid, None)

    @staticmethod
    def lifecycle_view(rec: dict) -> dict:
        """Read-only, secret-free record view handed to the ``a2a_task_lifecycle`` hook."""
        return {k: copy.deepcopy(rec.get(k)) for k in (
            "task_id", "context_id", "peer", "agent_slug", "tenant", "state", "execution_epoch",
            "execution_started_at", "cancel_requested_at", "request_metadata", "session_key", "message_id")}

    @staticmethod
    def to_task(rec: dict, include_artifacts: bool = True) -> dict:
        """Render a stored record as an A2A v1.0 Task (full reply, never a preview)."""
        task = build_task(rec["task_id"], rec["context_id"], rec["state"], rec.get("reply", ""),
                          created_at=rec.get("created_iso", ""))
        if not include_artifacts:
            task.pop("artifacts", None)
        if rec.get("cancel_requested_at") and rec["state"] not in TERMINAL_STATES:
            task["metadata"] = {"cancelRequested": True}  # requested, not yet a confirmed stop
        return task


def _conv_path(context_id: str) -> Path:
    safe = "".join(c for c in (context_id or "default") if c.isalnum() or c in "-_") or "default"
    return get_hermes_home() / "a2a_conversations" / f"{safe}.jsonl"


def persist_message(context_id: str, role: str, text: str, task_id: str = "") -> None:
    """Append one message to the context's on-disk conversation log. Never raises."""
    try:
        path = _conv_path(context_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), "role": role, "text": text, "task_id": task_id}, ensure_ascii=False) + "\n")
    except Exception:
        pass


def load_conversation(context_id: str, limit: int = 50) -> list[dict]:
    """Last *limit* messages for a context (empty list if none / unreadable)."""
    try:
        lines = _conv_path(context_id).read_text(encoding="utf-8-sig").splitlines()
    except Exception:
        return []
    out: list[dict] = []
    for line in lines:
        if line.strip():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                out.append(entry)
    return out[-limit:]


def list_conversations() -> list[str]:
    """Context-ids that have persisted conversations."""
    return sorted(p.stem for p in (get_hermes_home() / "a2a_conversations").glob("*.jsonl"))
