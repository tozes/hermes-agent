"""A2A inbound adapter: stdlib http.server (daemon thread) serving the Agent Card, /metrics and
JSON-RPC (message/send, message/stream SSE, tasks/*, push-config CRUD). Inbound tasks are framed
(security.wrap_inbound) and routed into the LIVE gateway session; ``send()`` fulfils the per-task
Future the HTTP handler blocks on. No token configured => binds 127.0.0.1 only."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import sqlite3
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FuturesTimeout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.config import Platform
from gateway.platforms._shared import coerce_port as _to_int, get_scoped_secret as _get_scoped_secret
from hermes_constants import get_hermes_home

from . import protocol, security

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 9900
# seconds: orphan grace floor / ceiling / watchdog period. The ceiling keeps the sweep
# meaningful when A2A_REPLY_TIMEOUT is absurd (1e18 would never fail an orphan).
_MIN_ORPHAN_TIMEOUT, _MAX_ORPHAN_TIMEOUT, _WATCHDOG_INTERVAL = 300, 86400, 60
_MAX_BODY = 1_048_576  # 1MB max request body — prevents DoS via memory exhaustion
_SSE_KEEPALIVE = 5  # seconds between SSE keepalive comments
_CANCEL_CONFIRM_SECONDS = 5  # how long tasks/cancel waits for the interrupted turn to confirm its stop
_PUSH_PREVIEW_CHARS = 2000  # push callbacks carry a preview; the full result is the stored Task (tasks/get)
_INTERRUPT_REASON_CANCEL = "a2a_task_cancel"
_DEFAULT_DESCRIPTION = "Hermes Agent — a general-purpose agent reachable over A2A."

_ok = protocol.jsonrpc_result
_err = protocol.jsonrpc_error

# (adapter handler, v1.0 PascalCase method per §5.3/§9.4, *legacy slash aliases still accepted)
_METHOD_TABLE = (
    ("_rpc_message_send", "SendMessage", "message/send"),
    ("_rpc_message_stream", "SendStreamingMessage", "message/stream"),
    ("_rpc_tasks_get", "GetTask", "tasks/get"),
    ("_rpc_tasks_list", "ListTasks", "tasks/list"),
    ("_rpc_tasks_cancel", "CancelTask", "tasks/cancel"),
    ("_rpc_tasks_subscribe", "SubscribeToTask", "tasks/subscribe"),
    ("_rpc_push_config_create", "CreateTaskPushNotificationConfig", "tasks/pushNotificationConfig/create",
     "tasks/pushNotificationConfig/set", "tasks/pushNotification/set"),
    ("_rpc_push_config_get", "GetTaskPushNotificationConfig", "tasks/pushNotificationConfig/get"),
    ("_rpc_push_config_list", "ListTaskPushNotificationConfigs", "tasks/pushNotificationConfig/list"),
    ("_rpc_push_config_delete", "DeleteTaskPushNotificationConfig", "tasks/pushNotificationConfig/delete"),
)
# JSON-RPC method -> (adapter handler name, is_v1)
_METHODS: dict[str, tuple[str, bool]] = {
    m: (handler, i == 0) for handler, *methods in _METHOD_TABLE for i, m in enumerate(methods)
}


def _task_store_path(extra: dict) -> Optional[str]:
    """Durable task store: ``extra.task_store`` (``":memory:"`` disables persistence — tests only),
    else ``<profile hermes home>/a2a_tasks.json``. Resolved at construction, inside profile scope."""
    configured = str(extra.get("task_store") or "").strip()
    if configured == ":memory:":
        return None
    return os.path.expanduser(configured) if configured else str(get_hermes_home() / protocol.TASK_STORE_FILENAME)


def _reply_timeout() -> float:
    """Seconds a blocking message/send long-polls for the answer before returning the task as it
    stands (WORKING). Never a cap on the work itself."""
    try:
        return max(1.0, float(os.getenv("A2A_REPLY_TIMEOUT", "300")))
    except (ValueError, TypeError):
        return 300.0


def _orphan_timeout() -> float:
    """Orphan grace must never expire before a configured reply window, but stays bounded."""
    return min(float(_MAX_ORPHAN_TIMEOUT), max(float(_MIN_ORPHAN_TIMEOUT), _reply_timeout()))


def _default_agent_name() -> str:
    # Scope-aware: a secondary multiplex profile must not borrow the default profile's A2A_AGENT_NAME.
    name = _get_scoped_secret("A2A_AGENT_NAME", "").strip()
    if name:
        return name
    try:
        import socket
        return f"hermes-{socket.gethostname()}"
    except Exception:
        return "hermes-agent"


def _clean_slug(value: str) -> str:
    """URL-safe-ish single-segment slug for a served agent ("" for the root agent)."""
    slug = str(value or "").strip().strip("/")
    return "" if slug in ("", "default", "root") else slug.split("/")[0]


def _join_url(base: str, prefix: str) -> str:
    base = (base or "").strip() or "/"
    base = base if base.endswith("/") else base + "/"
    prefix = (prefix or "").strip("/")
    return urllib.parse.urljoin(base, prefix + "/") if prefix else base


def _active_profile_name() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name() or "default"
    except Exception:
        return os.getenv("HERMES_PROFILE", "default") or "default"


def _profile_home(profile: str) -> Optional[str]:
    with contextlib.suppress(Exception):
        from hermes_cli.profiles import get_profile_dir
        return str(get_profile_dir(profile))
    if profile and profile != "default":
        return os.path.expanduser(f"~/.hermes/profiles/{profile}")
    with contextlib.suppress(Exception):
        from hermes_cli.config import get_hermes_home
        return str(get_hermes_home())
    return None


def _daemon_thread(target, name: str) -> threading.Thread:
    t = threading.Thread(target=target, name=name, daemon=True)
    t.start()
    return t


def _safe_context_slug(value: str, max_len: int = 96) -> str:
    """Sanitize attacker-provided context ids before using in session titles."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "")).strip("-._")
    return (slug or "ctx")[:max_len]


def _state_db(profile: str, sql: str, params: tuple, log_msg: str, *, commit: bool = False) -> str:
    """Run one statement against a profile's state.db; first column of the first row or ""."""
    home = _profile_home(profile)
    db = os.path.join(home, "state.db") if home else ""
    if not db or not os.path.exists(db):
        return ""
    try:
        with contextlib.closing(sqlite3.connect(db, timeout=5)) as con:
            cur = con.execute(sql, params)
            row = None if commit else cur.fetchone()
            if commit:
                con.commit()
        return str(row[0]) if row else ""
    except Exception:
        logger.debug(log_msg, exc_info=True)
        return ""


class A2ARequestHandler(BaseHTTPRequestHandler):
    """HTTP handler for the A2A JSON-RPC surface; all state lives on ``self.server.adapter``."""

    @property
    def adapter(self) -> "A2AAdapter":
        return self.server.adapter  # type: ignore[attr-defined]

    def log_message(self, format, *args):  # noqa: A002,N802
        logger.debug("A2A http: " + format, *args)  # silence the default stderr access log

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        for k, v in (("Content-Type", "application/json"), ("Content-Length", str(len(body)))):
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, http_code: int, req_id: Any, code: int, message: str):
        self._json(http_code, _err(req_id, code, message))

    def _client_ip(self) -> str:
        return self.client_address[0] if self.client_address else ""

    def _request_public_url(self) -> str:
        """A2A_PUBLIC_URL > X-Forwarded-Host / Host (scheme from X-Forwarded-Proto) > "" (bind host).

        Empty means "caller has no info, fall back to bind host". See #41711.

        A2A_PUBLIC_URL is read from ``self.adapter`` (captured at construction time, inside profile
        scope) rather than os.getenv here: do_GET/do_POST run on ThreadingHTTPServer's per-connection
        OS threads, which never inherit the profile scope contextvar, so a secondary multiplex
        profile's own A2A_PUBLIC_URL would otherwise resolve to the default profile's env value.
        """
        explicit = self.adapter._public_url
        if explicit:
            return explicit
        host = (self.headers.get("X-Forwarded-Host", "") or self.headers.get("Host", "")).split(",")[0].strip()
        scheme = (self.headers.get("X-Forwarded-Proto", "") or "http").split(",")[0].strip()
        return f"{scheme}://{host}/" if host else ""

    def do_GET(self):  # noqa: N802
        adapter = self.adapter
        route = adapter._route_for_path(self.path)
        agent = route["agent"]
        subpath = route["subpath"].rstrip("/") or "/"
        public_url = self._request_public_url() or None
        if subpath in ("/.well-known/agent.json", "/.well-known/agent-card.json"):
            return self._json(200, adapter._build_card(public_url, agent=agent))
        if subpath == "/metrics":
            return self._json(200, protocol.metrics.snapshot())
        if subpath not in ("/", "/health"):
            return self._json(404, {"error": "not found"})
        payload = {"status": "ok", "agent": agent.get("name") or adapter.agent_name}
        # Agent Cards are public; profile/tenant topology is not leaked on remote unauthenticated GETs.
        sec = adapter._security_context
        if sec.localhost_only() or sec.authenticate(self.headers.get("Authorization"), self._client_ip()) is not None:
            payload["served_agents"] = adapter._served_agent_summary(public_url=public_url)
        self._json(200, payload)

    def do_POST(self):  # noqa: N802
        adapter = self.adapter
        # Identity comes from the credential (or the socket in localhost-only mode) — never the body.
        identity = adapter._security_context.authenticate(self.headers.get("Authorization"), self._client_ip())
        if identity is None:
            return self._error(401, None, protocol.ERR_UNAUTHORIZED, "unauthorized")
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > _MAX_BODY:
                return self._error(413, None, protocol.ERR_PARSE, "payload too large")
            req = json.loads((self.rfile.read(length) if length else b"{}").decode("utf-8"))
        except Exception:
            return self._error(400, None, protocol.ERR_PARSE, "parse error")
        if not isinstance(req, dict):
            return self._error(400, None, protocol.ERR_INVALID_PARAMS, "JSON-RPC request must be an object")
        req_id, method = req.get("id"), str(req.get("method", ""))
        params = req["params"] if req.get("params") is not None else {}
        version = (self.headers.get("A2A-Version") or "").strip()
        route = adapter._route_for_request(self.path, params) if isinstance(params, dict) else {}
        handler_name, is_v1 = _METHODS.get(method, ("", False))
        # Ordered, lazily-evaluated checks -> (http status, error code, message); first failure wins
        # (the rate limiter must not be consulted for requests rejected before it).
        checks = (
            (lambda: not isinstance(params, dict), 200, protocol.ERR_INVALID_PARAMS, "params must be an object"),
            (lambda: version and version not in {"1.0", "1.0.0"}, 200, protocol.ERR_INVALID_PARAMS, f"unsupported A2A-Version: {version}"),
            (lambda: route.get("error"), 400, protocol.ERR_INVALID_PARAMS, route.get("error")),
            (lambda: not adapter._rate_limiter.allow(identity), 429, protocol.ERR_RATE_LIMITED, "rate limit exceeded"),
            (lambda: not adapter._security_context.is_trusted_peer(identity), 403, protocol.ERR_UNTRUSTED_PEER, f"peer '{identity}' not trusted"),
            (lambda: not handler_name, 200, protocol.ERR_METHOD_NOT_FOUND, f"method not found: {method}"),
        )
        for failed, http, code, msg in checks:
            if failed():
                if code == protocol.ERR_RATE_LIMITED:
                    protocol.metrics.rate_limit_triggers += 1
                return self._error(http, req_id, code, msg)
        agent = route["agent"]
        if handler_name == "_rpc_message_send":
            self._json(200, adapter._rpc_message_send(req_id, params, identity, agent=agent, v1_response=is_v1))
        elif handler_name == "_rpc_message_stream":
            adapter._rpc_message_stream(self, req_id, params, identity, agent=agent)
        elif handler_name == "_rpc_tasks_subscribe":
            adapter._rpc_tasks_subscribe(self, req_id, params, agent=agent, peer=identity)
        else:  # plain JSON task / push-config queries, visible only to the peer that created the task
            self._json(200, getattr(adapter, handler_name)(req_id, params, agent=agent, peer=identity))


class A2AAdapter(BasePlatformAdapter):
    """Inbound A2A server adapter."""

    def __init__(self, config, **kwargs):
        super().__init__(config=config, platform=Platform("a2a"))
        extra = getattr(config, "extra", {}) or {}
        # Scope-aware: a secondary multiplex profile must not borrow the default profile's bridged
        # A2A_PORT (falls closed to the module default). advertised_toolsets is deliberately unscoped.
        # (advertised_toolsets has the same env-leak shape but is left unscoped here — see the "Scope note"
        # in this fix's PR description: open PR #98937 is actively rewriting this field's None-vs-empty-list
        # semantics.)
        self._security_context = security.A2ASecurityContext.capture()
        _port_env = _get_scoped_secret("A2A_PORT")
        self.port = int(_port_env or extra.get("port", _DEFAULT_PORT))
        self.host = self._security_context.resolve_bind_host()
        self.agent_name = _default_agent_name()
        configured_toolsets = list(extra.get("advertised_toolsets") or []) or _get_scoped_secret("A2A_ADVERTISED_TOOLSETS", "").split(",")
        self._advertised_toolsets = [t.strip() for t in configured_toolsets if str(t).strip()]
        self._active_profile = _active_profile_name()
        # Captured here (construction runs inside _profile_runtime_scope), not read at request time:
        # do_GET/do_POST run on ThreadingHTTPServer's per-connection OS threads, which never inherit
        # the profile scope contextvar (same class as A2A_PORT above).
        self._public_url = _get_scoped_secret("A2A_PUBLIC_URL", "").strip()
        self._agents = self._load_served_agents(extra)
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._server_thread = self._watchdog_thread = None  # type: Optional[threading.Thread]
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._watchdog_stop = threading.Event()
        # Per-adapter protocol state (not module-global). The task store is durable per profile.
        self.tasks = protocol.TaskStore(_task_store_path(extra))
        self._turns, self._rate_limiter = protocol.TurnTracker(), protocol.RateLimiter()
        # Forwarded profile sessions: (profile, agent_slug, context_id) -> session_id.
        self._profile_sessions: Dict[tuple[str, str, str], str] = {}
        self._profile_session_locks: Dict[tuple[str, str, str], threading.Lock] = {}
        self._profile_session_locks_guard = threading.Lock()
        # Tasks an HTTP request currently owns (a waiter or a synchronous profile forward); the
        # orphan watchdog never fails these. Completion itself is event-owned (_complete_task):
        # waiters observe TaskStore.watch futures, they do not own the execution.
        self._active_tasks: set[str] = set()
        self._pending_lock = threading.Lock()
        # session_key -> task_id whose model turn the gateway is running right now (on_processing_*).
        self._executing: Dict[str, str] = {}
        self._stopping = threading.Event()

    @property
    def name(self) -> str:
        return "A2A"

    @property
    def authorization_is_upstream(self) -> bool:
        """Requests are authenticated in ``do_POST``; the gateway's A2A_ALLOWED_USERS list would
        otherwise reject peers (identity is a token-derived name or IP). Wrong credentials still 401.

        This is authorization delegated to the A2A bearer-token transport, not a fail-open: every request is
        401'd if the credential is wrong. Reported by kuangmi-bit (PR #41711 comment, Jun 27).
        """
        return True

    async def connect(self, **_kwargs) -> bool:
        # Capture the gateway loop so the HTTP thread can marshal events via run_coroutine_threadsafe.
        self._loop = asyncio.get_running_loop()
        try:
            self._httpd = ThreadingHTTPServer((self.host, self.port), A2ARequestHandler)
        except OSError as e:
            logger.error("A2A: could not bind %s:%s — %s", self.host, self.port, e)
            self._set_fatal_error("bind_failed", f"A2A bind failed: {e}", retryable=True)
            return False
        self._httpd.daemon_threads = True
        self._httpd.adapter = self  # type: ignore[attr-defined]
        self._server_thread = _daemon_thread(self._httpd.serve_forever, "a2a-http")
        self._watchdog_stop.clear()  # disconnect sets it; reset for reconnection
        self._stopping.clear()
        self._watchdog_thread = _daemon_thread(self._watchdog_loop, "a2a-watchdog")
        self._mark_connected()
        logger.info("A2A: serving Agent Card + JSON-RPC on http://%s:%s (%s) as %r; %d routed agent(s)", self.host, self.port,
                    "localhost-only" if self._security_context.localhost_only() else "REMOTE (bearer auth)", self.agent_name, len(self._agents))
        sec = self._security_context
        if sec.dispatch_fails_closed():
            logger.error("A2A: exposed on non-loopback bind %s with no A2A_TRUSTED_PEERS; every dispatch will be refused (403). "
                         "Set A2A_TRUSTED_PEERS, or A2A_ALLOW_ALL_USERS=true for a trusted network.", self.host)
        self._wire_plugin_handlers(None)  # plugin-registered native handlers
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
        self._watchdog_stop.set()
        if self._httpd is not None:
            with contextlib.suppress(Exception):
                self._httpd.shutdown()
                self._httpd.server_close()
            self._httpd = None
        # Release blocked HTTP waiters WITHOUT failing their tasks: in-flight work stays WORKING in
        # the durable store and is recovered after the restart (see handle_message).
        self._stopping.set()
        with self._pending_lock:
            self._active_tasks.clear()
            self._executing.clear()

    def _watchdog_loop(self) -> None:
        """Background thread that fails orphaned tasks (keeps them queryable)."""
        while not self._watchdog_stop.wait(_WATCHDOG_INTERVAL):
            try:
                self._fail_orphans_once()
            except Exception:
                logger.debug("A2A: watchdog error", exc_info=True)

    def _fail_orphans_once(self) -> list[str]:
        """Inactivity control (not a wall-clock cap): fail an idle task only when no request owns
        it AND its execution is not alive in the gateway."""
        with self._pending_lock:
            keep = set(self._active_tasks)
        keep |= {r["task_id"] for r in self.tasks.in_flight() if self._execution_alive(r)}
        timeout = _orphan_timeout()
        failed = self.tasks.fail_orphans(timeout, exclude=keep)
        for tid in failed:
            logger.warning("A2A: orphaned task %s marked failed (idle > %gs, execution not alive)", tid, timeout)
            protocol.metrics.tasks_failed += 1
            if rec := self.tasks.get(tid):
                self._lifecycle("closed", rec)
        return failed

    def _load_served_agents(self, extra: dict) -> dict[str, dict]:
        """Served-agent routing from ``platforms.a2a.extra.agents`` (top-level ``a2a_served_agents``
        fallback for scripts/tests). Root/default always maps to the live gateway session."""
        raw = extra.get("agents") or extra.get("served_agents")
        if raw is None:
            try:
                from hermes_cli.config import load_config
                cfg = load_config() or {}
            except Exception:
                cfg = {}
            cfg = cfg if isinstance(cfg, dict) else {}
            raw = cfg.get("a2a_served_agents") or (cfg.get("a2a") or {}).get("served_agents")
        # Scope-aware like port: a secondary profile must not inherit A2A_AGENT_DESCRIPTION.
        default_desc = _get_scoped_secret("A2A_AGENT_DESCRIPTION", _DEFAULT_DESCRIPTION)
        agents: dict[str, dict] = {"": {
            "slug": "", "path": "", "tenant": "", "profile": self._active_profile, "local": True,
            "name": self.agent_name, "description": default_desc, "advertised_toolsets": self._advertised_toolsets,
        }}
        reserved = {"health", "metrics", ".well-known"}
        tenants: dict[str, str] = {}
        items = raw.items() if isinstance(raw, dict) else enumerate(raw or []) if isinstance(raw, list) else []
        for key, val in items:
            if not isinstance(val, dict):
                continue
            slug = _clean_slug(str(val.get("slug") or val.get("id") or key))
            if not slug:
                continue
            path_segment = _clean_slug(str(val.get("path") or slug))
            if not path_segment or path_segment in reserved:
                logger.warning("A2A: ignoring served agent %r with reserved/invalid path %r", slug, path_segment)
                continue
            profile = str(val.get("profile") or slug).strip()
            toolsets = val.get("advertised_toolsets") or val.get("toolsets") or val.get("capabilities") or []
            if isinstance(toolsets, str):
                toolsets = [t.strip() for t in toolsets.split(",") if t.strip()]
            tenant = str(val.get("tenant") or slug).strip()
            if tenant:
                if tenant in tenants:
                    logger.warning("A2A: ignoring served agent %r with duplicate tenant %r already used by %r",
                                   slug, tenant, tenants[tenant])
                    continue
                tenants[tenant] = slug
            agents[slug] = {
                "slug": slug, "path": "/" + path_segment, "tenant": tenant, "profile": profile or slug,
                "local": bool(val.get("local")) or profile in ("", "default", self._active_profile),
                "name": str(val.get("name") or f"Hermes {slug}"),
                "description": str(val.get("description") or f"Hermes profile '{profile or slug}' exposed over A2A."),
                "advertised_toolsets": list(toolsets or []),
                "timeout": int(val.get("timeout") or _reply_timeout()),
            }
        return agents

    def _base_url(self, public_url: Optional[str]) -> str:
        return (public_url or "").strip() or f"http://{self.host}:{self.port}/"

    def _served_agent_summary(self, public_url: Optional[str] = None) -> list[dict]:
        base = self._base_url(public_url)
        return [{"slug": a["slug"] or "default", "name": a.get("name"), "url": _join_url(base, a.get("path", "")),
                 "tenant": a.get("tenant") or None, "profile": a.get("profile"), "local": bool(a.get("local"))}
                for a in self._agents.values()]

    def _route_for_path(self, raw_path: str) -> dict:
        path = urllib.parse.urlsplit(raw_path or "/").path or "/"
        # Longest prefix wins. Default/root agent is the fallback.
        for agent in sorted(self._agents.values(), key=lambda a: len(a.get("path", "")), reverse=True):
            if (prefix := agent.get("path", "") or "") and (path == prefix or path.startswith(prefix + "/")):
                return {"agent": agent, "subpath": path[len(prefix):] or "/"}
        return {"agent": self._agents[""], "subpath": path}

    def _route_for_request(self, raw_path: str, params: dict) -> dict:
        route = self._route_for_path(raw_path)
        agent = route["agent"]
        tenant = str((params or {}).get("tenant") or "")
        # If no URL prefix chose a non-default agent, allow v1.0 tenant routing.
        if agent.get("slug") == "" and tenant and (matches := [a for a in self._agents.values() if a.get("tenant") == tenant]):
            agent = matches[0]
            route = {"agent": agent, "subpath": route["subpath"]}
        expected = str(agent.get("tenant") or "")
        if tenant and expected and tenant != expected:
            return {"error": f"tenant {tenant!r} does not match routed agent {agent.get('slug') or 'default'}"}
        return route

    def _build_card(self, public_url: Optional[str] = None, agent: Optional[dict] = None) -> dict:
        # Per-request public URL beats the bind host so peers behind a reverse proxy can call back.
        agent = agent or self._agents[""]
        return protocol.build_agent_card(
            name=agent.get("name") or self.agent_name, url=_join_url(self._base_url(public_url), agent.get("path", "")),
            description=agent.get("description") or _DEFAULT_DESCRIPTION, skills=self._advertised_skills(agent),
            streaming=bool(agent.get("local", True)), push_notifications=True,
            auth_required=not self._security_context.localhost_only(), tenant=str(agent.get("tenant") or ""),
        )

    def _advertised_skills(self, agent: Optional[dict] = None) -> list[dict]:
        """Agent Card skills from the live tool registry, restricted by ``advertised_toolsets``;
        static fallback without a registry."""
        configured = (agent or {}).get("advertised_toolsets") if agent else self._advertised_toolsets
        try:
            from tools.registry import registry as tool_registry
            allowed = set(configured or []) or None
            mapping = {n: tool_registry.get_tool_names_for_toolset(n)
                       for n in tool_registry.get_registered_toolset_names() if allowed is None or n in allowed}
            if mapping:
                return protocol.skills_from_toolsets(mapping)
        except Exception:
            logger.debug("A2A: tool registry unavailable for Agent Card", exc_info=True)
        return protocol.skills_from_toolsets(configured or [])

    def _activate_task(self, task_id: str) -> None:
        with self._pending_lock:
            self._active_tasks.add(task_id)

    def _release_task(self, task_id: str) -> None:
        """The HTTP request no longer owns the task; the task itself keeps running."""
        with self._pending_lock:
            self._active_tasks.discard(task_id)

    def _execution_alive(self, rec: dict) -> bool:
        """Is the gateway running (or resuming) this task's turn?"""
        with self._pending_lock:
            if rec["task_id"] in self._executing.values():
                return True
        session_key = rec.get("session_key") or ""
        running = getattr(self.gateway_runner, "_is_session_running", None)
        return bool(session_key and callable(running) and running(session_key))

    def _lifecycle(self, phase: str, rec: dict) -> Optional[str]:
        """Fire ``protocol.LIFECYCLE_HOOK``. For ``recover`` returns the denial reason when any
        callback denies or raises (fail closed); other phases are observers and return None."""
        try:
            from hermes_cli.plugins import iter_hook_callbacks
            callbacks = iter_hook_callbacks(protocol.LIFECYCLE_HOOK)
        except Exception:
            logger.debug("A2A: lifecycle hook discovery failed", exc_info=True)
            callbacks = ()
        view = protocol.TaskStore.lifecycle_view(rec)
        for cb in callbacks:
            try:
                result = cb(phase=phase, task=dict(view))
            except Exception as e:  # plugin boundary: recover fails closed, observers log and continue
                logger.warning("A2A: %s(%s) hook failed for task %s", protocol.LIFECYCLE_HOOK, phase, rec["task_id"], exc_info=True)
                if phase == "recover":
                    return f"{getattr(cb, '__name__', 'hook')} raised: {e}"
                continue
            if phase == "recover" and isinstance(result, dict) and result.get("action") == "deny":
                return str(result.get("reason") or "recovery denied by plugin")
        return None

    def _source_for_record(self, rec: dict):
        peer = rec["peer"]
        return self.build_source(chat_id=rec["context_id"], chat_name=f"a2a:{peer}", chat_type="dm", user_id=peer, user_name=peer)

    def _scope_for_agent(self, agent: Optional[dict]) -> tuple[str, str]:
        return tuple(str((agent or self._agents[""]).get(k) or "") for k in ("slug", "tenant"))

    def _scope(self, agent: Optional[dict], peer: str) -> tuple[str, str, str]:
        """Read scope for the task store: (agent slug, tenant, authenticated peer)."""
        return (*self._scope_for_agent(agent), peer or "")

    def _forward_lock(self, key: tuple[str, str, str]) -> threading.Lock:
        with self._profile_session_locks_guard:
            return self._profile_session_locks.setdefault(key, threading.Lock())

    def _end_task(self, rec: dict, state: str, text: str, stored_reply: str = "") -> tuple[dict, None]:
        """Complete a task immediately (rejected / not ready) and build its terminal Task."""
        self.tasks.complete(rec["task_id"], state, stored_reply)
        protocol.metrics.tasks_failed += state == protocol.STATE_FAILED
        return protocol.build_task(rec["task_id"], rec["context_id"], state, text, created_at=rec["created_iso"]), None

    def _prepare_task(self, params: dict, peer: str, agent: Optional[dict] = None) -> tuple[Optional[dict], Optional[dict]]:
        """Validate, register, and dispatch an inbound message (HTTP worker thread). Returns
        (terminal_task, None) when it ends immediately, else (None, pending) with the future to wait on."""
        agent = agent or self._agents[""]
        text = protocol.extract_text(params)
        context_id = protocol.extract_context_id(params) or protocol.new_context_id()
        task_id = protocol.new_task_id()
        # A context belongs to the peer that opened it: another authenticated peer naming the same
        # contextId is refused up front (no record — nothing of the owner's is revealed or joined).
        if (owner := self.tasks.context_peer(context_id)) and owner != peer:
            logger.warning("A2A: peer %r refused on context %s owned by another peer", peer, context_id)
            return protocol.build_task(task_id, context_id, protocol.STATE_REJECTED,
                                       f"Context {context_id} belongs to another peer."), None
        turn = self._turns.track(context_id)
        max_turns = protocol.max_pingpong_turns()
        try:
            rec = self.tasks.create(task_id, context_id, peer, *self._scope_for_agent(agent))
        except protocol.TaskStoreUnavailable as e:  # fail closed and say why; never pretend the store is empty
            return protocol.build_task(task_id, context_id, protocol.STATE_FAILED, f"Task store unavailable: {e}"), None
        if turn > max_turns:
            protocol.metrics.anti_loop_triggers += 1
            logger.warning("A2A: anti-loop triggered for context %s (turn %d > %d)", context_id, turn, max_turns)
            return self._end_task(rec, protocol.STATE_REJECTED, f"Anti-loop protection: context {context_id} exceeded "
                                  f"{max_turns} turns. Start a new context or increase A2A_MAX_PINGPONG_TURNS.")
        if not text:
            return self._end_task(rec, protocol.STATE_REJECTED, "Empty task — nothing to do.")
        request_metadata = protocol.extract_request_metadata(params)
        if request_metadata is None:  # fail closed: a dropped locator must not demote a scoped task
            return self._end_task(rec, protocol.STATE_REJECTED, "message.metadata must be a JSON object of at most "
                                  f"{protocol.MAX_REQUEST_METADATA_BYTES} bytes.")
        framed = security.wrap_inbound(peer, text)
        security.audit("inbound", peer, task_id, text)
        protocol.persist_message(context_id, "user", text, task_id)
        protocol.metrics.inbound_total += 1
        self._register_inline_push(task_id, params, agent=agent, peer=peer)
        if not agent.get("local", True):
            self._activate_task(task_id)
            try:
                reply, state = self._forward_to_profile(agent, peer, context_id, framed)
                self._complete_task(task_id, state, reply)
                return protocol.TaskStore.to_task(self.tasks.get(task_id) or rec), None
            finally:
                self._release_task(task_id)
        if self._loop is None or self._message_handler is None:
            return self._end_task(rec, protocol.STATE_FAILED, "Agent gateway not ready to accept A2A tasks.")
        agent_slug, tenant = self._scope_for_agent(agent)
        message_id = protocol.extract_message_id(params)
        event = MessageEvent(text=framed, message_type=MessageType.TEXT, message_id=task_id,
                             source=self.build_source(chat_id=context_id, chat_name=f"a2a:{peer}", chat_type="dm", user_id=peer, user_name=peer),
                             metadata=protocol.inbound_event_metadata(
                                 peer=peer, task_id=task_id, context_id=context_id, message_id=message_id,
                                 agent_slug=agent_slug, tenant=tenant, request_metadata=request_metadata))
        # Durable session linkage BEFORE dispatch: a crash right after this still recovers the task.
        self.tasks.set_dispatch(task_id, session_key=self._event_session_key(event),
                                request_metadata=request_metadata, message_id=message_id)
        self._activate_task(task_id)
        fut = self.tasks.watch(task_id)
        try:
            asyncio.run_coroutine_threadsafe(self.handle_message(event), self._loop)
        except Exception as e:
            msg = security.redact_outbound(f"Dispatch failed: {e}")
            try:
                return self._end_task(rec, protocol.STATE_FAILED, msg, stored_reply=msg)
            finally:
                self._release_task(task_id)
        self.tasks.set_state(task_id, protocol.STATE_WORKING)
        return None, {"task_id": task_id, "context_id": context_id, "peer": peer, "future": fut, "created_iso": rec["created_iso"], "started": time.time()}

    def _forward_to_profile(self, agent: dict, peer: str, context_id: str, framed_text: str) -> tuple[str, str]:
        """Forward a routed task to another local profile via ``hermes chat``. First contact creates a
        ``source=a2a`` session and titles it deterministically; later turns ``--resume`` that id."""
        profile = str(agent.get("profile") or agent.get("slug") or "").strip()
        slug = str(agent.get("slug") or profile or "agent")
        safe_ctx = _safe_context_slug(context_id)
        session_title = f"a2a-{slug}-{safe_ctx}"
        key = (profile or "default", slug, safe_ctx)
        timeout = int(agent.get("timeout") or _reply_timeout())
        with self._forward_lock(key):
            session_id = self._profile_sessions.get(key) or _state_db(
                profile, "SELECT id FROM sessions WHERE title = ? ORDER BY started_at DESC LIMIT 1",
                (session_title,), "A2A: could not lookup forwarded session")
            cmd = ["hermes", "chat", "-q", framed_text, "-Q", "--source", "a2a"] + (["--resume", session_id] if session_id else [])
            # The child IS the target profile's turn: build its env for that home (launch .env /
            # TERMINAL_* residue dropped, the target's own secrets overlaid), not the gateway's raw environ.
            from tools.environments.local import served_profile_child_env
            env = served_profile_child_env(target_home=_profile_home(profile), inherit_credentials=True)
            env["HERMES_A2A_PEER"] = peer
            start = time.time()
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                      timeout=timeout, env=env, check=False, stdin=subprocess.DEVNULL)
            except subprocess.TimeoutExpired:
                return "[profile did not reply in time]", protocol.STATE_FAILED
            except Exception as e:
                return security.redact_outbound(f"Profile dispatch failed: {e}"), protocol.STATE_FAILED
            if proc.returncode != 0:
                msg = (proc.stderr or proc.stdout or f"profile exited {proc.returncode}").strip()
                return security.redact_outbound(msg[-2000:]), protocol.STATE_FAILED
            if not session_id and (session_id := _state_db(
                    profile, "SELECT id FROM sessions WHERE source = 'a2a' AND started_at >= ? ORDER BY started_at DESC LIMIT 1",
                    (start - 2.0,), "A2A: could not find latest forwarded session")):
                self._profile_sessions[key] = session_id
                _state_db(profile, "UPDATE sessions SET title = ? WHERE id = ?", (session_title, session_id),
                          "A2A: could not title forwarded session", commit=True)
            return security.redact_outbound((proc.stdout or "").strip()), protocol.STATE_COMPLETED

    def _complete_task(self, task_id: str, state: str, reply: str, context_id: Optional[str] = None) -> bool:
        """The single terminal transition for a dispatched task, whoever observed its end (a final
        send, the processing-complete hook, a cancel): redact, detect the [INPUT_REQUIRED] marker,
        store the FULL reply, persist/audit/count, push, fire the ``closed`` lifecycle phase.
        Idempotent; False when the task is unknown, already terminal, or not in ``context_id``.
        Blocking (push callback): call it off the event loop."""
        rec = self.tasks.get(task_id)
        if not rec or (context_id is not None and rec["context_id"] != context_id):
            return False
        reply = security.redact_outbound(reply or "")
        stripped = reply.lstrip()
        if state == protocol.STATE_COMPLETED and stripped.upper().startswith(protocol.INPUT_REQUIRED_MARKER):
            state, reply = protocol.STATE_INPUT_REQUIRED, stripped[len(protocol.INPUT_REQUIRED_MARKER):].strip()
        done = self.tasks.complete(task_id, state, reply)
        if done is None:
            return False
        with self._pending_lock:
            if self._executing.get(done.get("session_key") or "") == task_id:
                self._executing.pop(done["session_key"], None)
        protocol.persist_message(done["context_id"], "agent", reply, task_id)
        security.audit("outbound", done["peer"], task_id, reply)
        m = protocol.metrics
        if state in (protocol.STATE_COMPLETED, protocol.STATE_INPUT_REQUIRED):
            m.outbound_total, m.tasks_completed = m.outbound_total + 1, m.tasks_completed + 1
            m.record_latency(time.time() - done["created_at"])
        else:
            m.tasks_failed += 1
        self._send_push_notification(task_id, done["context_id"], reply, state)
        self._lifecycle("closed", done)
        return True

    def _await_future(self, fut: Future, deadline: Optional[float], keepalive, on_timeout):
        """Block until ``fut`` resolves (-> ``(state, reply)``), ``deadline`` passes, the client
        goes away (``keepalive`` raises) or the adapter stops (-> ``on_timeout``). ``deadline=None``
        waits for the terminal state: the work has no wall-clock cap."""
        while True:
            try:
                wait = _SSE_KEEPALIVE if deadline is None else max(0.0, min(_SSE_KEEPALIVE, deadline - time.time()))
                return fut.result(timeout=wait)
            except FuturesTimeout:
                if self._stopping.is_set() or (deadline is not None and time.time() >= deadline):
                    return on_timeout
                if keepalive:
                    try:
                        keepalive()
                    except Exception:
                        return on_timeout
            except Exception:
                return on_timeout

    def _await_reply(self, pending: dict, keepalive=None, *, bounded: bool = True):
        """Observe a dispatched task's outcome: ``(state, reply)`` or None when the bounded wait
        (A2A_REPLY_TIMEOUT for message/send) elapsed — the task keeps running either way."""
        deadline = pending["started"] + _reply_timeout() if bounded else None
        return self._await_future(pending["future"], deadline, keepalive, None)

    def _current_task(self, pending: dict) -> dict:
        rec = self.tasks.get(pending["task_id"])
        return protocol.TaskStore.to_task(rec) if rec else protocol.build_task(
            pending["task_id"], pending["context_id"], protocol.STATE_WORKING, created_at=pending["created_iso"])

    def _rpc_message_send(self, req_id: Any, params: dict, peer: str, agent: Optional[dict] = None, v1_response: bool = False) -> dict:
        """Blocking by default (bounded long-poll, then the task as it stands); v1.0
        ``configuration.returnImmediately`` answers with the WORKING task at once."""
        task, pending = self._prepare_task(params, peer, agent=agent)
        if task is None:
            try:
                if not protocol.extract_return_immediately(params):
                    self._await_reply(pending)
            finally:
                self._release_task(pending["task_id"])
            task = self._current_task(pending)
        return _ok(req_id, protocol.send_message_response(task) if v1_response else task)

    @staticmethod
    def _sse_headers(handler) -> None:
        handler.send_response(200)
        for k, v in (("Content-Type", "text/event-stream"), ("Cache-Control", "no-cache")):
            handler.send_header(k, v)
        handler.end_headers()
        handler.close_connection = True  # v1.0: stream closure signals the terminal state

    @staticmethod
    def _sse_write(handler, chunk: str) -> None:
        handler.wfile.write(chunk.encode("utf-8"))
        handler.wfile.flush()

    @classmethod
    def _keepalive(cls, handler):
        return lambda: cls._sse_write(handler, ": keepalive\n\n")

    def _emit_terminal(self, handler, task_id: str, context_id: str, state: str, reply: str, req_id: Any = None) -> None:
        """Emit the final artifact/status events and the closure marker. ``req_id`` threads into the
        JSON-RPC SSE envelope (§9.4)."""
        completed = bool(reply) and state == protocol.STATE_COMPLETED
        events = ([protocol.artifact_update(task_id, context_id, reply)] if completed else []) + [
            protocol.status_update(task_id, context_id, state, "" if completed else reply)]
        for ev in events:
            self._sse_write(handler, protocol.sse_data(ev, req_id))
        self._sse_write(handler, protocol.sse_done())

    def _rpc_message_stream(self, handler, req_id: Any, params: dict, peer: str, agent: Optional[dict] = None) -> None:
        """message/stream as an SSE response of JSON-RPC-wrapped StreamResponse events (§9.4)."""
        protocol.metrics.streams_started += 1
        self._sse_headers(handler)
        pending = None
        try:
            terminal, pending = self._prepare_task(params, peer, agent=agent)
            if terminal is not None:
                return self._emit_terminal(handler, terminal["id"], terminal["contextId"], terminal["status"]["state"],
                                           protocol.extract_text(terminal.get("status", {}).get("message", {}) or {}), req_id=req_id)
            task_id, context_id = pending["task_id"], pending["context_id"]
            submitted = protocol.build_task(task_id, context_id, protocol.STATE_SUBMITTED, created_at=pending["created_iso"])
            self._sse_write(handler, protocol.sse_data(protocol.stream_task(submitted), req_id))
            self._sse_write(handler, protocol.sse_data(protocol.status_update(task_id, context_id, protocol.STATE_WORKING), req_id))
            # Streams follow the task to its terminal state (no wall-clock cap); only an adapter stop
            # closes one early, on the task's current (non-terminal) state.
            outcome = self._await_reply(pending, keepalive=self._keepalive(handler), bounded=False) or (protocol.STATE_WORKING, "")
            self._emit_terminal(handler, task_id, context_id, *outcome, req_id=req_id)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("A2A: stream client disconnected; its task keeps running")
        finally:
            if pending is not None:
                self._release_task(pending["task_id"])

    def _rpc_tasks_subscribe(self, handler, req_id: Any, params: dict, agent: Optional[dict] = None, peer: str = "") -> None:
        """Reconnect to an existing task's stream (v1.0 SubscribeToTask)."""
        task_id, rec, error = self._find_task(req_id, params, agent, peer)
        if error:
            return handler._json(200, error)
        self._sse_headers(handler)
        try:
            if (fut := self.tasks.watch(task_id, *self._scope(agent, peer))) is None:
                return self._sse_write(handler, protocol.sse_done())
            state, reply = self._await_future(fut, None, self._keepalive(handler), None) or (rec["state"], rec.get("reply", ""))
            self._emit_terminal(handler, task_id, rec["context_id"], state, reply, req_id=req_id)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("A2A: subscribe client disconnected")

    def _find_task(self, req_id: Any, params: dict, agent: Optional[dict], peer: str = "") -> tuple[str, Optional[dict], Optional[dict]]:
        """(task_id, record, None) for a task visible to ``peer``, else (task_id, None, jsonrpc_error).
        Another peer's task is indistinguishable from a missing one."""
        task_id = str(params.get("taskId") or params.get("id") or "")
        rec = self.tasks.get(task_id, *self._scope(agent, peer))
        return task_id, rec, None if rec else _err(req_id, protocol.ERR_TASK_NOT_FOUND, f"task not found: {task_id}")

    def _rpc_tasks_get(self, req_id: Any, params: dict, agent: Optional[dict] = None, peer: str = "") -> dict:
        _task_id, rec, error = self._find_task(req_id, params, agent, peer)
        return error or _ok(req_id, protocol.TaskStore.to_task(rec))

    def _rpc_tasks_list(self, req_id: Any, params: dict, agent: Optional[dict] = None, peer: str = "") -> dict:
        offset = _to_int(params.get("pageToken") or 0, 0)
        page_size = _to_int(params.get("pageSize") or 50, 50)
        agent_slug, tenant = self._scope_for_agent(agent)
        recs, next_offset, total = self.tasks.list(
            context_id=str(params.get("contextId") or ""), state=str(params.get("status") or params.get("state") or ""),
            page_size=page_size, offset=max(0, offset), agent_slug=agent_slug, tenant=tenant, with_total=True, peer=peer)
        include_artifacts = bool(params.get("includeArtifacts", False))
        return _ok(req_id, {"tasks": [protocol.TaskStore.to_task(r, include_artifacts=include_artifacts) for r in recs],
                            "nextPageToken": str(next_offset) if next_offset else "",
                            "pageSize": max(1, min(page_size, 100)), "totalSize": total})

    def _rpc_tasks_cancel(self, req_id: Any, params: dict, agent: Optional[dict] = None, peer: str = "") -> dict:
        """Cancel exactly this peer's task. The request is recorded durably and announced
        (``cancel_requested`` lifecycle phase) first; if the task's own execution is running, the
        gateway's /stop path interrupts that session (and its background delegations) and we wait
        briefly for the confirmed stop. An unconfirmed stop answers WORKING + metadata.cancelRequested,
        never a claimed CANCELED; a task that is not executing is canceled outright."""
        task_id, rec, error = self._find_task(req_id, params, agent, peer)
        if error:
            return error
        if rec["state"] in protocol.TERMINAL_STATES:
            return _err(req_id, protocol.ERR_TASK_NOT_CANCELABLE, f"task {task_id} already {rec['state']}")
        rec = self.tasks.request_cancel(task_id) or rec
        self._turns.reset(rec["context_id"])
        self._lifecycle("cancel_requested", rec)
        if self._execution_alive(rec) and self._loop is not None:
            confirmed = self.tasks.watch(task_id)
            asyncio.run_coroutine_threadsafe(self._cancel_execution(rec), self._loop)
            self._await_future(confirmed, time.time() + _CANCEL_CONFIRM_SECONDS, None, None)
        else:
            self._drop_queued_event(rec)
            self._complete_task(task_id, protocol.STATE_CANCELED, "")
        rec = self.tasks.get(task_id, *self._scope(agent, peer)) or rec
        return _ok(req_id, protocol.TaskStore.to_task(rec))

    def _drop_queued_event(self, rec: dict) -> None:
        """A canceled task still queued behind a busy session must not run later."""
        session_key = rec.get("session_key") or ""
        queued = self._pending_messages.get(session_key)
        if queued is not None and getattr(queued, "message_id", None) == rec["task_id"]:
            self._pending_messages.pop(session_key, None)

    async def _cancel_execution(self, rec: dict) -> None:
        """Mirror of the /stop bypass for exactly this task's session: interrupt the running turn
        (the runner also interrupts its background delegations), cancel the processing task —
        which confirms the stop through on_processing_complete(CANCELLED) — then drain follow-ups."""
        session_key = rec.get("session_key") or ""
        with self._pending_lock:
            owns = self._executing.get(session_key) == rec["task_id"]
        if not session_key or not owns:
            return
        command_guard = asyncio.Event()
        current_guard = self._active_sessions.get(session_key)
        self._active_sessions[session_key] = command_guard
        try:
            interrupt = getattr(self.gateway_runner, "_interrupt_and_clear_session", None)
            if callable(interrupt):
                await interrupt(session_key, self._source_for_record(rec),
                                interrupt_reason=_INTERRUPT_REASON_CANCEL, invalidation_reason=_INTERRUPT_REASON_CANCEL)
            await self.cancel_session_processing(session_key, release_guard=False, discard_pending=False)
        except Exception:
            logger.warning("A2A: cancel of task %s could not interrupt session %s", rec["task_id"], session_key, exc_info=True)
            if self._active_sessions.get(session_key) is command_guard:
                if session_key in self._session_tasks and current_guard is not None:
                    self._active_sessions[session_key] = current_guard
                else:
                    self._release_session_guard(session_key, guard=command_guard)
            return
        await self._drain_pending_after_session_command(session_key, command_guard)

    def _register_inline_push(self, task_id: str, params: dict, agent: Optional[dict] = None, peer: str = "") -> None:
        """v1.0: message/send can carry configuration.taskPushNotificationConfig."""
        cfg = (params.get("configuration") or {}).get("taskPushNotificationConfig") or {}
        url = (cfg.get("url") or (cfg.get("pushNotificationConfig") or {}).get("url") or "") if isinstance(cfg, dict) else ""
        if url:
            self.tasks.set_push_config(task_id, str(url), *self._scope(agent, peer))

    def _rpc_push_config_create(self, req_id: Any, params: dict, agent: Optional[dict] = None, peer: str = "") -> dict:
        task_id = str(params.get("taskId") or "")
        url = str((params.get("pushNotificationConfig") or params.get("config") or {}).get("url") or "")
        if not task_id or not url:
            return _err(req_id, protocol.ERR_INVALID_PARAMS, "taskId and pushNotificationConfig.url required")
        stored = self.tasks.set_push_config(task_id, url, *self._scope(agent, peer))
        return _ok(req_id, stored) if stored is not None else _err(req_id, protocol.ERR_TASK_NOT_FOUND, f"task not found: {task_id}")

    def _push_config_op(self, req_id: Any, params: dict, agent: Optional[dict], peer: str, op, render) -> dict:
        """Shared get/list/delete: ``op(task_id, config_id, slug, tenant, peer)`` falsy => not found."""
        task_id = str(params.get("taskId") or "")
        if not task_id:
            return _err(req_id, protocol.ERR_INVALID_PARAMS, "taskId required")
        found = op(task_id, str(params.get("id") or params.get("configId") or ""), *self._scope(agent, peer))
        return _ok(req_id, render(found)) if found else _err(req_id, protocol.ERR_TASK_NOT_FOUND, f"push config not found for task: {task_id}")

    def _rpc_push_config_get(self, req_id: Any, params: dict, agent: Optional[dict] = None, peer: str = "") -> dict:
        return self._push_config_op(req_id, params, agent, peer, self.tasks.get_push_config, lambda cfg: cfg)

    def _rpc_push_config_list(self, req_id: Any, params: dict, agent: Optional[dict] = None, peer: str = "") -> dict:
        # An empty list is a valid (non-error) result, hence the ``or [[]]`` sentinel through the shared op.
        return self._push_config_op(req_id, params, agent, peer, lambda tid, _cid, *scope: self.tasks.list_push_configs(tid, *scope) or [[]],
                                    lambda found: {"configs": [c for c in found if c], "nextPageToken": ""})

    def _rpc_push_config_delete(self, req_id: Any, params: dict, agent: Optional[dict] = None, peer: str = "") -> dict:
        return self._push_config_op(req_id, params, agent, peer, self.tasks.delete_push_config, lambda _: {"deleted": True})

    def _send_push_notification(self, task_id: str, context_id: str, reply: str, state: str) -> None:
        """POST a v1.0 StreamResponse payload to the task's registered callback (SSRF-checked URL)."""
        def fail(msg: str, *args) -> None:
            protocol.metrics.push_failed += 1
            logger.warning("A2A: push notification for task %s " + msg, task_id, *args)

        callback_url = self.tasks.pop_push_url(task_id)
        if not callback_url:
            return
        if not security.is_safe_callback_url(callback_url, localhost_mode=self._security_context.localhost_only()):
            return fail("blocked — unsafe callback URL: %s", callback_url)
        # A push carries a PREVIEW; the authoritative full result is the stored Task (tasks/get).
        payload = protocol.status_update(task_id, context_id, state, (reply or "")[:_PUSH_PREVIEW_CHARS])
        headers = {"Content-Type": "application/json"}
        if signature := self._security_context.sign_push_payload(payload):
            headers["X-A2A-Signature"] = signature
        try:
            req = urllib.request.Request(callback_url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                status = resp.status
        except Exception as e:
            return fail("failed: %s", e)
        if not 200 <= status < 300:
            return fail("got HTTP %d", status)
        protocol.metrics.push_sent += 1
        logger.debug("A2A: push notification sent for task %s", task_id)

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Fulfil exactly the pending reply Future of the task a final send names: ``reply_to``
        (the gateway's reply anchor is the inbound event's message_id, i.e. the task id; the
        streaming path carries it as ``metadata['reply_to_message_id']``), which must belong to
        ``chat_id`` (the A2A context id). Only sends carrying ``metadata['notify']`` (the base
        adapter's final-reply marker) satisfy a waiter; progress/preview sends, sends naming no
        task (out-of-band notifications into the context) and late output for a task that was
        already canceled or settled never touch another task's waiter."""
        meta = metadata or {}
        task_id = str(reply_to or meta.get("reply_to_message_id") or "")
        if not meta.get("notify"):
            logger.debug("A2A: ignoring non-final send for context %s", chat_id)
        elif not task_id:
            logger.debug("A2A: final send for context %s names no task; not settling any task", chat_id)
        elif not await asyncio.to_thread(self._complete_task, task_id, protocol.STATE_COMPLETED, content or "", chat_id):
            logger.debug("A2A: final send for task %s (context %s) found no open task", task_id, chat_id)  # late / canceled
        return SendResult(success=True, message_id=str(int(time.time() * 1000)))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": f"a2a:{chat_id}", "type": "dm"}

    async def handle_message(self, event: MessageEvent) -> None:
        """Inbound tasks flow through the base pipeline unchanged. The gateway's restart resume
        (an internal, empty-text event with no message_id) is re-attached to the task whose
        execution the crash interrupted — after the ``recover`` lifecycle phase authorized it."""
        if getattr(event, "internal", False) and not (event.text or "") and not getattr(event, "message_id", None) \
                and event.source is not None and not await asyncio.to_thread(self._attach_recovery, event):
            return  # denied or canceled: nothing resumes (the caller releases its pre-claimed slot)
        await super().handle_message(event)

    def _attach_recovery(self, event: MessageEvent) -> bool:
        """Bind a resume event to the interrupted task of its session (bumping the local execution
        epoch), fail the queued tasks that died with the process, and refuse when the ``recover``
        phase denies or the task was canceled meanwhile. True = let the event run."""
        session_key = self._event_session_key(event)
        with self._pending_lock:
            if session_key in self._executing:
                return True  # a live turn: this is an ordinary internal wake, not a recovery
        records = self.tasks.in_flight(session_key)
        started = [r for r in records if r.get("execution_started_at")]
        target = max(started, key=lambda r: r["execution_started_at"]) if started else None
        for rec in records:
            if rec is not target:
                self._complete_task(rec["task_id"], protocol.STATE_FAILED, "[task lost on gateway restart]")
        if target is None:
            return True
        if target.get("cancel_requested_at"):
            self._complete_task(target["task_id"], protocol.STATE_CANCELED, "")
            return False
        if denial := self._lifecycle("recover", target):
            logger.warning("A2A: recovery of task %s denied: %s", target["task_id"], denial)
            self._complete_task(target["task_id"], protocol.STATE_FAILED, f"[recovery denied: {denial}]")
            self._clear_resume_pending(session_key)
            return False
        epoch = self.tasks.begin_recovery(target["task_id"])
        event.message_id = target["task_id"]
        event.metadata.update(protocol.inbound_event_metadata(
            peer=target["peer"], task_id=target["task_id"], context_id=target["context_id"], message_id=target.get("message_id", ""),
            agent_slug=target.get("agent_slug", ""), tenant=target.get("tenant", ""), request_metadata=target.get("request_metadata"),
            execution_epoch=epoch, recovery={"epoch": epoch, "reason": "restart"}))
        logger.info("A2A: resuming task %s (execution epoch %d)", target["task_id"], epoch)
        return True

    def _clear_resume_pending(self, session_key: str) -> None:
        clear = getattr(getattr(self.gateway_runner, "session_store", None), "clear_resume_pending", None)
        if callable(clear):
            with contextlib.suppress(Exception):
                clear(session_key)

    async def on_processing_start(self, event: MessageEvent) -> None:
        """The gateway began (or resumed) this task's turn: remember which task owns the session's
        execution and stamp it durably (the restart recovery keys on it)."""
        task_id = str(getattr(event, "message_id", "") or "")
        if task_id and self.tasks.get(task_id):
            with self._pending_lock:
                self._executing[self._event_session_key(event)] = task_id
            await asyncio.to_thread(self.tasks.mark_execution_started, task_id)

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        """Settle the exact task when its turn ends without a final send (failures, confirmed
        cancellations, empty runs, or a reply the gateway already streamed to the user); a task
        already completed by its final send is untouched."""
        task_id = str(getattr(event, "message_id", "") or "")
        if task_id:
            # A streamed turn never calls send() with notify=True (the gateway suppresses the
            # normal final send once streaming delivered the body), so the SUCCESS default must
            # not resolve with "" — that strands every A2A streaming reply as an empty completed
            # task (#116944). _streamed_final_response is the same stash _final_text_for_post_turn_hooks
            # reads for /goal and /loop.
            _streamed = getattr(event, "_streamed_final_response", "")
            default = (protocol.STATE_COMPLETED, _streamed if isinstance(_streamed, str) else "")
            state, text = {
                ProcessingOutcome.FAILURE: (protocol.STATE_FAILED, "[agent processing failed]"),
                ProcessingOutcome.CANCELLED: (protocol.STATE_CANCELED, ""),
            }.get(outcome, default)
            await asyncio.to_thread(self._complete_task, task_id, state, text)
