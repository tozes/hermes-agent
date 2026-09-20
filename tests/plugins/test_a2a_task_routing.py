"""Exact-task routing, peer task ownership and the namespaced inbound event metadata of the A2A
plugin (AI-3206).

Before this change ``adapter.send()`` settled the OLDEST pending task of a context, so after
cancelling task A and opening task B in the same context, A's late output completed B
(reproduced in exact-task-probe.json). Task records also carried no peer scope, so any
authenticated peer could read/cancel another peer's task, and the peer's request metadata never
reached the ``MessageEvent``.
"""
from __future__ import annotations

import asyncio
import json
import socket
import urllib.request
from types import SimpleNamespace

from gateway.platforms.event import ProcessingOutcome
from plugins.platforms.a2a import protocol


def _bare_adapter():
    from plugins.platforms.a2a.adapter import A2AAdapter
    from gateway.config import PlatformConfig
    return A2AAdapter(PlatformConfig(enabled=True))


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _live_adapter(monkeypatch, *, peer_tokens: str = "", on_event=None):
    """Live HTTP adapter whose fake handler echoes through the real ``send()`` contract
    (final reply named by the event's message_id)."""
    from plugins.platforms.a2a.adapter import A2AAdapter
    from gateway.config import PlatformConfig

    port = _free_port()
    monkeypatch.setenv("A2A_PORT", str(port))
    monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
    if peer_tokens:
        monkeypatch.setenv("A2A_PEER_TOKENS", peer_tokens)
    else:
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
    adapter = A2AAdapter(PlatformConfig(enabled=True))

    async def fake_handle_message(event):
        if on_event is not None:
            on_event(event)
        await adapter.send(event.source.chat_id, "ECHO: " + event.text, reply_to=event.message_id,
                           metadata={"notify": True})

    adapter.handle_message = fake_handle_message  # type: ignore[method-assign]
    adapter._message_handler = object()
    return adapter, f"http://127.0.0.1:{port}/"


def _rpc(url, method, params, token=""):
    body = {"jsonrpc": "2.0", "id": "1", "method": method, "params": params}
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


def _send_params(text, ctx="", metadata=None):
    msg = protocol.text_message(protocol.ROLE_USER, text, context_id=ctx)
    msg["messageId"] = "peer-msg-1"
    if metadata is not None:
        msg["metadata"] = metadata
    return {"message": msg}


# ---------------------------------------------------------------------------
# Exact-task completion / cancel routing (method level, no HTTP)
# ---------------------------------------------------------------------------

class TestExactTaskRouting:
    def test_late_output_of_canceled_task_never_completes_successor(self):
        """Cancel/pop A, open B in the same context, then A's late final send arrives:
        B must stay pending and A's output is dropped."""
        adapter = _bare_adapter()
        adapter.tasks.create("task-a", "ctx-1", "peer")
        fut_a = adapter._add_pending("task-a", "ctx-1")
        resp = adapter._rpc_tasks_cancel(1, {"taskId": "task-a"}, peer="peer")
        assert resp["result"]["status"]["state"] == protocol.STATE_CANCELED
        assert fut_a.result(timeout=0) == (protocol.STATE_CANCELED, "")
        adapter._pop_pending("task-a")
        fut_b = adapter._add_pending("task-b", "ctx-1")
        try:
            asyncio.run(adapter.send("ctx-1", "late result from A", reply_to="task-a", metadata={"notify": True}))
            assert fut_b.done() is False
            asyncio.run(adapter.send("ctx-1", "B's own answer", reply_to="task-b", metadata={"notify": True}))
            assert fut_b.result(timeout=0) == (protocol.STATE_COMPLETED, "B's own answer")
        finally:
            adapter._pop_pending("task-b")

    def test_notify_send_naming_no_task_settles_nothing(self):
        """An unrelated final send into the context (no reply anchor) cannot settle a task."""
        adapter = _bare_adapter()
        fut = adapter._add_pending("task-x", "ctx-n")
        try:
            asyncio.run(adapter.send("ctx-n", "background notification", metadata={"notify": True}))
            assert fut.done() is False
        finally:
            adapter._pop_pending("task-x")

    def test_send_requires_task_to_belong_to_context(self):
        adapter = _bare_adapter()
        fut = adapter._add_pending("task-x", "ctx-real")
        try:
            asyncio.run(adapter.send("ctx-other", "misrouted", reply_to="task-x", metadata={"notify": True}))
            assert fut.done() is False
        finally:
            adapter._pop_pending("task-x")

    def test_busy_queue_does_not_merge_tasks(self):
        """A is running and B queued behind it on the busy session. The follow-up turn's streamed
        final names B (``reply_to_message_id``): it settles only B — the older, still-pending A
        must not receive B's answer — and A is settled by its own processing-complete hook."""
        adapter = _bare_adapter()
        fut_a = adapter._add_pending("task-a", "ctx-busy")
        fut_b = adapter._add_pending("task-b", "ctx-busy")
        try:
            asyncio.run(adapter.send("ctx-busy", "answer to B", metadata={"notify": True, "reply_to_message_id": "task-b"}))
            assert fut_b.result(timeout=0) == (protocol.STATE_COMPLETED, "answer to B")
            assert fut_a.done() is False
            asyncio.run(adapter.on_processing_complete(SimpleNamespace(message_id="task-a"), ProcessingOutcome.FAILURE))
            assert fut_a.result(timeout=0)[0] == protocol.STATE_FAILED
        finally:
            adapter._pop_pending("task-a")
            adapter._pop_pending("task-b")

    def test_input_required_continuation_routes_to_new_task(self):
        """A ends INPUT_REQUIRED; the peer continues in the same context as B; B's final is B's."""
        adapter = _bare_adapter()
        pending_a = {"task_id": "task-a", "context_id": "ctx-q", "peer": "peer", "future": adapter._add_pending("task-a", "ctx-q"),
                     "created_iso": protocol.now_iso(), "started": 0.0}
        adapter.tasks.create("task-a", "ctx-q", "peer")
        state, reply = adapter._finalize_task(pending_a, protocol.STATE_COMPLETED, "[INPUT_REQUIRED] which file?")
        assert (state, reply) == (protocol.STATE_INPUT_REQUIRED, "which file?")
        adapter.tasks.create("task-b", "ctx-q", "peer")
        fut_b = adapter._add_pending("task-b", "ctx-q")
        try:
            asyncio.run(adapter.send("ctx-q", "the answer", reply_to="task-b", metadata={"notify": True}))
            assert fut_b.result(timeout=0) == (protocol.STATE_COMPLETED, "the answer")
            assert adapter._rpc_tasks_get(1, {"taskId": "task-a"}, peer="peer")["result"]["status"]["state"] == protocol.STATE_INPUT_REQUIRED
        finally:
            adapter._pop_pending("task-b")


# ---------------------------------------------------------------------------
# Peer task ownership
# ---------------------------------------------------------------------------

class TestPeerOwnership:
    def test_other_peer_cannot_query_cancel_or_list(self):
        adapter = _bare_adapter()
        adapter.tasks.create("task-a", "ctx-own", "alice")
        adapter.tasks.set_push_config("task-a", "https://example.com/cb", peer="alice")
        for method in ("_rpc_tasks_get", "_rpc_tasks_cancel", "_rpc_push_config_get", "_rpc_push_config_delete"):
            assert getattr(adapter, method)(1, {"taskId": "task-a"}, peer="bob")["error"]["code"] == protocol.ERR_TASK_NOT_FOUND, method
        assert adapter._rpc_push_config_create(1, {"taskId": "task-a", "pushNotificationConfig": {"url": "https://x/"}}, peer="bob")["error"]["code"] == protocol.ERR_TASK_NOT_FOUND
        assert adapter._rpc_tasks_list(1, {"contextId": "ctx-own"}, peer="bob")["result"]["tasks"] == []
        assert adapter.tasks.watch("task-a", peer="bob") is None
        # The owner still sees everything, and the task was not cancelled by the stranger.
        assert adapter._rpc_tasks_get(1, {"taskId": "task-a"}, peer="alice")["result"]["status"]["state"] == protocol.STATE_SUBMITTED
        assert [t["id"] for t in adapter._rpc_tasks_list(1, {}, peer="alice")["result"]["tasks"]] == ["task-a"]

    def test_cross_peer_chosen_context_is_rejected_without_record(self):
        adapter = _bare_adapter()
        adapter.tasks.create("task-a", "ctx-shared", "alice")
        task, pending = adapter._prepare_task(_send_params("hi", ctx="ctx-shared"), "bob")
        assert pending is None
        assert task["status"]["state"] == protocol.STATE_REJECTED
        assert adapter.tasks.get(task["id"]) is None
        assert adapter.tasks.context_peer("ctx-shared") == "alice"

    def test_live_peer_scoping_over_http(self, monkeypatch):
        adapter, base = _live_adapter(monkeypatch, peer_tokens="alice:tok-a,bob:tok-b")

        async def run():
            assert await adapter.connect() is True
            created = await asyncio.to_thread(_rpc, base, "message/send", _send_params("hello", ctx="ctx-live"), "tok-a")
            task = protocol.unwrap_send_message_response(created["result"])
            assert task["status"]["state"] == protocol.STATE_COMPLETED
            mine = await asyncio.to_thread(_rpc, base, "tasks/get", {"taskId": task["id"]}, "tok-a")
            assert mine["result"]["id"] == task["id"]
            theirs = await asyncio.to_thread(_rpc, base, "tasks/get", {"taskId": task["id"]}, "tok-b")
            assert theirs["error"]["code"] == protocol.ERR_TASK_NOT_FOUND
            joined = await asyncio.to_thread(_rpc, base, "message/send", _send_params("me too", ctx="ctx-live"), "tok-b")
            assert protocol.unwrap_send_message_response(joined["result"])["status"]["state"] == protocol.STATE_REJECTED
            await adapter.disconnect()

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Namespaced inbound event metadata
# ---------------------------------------------------------------------------

class TestInboundEventMetadata:
    def test_event_carries_authenticated_peer_and_bounded_request_metadata(self, monkeypatch):
        seen = []
        adapter, base = _live_adapter(monkeypatch, peer_tokens="alice:tok-a", on_event=seen.append)
        hostile = {"gateway_session_key": "forged", "gateway_session_id": "forged", "a2a": {"peer": "mallory"},
                   "locator": {"delegation": "d-1"}}

        async def run():
            assert await adapter.connect() is True
            resp = await asyncio.to_thread(_rpc, base, "message/send", _send_params("do it", ctx="ctx-meta", metadata=hostile), "tok-a")
            task = protocol.unwrap_send_message_response(resp["result"])
            assert task["status"]["state"] == protocol.STATE_COMPLETED
            await adapter.disconnect()

        asyncio.run(run())
        (event,) = seen
        ns = event.metadata[protocol.EVENT_METADATA_KEY]
        assert ns["peer"] == "alice"  # credential identity, not the body's claim
        assert ns["task_id"] == event.message_id
        assert ns["context_id"] == event.source.chat_id == "ctx-meta"
        assert ns["message_id"] == "peer-msg-1"
        assert ns["request_metadata"] == hostile  # verbatim, as data
        assert set(event.metadata) == {protocol.EVENT_METADATA_KEY}  # nothing flattened to the top level

    def test_invalid_or_oversized_request_metadata_rejects_task(self):
        adapter = _bare_adapter()
        adapter._loop, adapter._message_handler = object(), object()
        for bad in ("not-an-object", {"blob": "x" * (protocol.MAX_REQUEST_METADATA_BYTES + 1)}):
            task, pending = adapter._prepare_task(_send_params("hi", metadata=bad), "alice")
            assert pending is None and task["status"]["state"] == protocol.STATE_REJECTED, bad
            assert adapter.tasks.get(task["id"], peer="alice")["state"] == protocol.STATE_REJECTED
        assert protocol.extract_request_metadata({"message": {"parts": []}}) == {}
        assert protocol.extract_request_metadata({"metadata": {"k": 1}}) == {"k": 1}
