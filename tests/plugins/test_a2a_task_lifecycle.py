"""Durable A2A task lifecycle (AI-3207): records and full output survive a restart, execution is
independent of the HTTP request (returnImmediately, long-poll timeout, disconnect), cancellation
targets the exact owning execution, and a restart-interrupted task is re-attached to the gateway's
resume turn only after the ``a2a_task_lifecycle`` ``recover`` phase authorized it.

Before this change the task store was in-memory only, a blocking request that timed out or
disconnected FAILED the task while the agent kept working, and a result produced after the request
had gone was lost.
"""
from __future__ import annotations

import asyncio
import json
import socket
import time
import urllib.request

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from plugins.platforms.a2a import adapter as adapter_mod, protocol
from plugins.platforms.a2a.adapter import A2AAdapter


def _adapter(store=":memory:"):
    return A2AAdapter(PlatformConfig(enabled=True, extra={"task_store": str(store)}))


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _rpc(url, method, params):
    body = {"jsonrpc": "2.0", "id": "1", "method": method, "params": params}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


def _send_params(text, ctx="", **configuration):
    params = {"message": protocol.text_message(protocol.ROLE_USER, text, context_id=ctx)}
    if configuration:
        params["configuration"] = configuration
    return params


def _source(adapter, ctx="ctx-r", peer="peer"):
    return adapter.build_source(chat_id=ctx, chat_name=f"a2a:{peer}", chat_type="dm", user_id=peer, user_name=peer)


def _dispatch(adapter, task_id, ctx="ctx-r", peer="peer", started=True):
    """A task the gateway has (started=True) or has not yet begun executing, with durable linkage."""
    event = MessageEvent(text="x", message_type=MessageType.TEXT, message_id=task_id, source=_source(adapter, ctx, peer))
    adapter.tasks.create(task_id, ctx, peer)
    adapter.tasks.set_dispatch(task_id, session_key=adapter._event_session_key(event), request_metadata={"locator": "d-1"}, message_id="m-1")
    adapter.tasks.set_state(task_id, protocol.STATE_WORKING)
    if started:
        asyncio.run(adapter.on_processing_start(event))
    return event


def _resume_event(adapter, ctx="ctx-r", peer="peer"):
    """The gateway's startup auto-resume: internal, empty text, no message_id."""
    return MessageEvent(text="", message_type=MessageType.TEXT, source=_source(adapter, ctx, peer), internal=True)


@pytest.fixture
def hooks(monkeypatch):
    """Install ``a2a_task_lifecycle`` callbacks; records (phase, task) calls."""
    calls: list[tuple[str, dict]] = []
    handlers: list = []

    def install(*callbacks):
        handlers[:] = list(callbacks)

    def recorder(phase, task):
        calls.append((phase, task))

    handlers.append(recorder)
    import hermes_cli.plugins as plugins_mod
    monkeypatch.setattr(plugins_mod, "iter_hook_callbacks", lambda name: tuple(handlers) if name == protocol.LIFECYCLE_HOOK else ())
    install.calls = calls  # type: ignore[attr-defined]
    install.recorder = recorder  # type: ignore[attr-defined]
    return install


# ---------------------------------------------------------------------------
# Durable store
# ---------------------------------------------------------------------------

class TestDurableStore:
    def test_records_and_full_output_survive_a_new_process(self, tmp_path):
        path = tmp_path / "a2a_tasks.json"
        first = protocol.TaskStore(path)
        first.create("t1", "c1", "alice")
        first.set_dispatch("t1", session_key="k1", request_metadata={"locator": "d-1"}, message_id="m-1")
        first.mark_execution_started("t1")
        long_reply = "x" * (adapter_mod._PUSH_PREVIEW_CHARS * 3)
        first.complete("t1", protocol.STATE_COMPLETED, long_reply)

        second = protocol.TaskStore(path)  # "after restart"
        rec = second.get("t1", peer="alice")
        assert rec["state"] == protocol.STATE_COMPLETED
        assert rec["reply"] == long_reply  # the stored result is never the push preview
        assert protocol.extract_text(protocol.TaskStore.to_task(rec)["artifacts"][0]) == long_reply
        assert rec["session_key"] == "k1" and rec["request_metadata"] == {"locator": "d-1"} and rec["execution_epoch"] == 1

    def test_older_shape_record_loads_with_defaults(self, tmp_path):
        path = tmp_path / "a2a_tasks.json"
        legacy = {"task_id": "t-old", "context_id": "c", "peer": "p", "agent_slug": "", "tenant": "",
                  "state": protocol.STATE_WORKING, "reply": "", "created_at": 1.0, "created_iso": "2026-01-01T00:00:00.000Z",
                  "push_url": "", "push_config_id": ""}
        path.write_text(json.dumps({"version": 1, "tasks": [legacy]}), encoding="utf-8")
        store = protocol.TaskStore(path)
        rec = store.get("t-old")
        assert not store.unavailable
        assert rec["execution_epoch"] == 1 and rec["request_metadata"] == {} and rec["session_key"] == ""
        assert rec["cancel_requested_at"] is None and rec["execution_started_at"] is None and rec["updated_at"] == 1.0

    @pytest.mark.parametrize("content, needle", [
        ("{not json", "corrupt"),
        (json.dumps({"version": 1, "tasks": "nope"}), "corrupt"),
        (json.dumps({"version": 2, "tasks": []}), "version 2"),
    ])
    def test_unreadable_or_newer_store_fails_closed(self, tmp_path, content, needle):
        path = tmp_path / "a2a_tasks.json"
        path.write_text(content, encoding="utf-8")
        store = protocol.TaskStore(path)
        assert needle in store.unavailable
        with pytest.raises(protocol.TaskStoreUnavailable):
            store.create("t", "c", "p")
        adapter = _adapter(path)
        adapter._loop, adapter._message_handler = object(), object()
        task, pending = adapter._prepare_task(_send_params("hi"), "peer")
        assert pending is None and task["status"]["state"] == protocol.STATE_FAILED
        assert "Task store unavailable" in protocol.extract_text(task["status"]["message"])
        assert adapter.tasks.get(task["id"]) is None
        assert path.read_text(encoding="utf-8") == content  # never overwritten with an empty store

    def test_missing_file_is_a_fresh_install(self, tmp_path):
        store = protocol.TaskStore(tmp_path / "missing" / "a2a_tasks.json")
        assert not store.unavailable
        store.create("t", "c", "p")
        assert (tmp_path / "missing" / "a2a_tasks.json").exists()


# ---------------------------------------------------------------------------
# Execution independent of the HTTP request
# ---------------------------------------------------------------------------

class TestDetachedExecution:
    def test_return_immediately_and_long_poll_timeout_keep_the_task_working(self, monkeypatch, tmp_path):
        monkeypatch.setenv("A2A_PORT", str(_free_port()))
        monkeypatch.setenv("A2A_REPLY_TIMEOUT", "1")
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        adapter = _adapter(tmp_path / "a2a_tasks.json")
        long_reply = "R" * (adapter_mod._PUSH_PREVIEW_CHARS * 2)

        async def slow_handler(event):
            await asyncio.sleep(1.6)  # longer than the long-poll bound
            await adapter.send(event.source.chat_id, long_reply, reply_to=event.message_id, metadata={"notify": True})

        adapter.handle_message = slow_handler  # type: ignore[method-assign]
        adapter._message_handler = object()
        base = f"http://127.0.0.1:{adapter.port}/"

        async def poll_until_terminal(task_id):
            for _ in range(60):
                rec = adapter.tasks.get(task_id)
                if rec["state"] in protocol.TERMINAL_STATES:
                    return rec
                await asyncio.sleep(0.1)
            raise AssertionError("task never completed")

        async def run():
            assert await adapter.connect() is True
            t0 = time.time()
            resp = await asyncio.to_thread(_rpc, base, "message/send", _send_params("go", returnImmediately=True))
            assert time.time() - t0 < 1.0
            task = protocol.unwrap_send_message_response(resp["result"])
            assert task["status"]["state"] == protocol.STATE_WORKING
            assert adapter.tasks.get(task["id"])["session_key"]  # durable linkage recorded at dispatch
            rec = await poll_until_terminal(task["id"])
            assert rec["state"] == protocol.STATE_COMPLETED and rec["reply"] == long_reply
            got = await asyncio.to_thread(_rpc, base, "tasks/get", {"taskId": task["id"]})
            assert protocol.extract_text(got["result"]["artifacts"][0]) == long_reply

            # Blocking send: the long-poll bound elapses, the task is returned WORKING — never failed.
            resp = await asyncio.to_thread(_rpc, base, "message/send", _send_params("again"))
            task = protocol.unwrap_send_message_response(resp["result"])
            assert task["status"]["state"] == protocol.STATE_WORKING
            rec = await poll_until_terminal(task["id"])
            assert rec["state"] == protocol.STATE_COMPLETED
            await adapter.disconnect()

        asyncio.run(run())
        assert protocol.TaskStore(tmp_path / "a2a_tasks.json").list(state=protocol.STATE_COMPLETED)[0].__len__() == 2

    def test_result_after_the_request_left_is_still_recorded(self):
        """No waiter, no request: the final send still lands in the durable record."""
        adapter = _adapter()
        _dispatch(adapter, "t-late")
        assert adapter._active_tasks == set()
        asyncio.run(adapter.send("ctx-r", "final", reply_to="t-late", metadata={"notify": True}))
        assert adapter.tasks.get("t-late")["state"] == protocol.STATE_COMPLETED
        assert adapter.tasks.get("t-late")["reply"] == "final"


# ---------------------------------------------------------------------------
# Inactivity control (not a wall-clock cap)
# ---------------------------------------------------------------------------

class TestInactivityControl:
    def test_watchdog_spares_live_executions_and_fails_dead_ones(self, monkeypatch):
        adapter = _adapter()
        _dispatch(adapter, "t-running")          # in adapter._executing
        _dispatch(adapter, "t-gateway", ctx="ctx-g", started=False)
        _dispatch(adapter, "t-dead", ctx="ctx-d", started=False)
        for tid in ("t-running", "t-gateway", "t-dead"):
            adapter.tasks._tasks[tid]["updated_at"] = time.time() - 100_000
        gateway_key = adapter.tasks.get("t-gateway")["session_key"]

        class Runner:
            def _is_session_running(self, key):
                return key == gateway_key

        adapter.gateway_runner = Runner()
        assert adapter._fail_orphans_once() == ["t-dead"]
        assert adapter.tasks.get("t-running")["state"] == protocol.STATE_WORKING
        assert adapter.tasks.get("t-gateway")["state"] == protocol.STATE_WORKING


# ---------------------------------------------------------------------------
# Restart recovery
# ---------------------------------------------------------------------------

class TestRestartRecovery:
    def _restarted(self, tmp_path, monkeypatch, seen):
        """Crash after execution started; a fresh adapter on the same store, base dispatch stubbed."""
        path = tmp_path / "a2a_tasks.json"
        crashed = _adapter(path)
        _dispatch(crashed, "t-a")
        _dispatch(crashed, "t-queued", started=False)   # queued behind t-a, never started
        del crashed
        revived = _adapter(path)

        async def fake_base(self, event):
            seen.append(event)

        monkeypatch.setattr(BasePlatformAdapter, "handle_message", fake_base)
        return revived

    def test_resume_reattaches_the_interrupted_task_after_the_recover_phase(self, tmp_path, monkeypatch, hooks):
        seen: list = []
        revived = self._restarted(tmp_path, monkeypatch, seen)
        event = _resume_event(revived)
        asyncio.run(revived.handle_message(event))

        (dispatched,) = seen
        assert dispatched is event and event.message_id == "t-a"
        ns = event.metadata[protocol.EVENT_METADATA_KEY]
        assert ns["peer"] == "peer" and ns["request_metadata"] == {"locator": "d-1"}
        assert ns["execution_epoch"] == 2 and ns["recovery"] == {"epoch": 2, "reason": "restart"}
        assert revived.tasks.get("t-a")["execution_epoch"] == 2
        assert [phase for phase, _ in hooks.calls if phase == "recover"] == ["recover"]
        recover_task = next(task for phase, task in hooks.calls if phase == "recover")
        assert recover_task["task_id"] == "t-a" and recover_task["execution_epoch"] == 1
        # The queued task died with the process: failed, not silently forgotten.
        assert revived.tasks.get("t-queued")["state"] == protocol.STATE_FAILED
        assert "lost on gateway restart" in revived.tasks.get("t-queued")["reply"]
        # The resumed turn's final output lands on the original task.
        asyncio.run(revived.send("ctx-r", "resumed answer", reply_to="t-a", metadata={"notify": True}))
        assert protocol.TaskStore(tmp_path / "a2a_tasks.json").get("t-a")["reply"] == "resumed answer"

    @pytest.mark.parametrize("deny", [
        lambda phase, task: {"action": "deny", "reason": "bootstrap refused"} if phase == "recover" else None,
        lambda phase, task: (_ for _ in ()).throw(RuntimeError("bootstrap refused")),
    ])
    def test_denied_recovery_fails_the_task_and_runs_nothing(self, tmp_path, monkeypatch, hooks, deny):
        seen: list = []
        revived = self._restarted(tmp_path, monkeypatch, seen)
        hooks(hooks.recorder, deny)
        cleared: list = []

        class Runner:
            class session_store:  # noqa: N801 — attribute shape of the gateway runner
                @staticmethod
                def clear_resume_pending(key):
                    cleared.append(key)

        revived.gateway_runner = Runner()
        asyncio.run(revived.handle_message(_resume_event(revived)))
        assert seen == []  # no model work
        rec = revived.tasks.get("t-a")
        assert rec["state"] == protocol.STATE_FAILED and "bootstrap refused" in rec["reply"]
        assert cleared == [rec["session_key"]]
        assert rec["execution_epoch"] == 1

    def test_task_canceled_before_restart_is_not_resumed(self, tmp_path, monkeypatch, hooks):
        seen: list = []
        revived = self._restarted(tmp_path, monkeypatch, seen)
        revived.tasks.request_cancel("t-a")
        asyncio.run(revived.handle_message(_resume_event(revived)))
        assert seen == []
        assert revived.tasks.get("t-a")["state"] == protocol.STATE_CANCELED
        assert not [p for p, _ in hooks.calls if p == "recover"]

    def test_internal_wake_during_a_live_turn_is_not_a_recovery(self, monkeypatch, hooks):
        adapter = _adapter()
        _dispatch(adapter, "t-live")
        seen: list = []

        async def fake_base(self, event):
            seen.append(event)

        monkeypatch.setattr(BasePlatformAdapter, "handle_message", fake_base)
        event = _resume_event(adapter)
        asyncio.run(adapter.handle_message(event))
        assert seen == [event] and event.message_id is None and adapter.tasks.get("t-live")["execution_epoch"] == 1


# ---------------------------------------------------------------------------
# Exact owner cancellation
# ---------------------------------------------------------------------------

class TestExactOwnerCancel:
    def _wire(self, adapter, monkeypatch, *, confirm: bool):
        interrupts: list = []

        class Runner:
            async def _interrupt_and_clear_session(self, session_key, source, **kw):
                interrupts.append((session_key, source.chat_id, kw["interrupt_reason"]))

            def _is_session_running(self, key):
                return False

        async def cancel_session_processing(session_key, **kw):
            if confirm:  # the gateway's cancelled turn reports back through the lifecycle hook
                await adapter.on_processing_complete(adapter._events[session_key], ProcessingOutcome.CANCELLED)

        async def drain(*a, **k):
            return None

        adapter.gateway_runner = Runner()
        monkeypatch.setattr(adapter, "cancel_session_processing", cancel_session_processing)
        monkeypatch.setattr(adapter, "_drain_pending_after_session_command", drain)
        monkeypatch.setattr(adapter_mod, "_CANCEL_CONFIRM_SECONDS", 0.3)
        return interrupts

    def test_cancel_interrupts_exactly_the_owning_execution_and_confirms(self, monkeypatch, hooks):
        adapter = _adapter()
        ev_a = _dispatch(adapter, "t-a")
        ev_b = _dispatch(adapter, "t-b", ctx="ctx-b")
        adapter._events = {adapter.tasks.get("t-a")["session_key"]: ev_a, adapter.tasks.get("t-b")["session_key"]: ev_b}
        interrupts = self._wire(adapter, monkeypatch, confirm=True)

        async def run():
            adapter._loop = asyncio.get_running_loop()
            return await asyncio.to_thread(adapter._rpc_tasks_cancel, 1, {"taskId": "t-a"}, peer="peer")

        resp = asyncio.run(run())
        assert resp["result"]["status"]["state"] == protocol.STATE_CANCELED
        assert interrupts == [(adapter.tasks.get("t-a")["session_key"], "ctx-r", adapter_mod._INTERRUPT_REASON_CANCEL)]
        assert adapter.tasks.get("t-b")["state"] == protocol.STATE_WORKING  # the other execution is untouched
        assert [p for p, t in hooks.calls if t["task_id"] == "t-a"] == ["cancel_requested", "closed"]

    def test_unconfirmed_stop_is_reported_as_a_request_not_a_cancel(self, monkeypatch, hooks):
        adapter = _adapter()
        ev = _dispatch(adapter, "t-a")
        adapter._events = {adapter.tasks.get("t-a")["session_key"]: ev}
        self._wire(adapter, monkeypatch, confirm=False)

        async def run():
            adapter._loop = asyncio.get_running_loop()
            resp = await asyncio.to_thread(adapter._rpc_tasks_cancel, 1, {"taskId": "t-a"}, peer="peer")
            assert resp["result"]["status"]["state"] == protocol.STATE_WORKING
            assert resp["result"]["metadata"] == {"cancelRequested": True}
            assert adapter.tasks.get("t-a")["cancel_requested_at"]
            # The stop is confirmed later by the turn itself.
            await adapter.on_processing_complete(ev, ProcessingOutcome.CANCELLED)

        asyncio.run(run())
        assert adapter.tasks.get("t-a")["state"] == protocol.STATE_CANCELED

    def test_cancel_of_a_queued_task_drops_its_event(self, hooks):
        adapter = _adapter()
        ev = _dispatch(adapter, "t-q", started=False)
        session_key = adapter.tasks.get("t-q")["session_key"]
        adapter._pending_messages[session_key] = ev
        resp = adapter._rpc_tasks_cancel(1, {"taskId": "t-q"}, peer="peer")
        assert resp["result"]["status"]["state"] == protocol.STATE_CANCELED
        assert session_key not in adapter._pending_messages
