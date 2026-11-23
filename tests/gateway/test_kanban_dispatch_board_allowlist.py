"""Board allowlist (``kanban.dispatch_boards``) and explicitly scoped dispatch-tick
observer profiles (``kanban.dispatch_tick_observer_profiles``) for the gateway-embedded
kanban dispatcher.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent.secret_scope import set_multiplex_active
from gateway import kanban_watchers_dispatcher as kwd
from gateway.kanban_watchers_common import _board_slugs, _run_in_fresh_context
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import plugins as plugins_mod
from hermes_cli.config import _validate_config_key, set_config_value
from hermes_cli.config_defaults import DEFAULT_CONFIG


@pytest.fixture
def isolated_gateway_home(tmp_path, monkeypatch):
    """Isolated root HERMES_HOME with boards ``default``, ``factonio-work`` and ``other-board``."""
    root = tmp_path / "home"
    root.mkdir()
    (root / "config.yaml").write_text("{}\n", encoding="utf-8")
    (root / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(root))
    for pin in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_RUN_ID",
        "HERMES_KANBAN_CLAIM_LOCK",
        "HERMES_DELEGATED_CHILD_CONTEXT",
    ):
        monkeypatch.delenv(pin, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda *, home=None: root)

    # Create profiles remote-worker and ab-coordinator as real profile directories.
    for prof in ("remote-worker", "ab-coordinator", "sibling"):
        pdir = root / "profiles" / prof
        pdir.mkdir(parents=True)
        (pdir / "config.yaml").write_text("{}\n", encoding="utf-8")
        (pdir / ".env").write_text("", encoding="utf-8")

    kb.init_db(board="default")
    kb.create_board("factonio-work")
    kb.create_board("other-board")
    plugins_mod._reset_plugin_managers_for_tests()
    try:
        yield root
    finally:
        set_multiplex_active(False)
        plugins_mod._reset_plugin_managers_for_tests()


def _make_dispatcher() -> kwd._KanbanDispatcher:
    settings = kwd._resolve_dispatcher_settings({}, kb)
    return kwd._KanbanDispatcher(kb, settings)


def _create_ready_card(board: str, *, assignee: str = "remote-worker", title: str = "task") -> str:
    with kbc.connect_closing(board=board) as conn:
        return kb.create_task(conn, title=title, assignee=assignee, board=board)


def _task_status(board: str, task_id: str) -> str:
    with kbc.connect_closing(board=board) as conn:
        return kb.get_task(conn, task_id).status


def test_config_defaults_and_validation_expose_board_and_observer_keys(isolated_gateway_home):
    assert "dispatch_boards" in DEFAULT_CONFIG["kanban"]
    assert DEFAULT_CONFIG["kanban"]["dispatch_boards"] is None
    assert "dispatch_tick_observer_profiles" in DEFAULT_CONFIG["kanban"]
    assert DEFAULT_CONFIG["kanban"]["dispatch_tick_observer_profiles"] is None

    assert _validate_config_key("kanban.dispatch_boards") == (True, None)
    assert _validate_config_key("kanban.dispatch_tick_observer_profiles") == (True, None)

    set_config_value("kanban.dispatch_boards", '["factonio-work"]')
    set_config_value("kanban.dispatch_tick_observer_profiles", '["ab-coordinator"]')
    assert _board_slugs(kb) == ["factonio-work"]


def test_unconfigured_preserves_legacy_all_boards_and_enumeration_fallback(
    isolated_gateway_home, monkeypatch,
):
    """When ``kanban.dispatch_boards`` is absent, all live boards are eligible and
    enumeration failure preserves the legacy fallback to ``default``."""
    spawned: list[tuple[str, str]] = []
    monkeypatch.setattr(kbd, "_default_spawn", lambda task, ws, board=None: spawned.append((board, task.id)) or 4242)

    default_tid = _create_ready_card("default")
    pilot_tid = _create_ready_card("factonio-work")
    other_tid = _create_ready_card("other-board")

    dispatcher = _make_dispatcher()
    assert dispatcher.ready_nonempty() is True
    results = _run_in_fresh_context(dispatcher.tick_once)
    assert [slug for slug, _ in results] == ["default", "factonio-work", "other-board"]
    assert set(spawned) == {
        ("default", default_tid),
        ("factonio-work", pilot_tid),
        ("other-board", other_tid),
    }

    # Unconfigured legacy enumeration fallback still returns default.
    monkeypatch.setattr(kb, "list_boards", lambda **kw: (_ for _ in ()).throw(OSError("boom")))
    assert _board_slugs(kb) == ["default"]


def test_allowlist_confines_tick_ready_probe_and_auto_decompose_ignoring_current_board(
    isolated_gateway_home, monkeypatch,
):
    """Real dispatcher tick claims ONLY pilot remote-worker cards; ready non-pilot/default
    cards are never claimed, probed, or auto-decomposed even if current board / env pin names them."""
    (isolated_gateway_home / "config.yaml").write_text(
        "kanban:\n"
        "  dispatch_boards:\n"
        "    - factonio-work\n"
        "  dispatch_profiles:\n"
        "    - remote-worker\n",
        encoding="utf-8",
    )
    # Current board / env selection is NOT confinement and must not widen dispatch.
    kb.set_current_board("other-board")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "other-board")

    spawned: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        kbd,
        "_default_spawn",
        lambda task, ws, board=None: spawned.append((board, task.id, task.assignee)) or 4242,
    )

    default_tid = _create_ready_card("default", assignee="remote-worker")
    other_tid = _create_ready_card("other-board", assignee="remote-worker")
    pilot_non_worker = _create_ready_card("factonio-work", assignee="default")

    dispatcher = _make_dispatcher()

    # Only non-pilot boards (and non-allowlisted assignee on pilot) have ready cards -> ready_nonempty False.
    assert _run_in_fresh_context(dispatcher.ready_nonempty) is False

    # Add a ready remote-worker card on the pilot board -> ready_nonempty True.
    pilot_tid = _create_ready_card("factonio-work", assignee="remote-worker")
    assert _run_in_fresh_context(dispatcher.ready_nonempty) is True

    # Real dispatcher tick claims ONLY pilot_tid.
    results = _run_in_fresh_context(dispatcher.tick_once)
    assert [slug for slug, _ in results] == ["factonio-work"]
    assert spawned == [("factonio-work", pilot_tid, "remote-worker")]
    assert _task_status("default", default_tid) == "ready"
    assert _task_status("other-board", other_tid) == "ready"
    assert _task_status("factonio-work", pilot_non_worker) == "ready"
    assert _task_status("factonio-work", pilot_tid) == "running"

    # Direct tick_once_for_board on non-allowlisted boards is also refused.
    assert _run_in_fresh_context(dispatcher.tick_once_for_board, "default") is None
    assert _run_in_fresh_context(dispatcher.tick_once_for_board, "other-board") is None
    assert _task_status("default", default_tid) == "ready"
    assert _task_status("other-board", other_tid) == "ready"

    # Auto-decompose only scans the allowlisted board.
    from hermes_cli import kanban_decompose as decomp

    visited_boards: list[str] = []

    def fake_list_triage_ids():
        visited_boards.append(kb.get_current_board())
        return []

    monkeypatch.setattr(decomp, "list_triage_ids", fake_list_triage_ids)
    assert _run_in_fresh_context(dispatcher.auto_decompose_tick, 3) == 0
    assert visited_boards == ["factonio-work"]


def test_allowlist_live_reloads_across_ticks_on_same_dispatcher(isolated_gateway_home, monkeypatch):
    """Editing config.yaml changes board confinement on the very next tick without restart."""
    cfg_path = isolated_gateway_home / "config.yaml"
    spawned: list[tuple[str, str]] = []
    monkeypatch.setattr(kbd, "_default_spawn", lambda task, ws, board=None: spawned.append((board, task.id)) or 4242)

    pilot_tid = _create_ready_card("factonio-work")
    other_tid = _create_ready_card("other-board")
    default_tid = _create_ready_card("default")

    dispatcher = _make_dispatcher()

    # Tick 1: only factonio-work.
    cfg_path.write_text("kanban:\n  dispatch_boards: [factonio-work]\n", encoding="utf-8")
    res1 = _run_in_fresh_context(dispatcher.tick_once)
    assert [slug for slug, _ in res1] == ["factonio-work"]
    assert spawned == [("factonio-work", pilot_tid)]
    assert _task_status("other-board", other_tid) == "ready"
    assert _task_status("default", default_tid) == "ready"

    # Tick 2: reload to empty allowlist -> fail closed, 0 boards.
    cfg_path.write_text("kanban:\n  dispatch_boards: []\n", encoding="utf-8")
    res2 = _run_in_fresh_context(dispatcher.tick_once)
    assert res2 == []
    assert _task_status("other-board", other_tid) == "ready"
    assert _task_status("default", default_tid) == "ready"

    # Tick 3: reload to comma-separated other-board (mixed case normalized).
    cfg_path.write_text('kanban:\n  dispatch_boards: "Other-Board"\n', encoding="utf-8")
    res3 = _run_in_fresh_context(dispatcher.tick_once)
    assert [slug for slug, _ in res3] == ["other-board"]
    assert spawned == [("factonio-work", pilot_tid), ("other-board", other_tid)]
    assert _task_status("default", default_tid) == "ready"

    # Tick 4: remove key -> legacy all boards resumed.
    cfg_path.write_text("kanban:\n  dispatch_in_gateway: true\n", encoding="utf-8")
    res4 = _run_in_fresh_context(dispatcher.tick_once)
    assert [slug for slug, _ in res4] == ["default", "factonio-work", "other-board"]
    assert _task_status("default", default_tid) == "running"


@pytest.mark.parametrize(
    "config_text",
    [
        "kanban:\n  dispatch_boards: []\n",
        "kanban:\n  dispatch_boards:\n",
        "kanban:\n  dispatch_boards: null\n",
        'kanban:\n  dispatch_boards: ""\n',
        'kanban:\n  dispatch_boards: "   "\n',
        'kanban:\n  dispatch_boards: "factonio-work,"\n',
        "kanban:\n  dispatch_boards: 123\n",
        "kanban:\n  dispatch_boards: true\n",
        "kanban:\n  dispatch_boards: {a: b}\n",
        "kanban:\n  dispatch_boards: [factonio-work, 1]\n",
        'kanban:\n  dispatch_boards: [factonio-work, ""]\n',
        "kanban:\n  dispatch_boards: [bad/slug]\n",
        "kanban:\n  dispatch_boards: [factonio-work, ../escape]\n",
        "kanban: [1, 2]\n",
        'kanban: "corrupt"\n',
        "[1, 2]\n",
        "kanban:\n  dispatch_boards: [unclosed\n",
    ],
    ids=[
        "empty_list",
        "bare_key",
        "null",
        "empty_str",
        "blank_str",
        "trailing_comma",
        "int",
        "bool",
        "dict",
        "non_str_item",
        "empty_item",
        "bad_slug",
        "mixed_bad_slug",
        "non_mapping_kanban_list",
        "non_mapping_kanban_str",
        "non_mapping_root",
        "broken_yaml",
    ],
)
def test_invalid_config_fails_closed_across_all_dispatcher_paths(
    isolated_gateway_home, monkeypatch, config_text,
):
    """Any invalid or unreadable board allowlist config dispatches 0 boards and never falls back."""
    (isolated_gateway_home / "config.yaml").write_text(config_text, encoding="utf-8")
    spawned: list[str] = []
    monkeypatch.setattr(kbd, "_default_spawn", lambda task, ws, board=None: spawned.append(task.id) or 4242)

    default_tid = _create_ready_card("default")
    pilot_tid = _create_ready_card("factonio-work")

    dispatcher = _make_dispatcher()
    assert _run_in_fresh_context(dispatcher.tick_once) == []
    assert _run_in_fresh_context(dispatcher.ready_nonempty) is False
    assert _run_in_fresh_context(dispatcher.auto_decompose_tick, 3) == 0
    assert spawned == []
    assert _task_status("default", default_tid) == "ready"
    assert _task_status("factonio-work", pilot_tid) == "ready"


@pytest.mark.parametrize(
    "boards_yaml",
    [
        "kanban:\n  dispatch_boards: [nonexistent-board]\n",
        "kanban:\n  dispatch_boards: [factonio-work, nonexistent-board]\n",
    ],
    ids=["single_unknown", "mixed_known_and_unknown"],
)
def test_unknown_board_fails_closed_without_creating_db_or_falling_back(
    isolated_gateway_home, monkeypatch, boards_yaml,
):
    (isolated_gateway_home / "config.yaml").write_text(boards_yaml, encoding="utf-8")
    spawned: list[str] = []
    monkeypatch.setattr(kbd, "_default_spawn", lambda task, ws, board=None: spawned.append(task.id) or 4242)

    default_tid = _create_ready_card("default")
    pilot_tid = _create_ready_card("factonio-work")

    dispatcher = _make_dispatcher()
    assert _run_in_fresh_context(dispatcher.tick_once) == []
    assert _run_in_fresh_context(dispatcher.ready_nonempty) is False
    assert _run_in_fresh_context(dispatcher.auto_decompose_tick, 3) == 0
    assert spawned == []
    assert _task_status("default", default_tid) == "ready"
    assert _task_status("factonio-work", pilot_tid) == "ready"
    assert not kb.board_dir("nonexistent-board").exists()


def test_archived_board_in_allowlist_fails_closed(isolated_gateway_home, monkeypatch):
    (isolated_gateway_home / "config.yaml").write_text(
        "kanban:\n  dispatch_boards: [factonio-work]\n", encoding="utf-8",
    )
    default_tid = _create_ready_card("default")
    kb.remove_board("factonio-work", archive=True)

    spawned: list[str] = []
    monkeypatch.setattr(kbd, "_default_spawn", lambda task, ws, board=None: spawned.append(task.id) or 4242)

    dispatcher = _make_dispatcher()
    assert _run_in_fresh_context(dispatcher.tick_once) == []
    assert _run_in_fresh_context(dispatcher.ready_nonempty) is False
    assert spawned == []
    assert _task_status("default", default_tid) == "ready"


def test_enumeration_error_with_allowlist_fails_closed_never_falls_back_to_default(
    isolated_gateway_home, monkeypatch,
):
    (isolated_gateway_home / "config.yaml").write_text(
        "kanban:\n  dispatch_boards: [factonio-work]\n", encoding="utf-8",
    )
    default_tid = _create_ready_card("default")
    pilot_tid = _create_ready_card("factonio-work")

    monkeypatch.setattr(kb, "list_boards", lambda **kw: (_ for _ in ()).throw(OSError("io error")))
    spawned: list[str] = []
    monkeypatch.setattr(kbd, "_default_spawn", lambda task, ws, board=None: spawned.append(task.id) or 4242)

    dispatcher = _make_dispatcher()
    assert _run_in_fresh_context(dispatcher.tick_once) == []
    assert _run_in_fresh_context(dispatcher.ready_nonempty) is False
    assert _run_in_fresh_context(dispatcher.auto_decompose_tick, 3) == 0
    assert spawned == []
    assert _task_status("default", default_tid) == "ready"
    assert _task_status("factonio-work", pilot_tid) == "ready"


def _install_file_observer_plugin(profile_home: Path, out_file: Path, secret_key: str) -> None:
    """Install a real native Hermes plugin that writes each observed tick as JSONL to *out_file*."""
    plugin_dir = profile_home / "plugins" / "tick-observer"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text("name: tick-observer\nversion: '1.0'\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "from agent.secret_scope import UnscopedSecretError, get_secret\n"
        "from hermes_constants import get_hermes_home\n"
        f"OUT = Path({str(out_file)!r})\n"
        f"SECRET_KEY = {secret_key!r}\n"
        "def register(ctx):\n"
        "    registered_home = str(get_hermes_home())\n"
        "    def on_tick(**kw):\n"
        "        try:\n"
        "            secret = get_secret(SECRET_KEY, '')\n"
        "        except UnscopedSecretError:\n"
        "            secret = '<unscoped>'\n"
        "        rec = {\n"
        "            'registered_home': registered_home,\n"
        "            'tick_home': str(get_hermes_home()),\n"
        "            'board': kw.get('board'),\n"
        "            'profile_name': kw.get('profile_name'),\n"
        "            'secret': secret,\n"
        "        }\n"
        "        with OUT.open('a', encoding='utf-8') as f:\n"
        "            f.write(json.dumps(rec) + '\\n')\n"
        "    ctx.register_hook('on_kanban_dispatch_tick', on_tick)\n",
        encoding="utf-8",
    )
    (profile_home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - tick-observer\n", encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    import json
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_observer_profile_invokes_only_configured_profile_in_its_own_scope_and_secret(
    isolated_gateway_home, monkeypatch,
):
    """Real root-gateway dispatcher tick invokes ONLY the explicitly configured observer
    profile via the native PluginManager lifecycle, in its own home and secret scope,
    never broadcasting to sibling profiles and never leaking root or sibling secrets."""
    root = isolated_gateway_home
    coord = root / "profiles" / "ab-coordinator"
    sibling = root / "profiles" / "sibling"
    coord_log = root / "coord_ticks.jsonl"
    sibling_log = root / "sibling_ticks.jsonl"
    root_log = root / "root_ticks.jsonl"

    _install_file_observer_plugin(coord, coord_log, "AB_OBSERVER_SECRET")
    _install_file_observer_plugin(sibling, sibling_log, "AB_OBSERVER_SECRET")
    (coord / ".env").write_text("AB_OBSERVER_SECRET=cri_coord_only\n", encoding="utf-8")
    (sibling / ".env").write_text("AB_OBSERVER_SECRET=cri_sibling_secret\n", encoding="utf-8")
    # Put a different secret in process env to prove routed observer never falls back to os.environ.
    monkeypatch.setenv("AB_OBSERVER_SECRET", "cri_process_env_should_not_leak")
    set_multiplex_active(True)

    # 1. Unconfigured observer list: neither secondary profile fires.
    (root / "config.yaml").write_text(
        "kanban:\n  dispatch_boards: [factonio-work]\n", encoding="utf-8",
    )
    dispatcher = _make_dispatcher()
    _run_in_fresh_context(dispatcher.tick_once)
    assert _read_jsonl(coord_log) == []
    assert _read_jsonl(sibling_log) == []

    # 2. Configure ab-coordinator only: ONLY ab-coordinator fires (cold native plugin discovery),
    # in its own home and with its own .env secret; sibling is never invoked.
    (root / "config.yaml").write_text(
        "kanban:\n"
        "  dispatch_boards: [factonio-work]\n"
        "  dispatch_tick_observer_profiles: [ab-coordinator]\n",
        encoding="utf-8",
    )
    _run_in_fresh_context(dispatcher.tick_once)
    coord_events = _read_jsonl(coord_log)
    assert len(coord_events) == 1
    assert coord_events[0] == {
        "registered_home": str(coord),
        "tick_home": str(coord),
        "board": "factonio-work",
        "profile_name": "ab-coordinator",
        "secret": "cri_coord_only",
    }
    assert _read_jsonl(sibling_log) == []

    # 3. Missing secret in ab-coordinator/.env fails closed to "" (never reads process env or sibling).
    (coord / ".env").write_text("", encoding="utf-8")
    _run_in_fresh_context(dispatcher.tick_once)
    coord_events = _read_jsonl(coord_log)
    assert len(coord_events) == 2
    assert coord_events[1]["secret"] == ""
    assert _read_jsonl(sibling_log) == []

    # 4. Listing "default" does NOT double-fire root's own hook.
    _install_file_observer_plugin(root, root_log, "AB_OBSERVER_SECRET")
    (root / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - tick-observer\n"
        "kanban:\n"
        "  dispatch_boards: [factonio-work]\n"
        "  dispatch_tick_observer_profiles: [default, ab-coordinator]\n",
        encoding="utf-8",
    )
    plugins_mod._reset_plugin_managers_for_tests()
    _run_in_fresh_context(dispatcher.tick_once)
    assert len(_read_jsonl(root_log)) == 1
    assert len(_read_jsonl(coord_log)) == 3
    assert _read_jsonl(sibling_log) == []


@pytest.mark.parametrize(
    "observer_yaml",
    [
        "kanban:\n  dispatch_boards: [factonio-work]\n  dispatch_tick_observer_profiles: []\n",
        "kanban:\n  dispatch_boards: [factonio-work]\n  dispatch_tick_observer_profiles: null\n",
        'kanban:\n  dispatch_boards: [factonio-work]\n  dispatch_tick_observer_profiles: ""\n',
        "kanban:\n  dispatch_boards: [factonio-work]\n  dispatch_tick_observer_profiles: 42\n",
        "kanban:\n  dispatch_boards: [factonio-work]\n  dispatch_tick_observer_profiles: [bad/profile]\n",
        "kanban:\n  dispatch_boards: [factonio-work]\n  dispatch_tick_observer_profiles: [nonexistent-profile]\n",
        "kanban:\n  dispatch_boards: [factonio-work]\n  dispatch_tick_observer_profiles: [ab-coordinator, nonexistent-profile]\n",
    ],
    ids=[
        "empty",
        "null",
        "empty_str",
        "int",
        "bad_name",
        "unknown_profile",
        "mixed_known_and_unknown_profile",
    ],
)
def test_invalid_or_unknown_observer_profile_fails_closed(isolated_gateway_home, observer_yaml):
    root = isolated_gateway_home
    coord = root / "profiles" / "ab-coordinator"
    sibling = root / "profiles" / "sibling"
    coord_log = root / "coord_ticks.jsonl"
    sibling_log = root / "sibling_ticks.jsonl"
    _install_file_observer_plugin(coord, coord_log, "AB_OBSERVER_SECRET")
    _install_file_observer_plugin(sibling, sibling_log, "AB_OBSERVER_SECRET")
    set_multiplex_active(True)

    (root / "config.yaml").write_text(observer_yaml, encoding="utf-8")
    dispatcher = _make_dispatcher()
    _run_in_fresh_context(dispatcher.tick_once)
    assert _read_jsonl(coord_log) == []
    assert _read_jsonl(sibling_log) == []
