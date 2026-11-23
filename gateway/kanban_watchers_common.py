"""Plumbing shared by the kanban notifier and dispatcher loops.

Thread offload, board enumeration, live-config coercers and the singleton
dispatcher lock live here so the notifier, dispatcher and mixin modules read
them from one place.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Mapping
from contextvars import Context
from pathlib import Path
from typing import Any, Callable, Optional

# Keep the logger name run.py used so extracted log records are unchanged.
logger = logging.getLogger("gateway.run")


def _run_in_fresh_context(func: Callable[..., Any], /, *args: Any) -> Any:
    """Run *func* in an empty ``Context`` so request-local ContextVars stay behind.

    ``asyncio.to_thread`` copies the caller's context; a lingering
    ``delegate_task`` child marker would make ``write_txn`` false-trip for
    these process-owned writers. An empty Context keeps the DB guard intact
    for real children without exempting dispatcher writes.
    """
    return Context().run(func, *args)


async def _to_thread_process_service(func: Callable[..., Any], /, *args: Any) -> Any:
    """Offload blocking process-service work without inheriting request ContextVars."""
    return await asyncio.to_thread(_run_in_fresh_context, func, *args)


def _list_boards(kb: Any) -> list:
    """Enumerate live boards; fall back to the default board when listing fails."""
    try:
        return kb.list_boards(include_archived=False)
    except Exception:
        return [kb.read_board_metadata(kb.DEFAULT_BOARD)]


def _read_user_kanban_section(setting_name: str) -> tuple[bool, Optional[Mapping[str, Any]]]:
    """Read the raw user ``kanban`` mapping without ``DEFAULT_CONFIG`` merge.

    Returns ``(True, mapping_or_none)`` on success, or ``(False, None)`` when
    config cannot be read or has a non-mapping root / ``kanban`` section so
    presence-sensitive allowlists fail closed.
    """
    try:
        from hermes_cli.config_effective import load_user_config_effective

        cfg = load_user_config_effective(fail_closed=True)
        if cfg is None:
            return True, None
        if not isinstance(cfg, Mapping):
            raise TypeError(f"config root is {type(cfg).__name__}, not a mapping")
        kanban = cfg.get("kanban")
        if kanban is not None and not isinstance(kanban, Mapping):
            raise TypeError(f"the `kanban` section is {type(kanban).__name__}, not a mapping")
        return True, kanban
    except Exception as exc:
        logger.warning(
            "kanban dispatcher: kanban.%s cannot be resolved (%s: %s) — failing closed",
            setting_name, type(exc).__name__, exc,
        )
        return False, None


def _parse_allowlist_entries(
    raw: Any,
    setting_name: str,
    normalize_fn: Callable[[str], Optional[str]],
) -> tuple[str, ...]:
    """Validate a configured list or comma-separated allowlist; fail closed to ``()`` on any error."""
    if raw is None or isinstance(raw, bool):
        logger.warning(
            "kanban dispatcher: kanban.%s is present but empty/invalid (%r) — failing closed; "
            "omit the key for legacy behavior",
            setting_name, raw,
        )
        return ()
    if isinstance(raw, str):
        if not raw.strip():
            logger.warning(
                "kanban dispatcher: kanban.%s is present but empty — failing closed; "
                "omit the key for legacy behavior",
                setting_name,
            )
            return ()
        parts = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, (list, tuple)):
        if not raw:
            logger.warning(
                "kanban dispatcher: kanban.%s is empty — failing closed; "
                "omit the key for legacy behavior",
                setting_name,
            )
            return ()
        parts = []
        for item in raw:
            if not isinstance(item, str) or not item.strip():
                logger.warning(
                    "kanban dispatcher: kanban.%s contains non-string or empty entry %r — failing closed",
                    setting_name, item,
                )
                return ()
            parts.append(item.strip())
    else:
        logger.warning(
            "kanban dispatcher: kanban.%s must be a list or comma-separated string, got %s — failing closed",
            setting_name, type(raw).__name__,
        )
        return ()

    normalized: list[str] = []
    for part in parts:
        if not part:
            logger.warning(
                "kanban dispatcher: kanban.%s contains an empty entry — failing closed",
                setting_name,
            )
            return ()
        try:
            normed = normalize_fn(part)
        except Exception as exc:
            logger.warning(
                "kanban dispatcher: kanban.%s contains invalid entry %r (%s) — failing closed",
                setting_name, part, exc,
            )
            return ()
        if not normed:
            logger.warning(
                "kanban dispatcher: kanban.%s contains invalid entry %r — failing closed",
                setting_name, part,
            )
            return ()
        normalized.append(normed)
    return tuple(dict.fromkeys(normalized))


def _dispatch_board_allowlist(kb: Any) -> Optional[tuple[str, ...]]:
    """Configured ``kanban.dispatch_boards`` allowlist, re-read live from user config.

    Returns ``None`` only when the key is absent from a readable user config
    (legacy semantics: all live boards). Returns a normalized tuple when valid,
    or ``()`` to fail closed on unreadable/invalid/empty config.
    """
    ok, kanban = _read_user_kanban_section("dispatch_boards")
    if not ok:
        return ()
    if "dispatch_boards" not in (kanban or {}):
        return None
    normalize_slug = getattr(kb, "_normalize_board_slug", None)
    if not callable(normalize_slug):
        logger.warning("kanban dispatcher: kanban_db._normalize_board_slug unavailable — failing closed")
        return ()
    return _parse_allowlist_entries(kanban["dispatch_boards"], "dispatch_boards", normalize_slug)


def _dispatch_tick_observer_profiles() -> Optional[tuple[str, ...]]:
    """Configured ``kanban.dispatch_tick_observer_profiles``, re-read live from user config.

    Returns ``None`` when absent (legacy semantics: only the dispatcher's own
    profile fires ``on_kanban_dispatch_tick``). Returns a normalized tuple when
    valid, or ``()`` to fail closed on unreadable/invalid/empty config.
    """
    ok, kanban = _read_user_kanban_section("dispatch_tick_observer_profiles")
    if not ok:
        return ()
    if "dispatch_tick_observer_profiles" not in (kanban or {}):
        return None
    try:
        from hermes_cli.profiles import normalize_profile_name
    except Exception as exc:
        logger.warning(
            "kanban dispatcher: hermes_cli.profiles unavailable for kanban.dispatch_tick_observer_profiles "
            "(%s: %s) — failing closed",
            type(exc).__name__, exc,
        )
        return ()
    return _parse_allowlist_entries(
        kanban["dispatch_tick_observer_profiles"],
        "dispatch_tick_observer_profiles",
        normalize_profile_name,
    )


def _board_slugs(kb: Any) -> list:
    """Live board slugs eligible for the embedded dispatcher this tick.

    Unconfigured (``kanban.dispatch_boards`` absent): legacy behavior — every
    live board, with ``_list_boards`` falling back to ``default`` on error.

    Configured (or config unreadable): fail closed — never falls back to
    ``default`` or all boards on invalid config, unknown/archived boards, or
    enumeration failure.
    """
    allowlist = _dispatch_board_allowlist(kb)
    if allowlist is None:
        return [b.get("slug") or kb.DEFAULT_BOARD for b in _list_boards(kb)]
    if not allowlist:
        return []
    try:
        live_boards = kb.list_boards(include_archived=False)
    except Exception as exc:
        logger.warning(
            "kanban dispatcher: board enumeration failed with kanban.dispatch_boards configured "
            "(%s: %s) — failing closed (0 boards)",
            type(exc).__name__, exc,
        )
        return []
    normalize_slug = getattr(kb, "_normalize_board_slug", None)
    live_slugs: set[str] = set()
    for entry in live_boards or ():
        if not isinstance(entry, Mapping):
            continue
        raw_slug = entry.get("slug")
        if not isinstance(raw_slug, str) or not raw_slug.strip():
            continue
        try:
            normed = normalize_slug(raw_slug) if callable(normalize_slug) else raw_slug.strip().lower()
        except Exception:
            continue
        if normed:
            live_slugs.add(normed)
    unknown = [slug for slug in allowlist if slug not in live_slugs]
    if unknown:
        logger.warning(
            "kanban dispatcher: kanban.dispatch_boards names unknown or archived board(s) %r "
            "(live: %s) — failing closed (0 boards)",
            unknown, sorted(live_slugs),
        )
        return []
    return list(allowlist)


def _positive_int_setting(kanban_cfg: dict, key: str) -> Optional[int]:
    """Parse an optional ``kanban.<key>`` int cap; None when unset or invalid (< 1 is invalid)."""
    raw = kanban_cfg.get(key)
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning("kanban dispatcher: invalid kanban.%s=%r; ignoring", key, raw)
        return None
    if value < 1:
        logger.warning("kanban dispatcher: kanban.%s=%r is below 1; ignoring", key, raw)
        return None
    logger.info("kanban dispatcher: %s=%d", key, value)
    return value


def _resolve_auto_decompose_settings(load_config: Callable[[], Any]) -> tuple[bool, int]:
    """Live (enabled, per_tick) auto-decompose settings, re-read every dispatcher tick.

    Fails safe: a config read error returns ``(False, 3)`` rather than
    re-enabling a feature the user turned off. ``per_tick`` is clamped to ``>= 1``.

    Read fresh from config on every dispatcher tick (#49638) so that flipping ``kanban.auto_decompose:
    false`` to STOP runaway fan-out takes effect on the next tick instead of requiring a gateway restart.
    Auto-decompose is a safety toggle — a user who sees it create and launch tasks they didn't intend
    reaches for this flag to halt it, and a stale boot-captured value silently ignoring that change is the
    bug reported in #49638.
    """
    try:
        cfg = load_config()
    except Exception:
        return False, 3
    kcfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    try:
        per_tick = int(kcfg.get("auto_decompose_per_tick", 3) or 3)
    except (TypeError, ValueError):
        per_tick = 3
    return bool(kcfg.get("auto_decompose", True)), max(per_tick, 1)


def _gc_retention_days() -> int:
    """``kanban.done_sub_retention_days`` (default 30; 0 disables), re-read per sweep; fails safe to 30."""
    try:
        from hermes_cli.config import load_config

        return int(((load_config() or {}).get("kanban") or {}).get("done_sub_retention_days", 30))
    except Exception:
        return 30


def _kanban_dispatch_allowed() -> bool:
    """False while the global emergency stop (`hermes pause`) is engaged.

    Checked every tick before spawning, so a pause applies on the next tick;
    in-flight workers are never touched. Fails open if estop is unimportable.
    """
    try:
        from agent.estop import check_paused
    except ImportError:
        return True
    return not check_paused("kanban", logger)


def _acquire_singleton_lock(lock_path) -> tuple[Optional[object], str]:
    """Take the exclusive, non-blocking advisory lock for the sole dispatcher.

    Only one gateway machine-wide may run the embedded dispatcher: concurrent
    dispatchers double reclaim frequency and claim events, and with
    ``wal_autocheckpoint=0`` concurrent manual checkpoints can corrupt index
    pages. ``dispatch_in_gateway`` is the primary control; this is the backstop.

    Returns ``(handle, "held")`` (release via :func:`_release_singleton_lock`),
    ``(None, "contended")`` when another process holds it (caller must NOT
    dispatch), or ``(None, "unavailable")`` when locking cannot be performed
    (caller falls back to config control).
    """
    try:
        from gateway.status import _try_acquire_file_lock  # deferred; same package
    except ImportError:
        return None, "unavailable"
    try:
        Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
        handle = open(str(lock_path), "a+", encoding="utf-8")  # windows-footgun: ok (append-mode lock handle, write not read)
    except OSError:
        return None, "unavailable"
    if not _try_acquire_file_lock(handle):
        handle.close()
        return None, "contended"
    return handle, "held"


def _release_singleton_lock(handle) -> None:
    """Release a lock acquired via :func:`_acquire_singleton_lock`."""
    if handle is None:
        return
    with contextlib.suppress(Exception):
        from gateway.status import _release_file_lock

        _release_file_lock(handle)
    with contextlib.suppress(Exception):
        handle.close()
