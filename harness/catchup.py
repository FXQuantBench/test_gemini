"""Pure date and state helpers for the one-time daily-evaluation catch-up."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any


CATCH_UP_START = date(2026, 6, 1)
DAILY_EVAL_TIME_UTC = time(8, 30)
DAILY_EVAL_WEEKDAYS = {1, 2, 3, 4, 5}  # Tuesday through Saturday


def is_business_day(value: date) -> bool:
    """Return whether a date is eligible for a one-day GBPUSD evaluation."""
    return value.weekday() < 5


def next_business_day(value: date) -> date:
    """Return the first eligible evaluation day strictly after ``value``."""
    candidate = value + timedelta(days=1)
    while not is_business_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def previous_business_day(value: date) -> date:
    """Return the first eligible evaluation day strictly before ``value``."""
    candidate = value - timedelta(days=1)
    while not is_business_day(candidate):
        candidate -= timedelta(days=1)
    return candidate


def next_regular_eval_target(now: datetime) -> date:
    """Return the date that the next scheduled daily evaluation will target."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    now = now.astimezone(timezone.utc)
    for offset in range(8):
        scheduled_day = now.date() + timedelta(days=offset)
        if scheduled_day.weekday() not in DAILY_EVAL_WEEKDAYS:
            continue
        scheduled_at = datetime.combine(
            scheduled_day,
            DAILY_EVAL_TIME_UTC,
            tzinfo=timezone.utc,
        )
        if scheduled_at > now:
            return scheduled_day - timedelta(days=1)
    raise RuntimeError("Could not find the next scheduled daily evaluation")


def catch_up_cutoff(now: datetime) -> date:
    """Return the final catch-up date before the next normal daily evaluation."""
    return previous_business_day(next_regular_eval_target(now))


def plan_catch_up(
    state: dict[str, Any],
    *,
    event_name: str,
    strategy_changed: bool,
    resume_after_reflection: bool,
    now: datetime,
    capacity_available: bool,
) -> tuple[dict[str, Any], str, date | None]:
    """Plan one catch-up controller action without performing workflow I/O.

    Actions are ``evaluate``, ``reflect``, and ``noop``. A push only starts a
    previously untouched repository when it changes ``strategy.py``; subsequent
    progress is driven by explicit continuation dispatches or the daily schedule.
    """
    next_state = dict(state)
    status = next_state.get("status", "not_started")
    started_now = False

    if status == "not_started":
        if event_name != "push" or not strategy_changed:
            return next_state, "noop", None
        next_state["status"] = "running"
        next_state["next_eval_date"] = CATCH_UP_START.isoformat()
        status = "running"
        started_now = True

    if event_name == "push" and not started_now:
        return next_state, "noop", None

    if status == "complete":
        return next_state, "noop", None

    if status == "awaiting_reflection":
        if not capacity_available:
            return next_state, "noop", None
        if not resume_after_reflection:
            return next_state, "reflect", None
        next_state["status"] = "running"
        status = "running"

    if status != "running":
        raise ValueError(f"Unsupported catch-up state: {status!r}")

    next_eval_date = date.fromisoformat(next_state["next_eval_date"])
    if next_eval_date > catch_up_cutoff(now):
        next_state["status"] = "complete"
        return next_state, "noop", None
    if not capacity_available:
        return next_state, "noop", None
    return next_state, "evaluate", next_eval_date