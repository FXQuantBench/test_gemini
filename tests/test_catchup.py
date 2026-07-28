from datetime import date, datetime, timezone

from harness.catchup import (
    CATCH_UP_START,
    catch_up_cutoff,
    next_business_day,
    next_regular_eval_target,
    plan_catch_up,
)


def test_next_business_day_skips_weekend():
    assert next_business_day(date(2026, 6, 5)) == date(2026, 6, 8)


def test_cutoff_excludes_next_regular_daily_eval_target():
    now = datetime(2026, 7, 14, 7, 0, tzinfo=timezone.utc)  # Tuesday before 08:30
    assert next_regular_eval_target(now) == date(2026, 7, 13)
    assert catch_up_cutoff(now) == date(2026, 7, 10)


def test_cutoff_advances_after_daily_eval_runs():
    now = datetime(2026, 7, 14, 9, 0, tzinfo=timezone.utc)  # Tuesday after 08:30
    assert next_regular_eval_target(now) == date(2026, 7, 14)
    assert catch_up_cutoff(now) == date(2026, 7, 13)


def test_first_strategy_push_starts_at_fixed_date():
    state, action, eval_date = plan_catch_up(
        {"status": "not_started", "next_eval_date": None},
        event_name="push",
        strategy_changed=True,
        resume_after_reflection=False,
        now=datetime(2026, 7, 14, 7, 0, tzinfo=timezone.utc),
        capacity_available=True,
    )
    assert state["status"] == "running"
    assert action == "evaluate"
    assert eval_date == CATCH_UP_START
    assert state["next_eval_date"] == CATCH_UP_START.isoformat()


def test_manual_continuation_evaluates_pending_date():
    state, action, eval_date = plan_catch_up(
        {"status": "running", "next_eval_date": "2026-06-01"},
        event_name="workflow_dispatch",
        strategy_changed=False,
        resume_after_reflection=False,
        now=datetime(2026, 7, 14, 7, 0, tzinfo=timezone.utc),
        capacity_available=True,
    )
    assert state["status"] == "running"
    assert action == "evaluate"
    assert eval_date == CATCH_UP_START


def test_state_commit_push_does_not_restart_active_catch_up():
    state, action, eval_date = plan_catch_up(
        {"status": "awaiting_reflection", "next_eval_date": "2026-06-02"},
        event_name="push",
        strategy_changed=False,
        resume_after_reflection=False,
        now=datetime(2026, 7, 14, 7, 0, tzinfo=timezone.utc),
        capacity_available=True,
    )
    assert state["status"] == "awaiting_reflection"
    assert action == "noop"
    assert eval_date is None


def test_awaiting_reflection_requires_reflection_before_next_eval():
    state, action, eval_date = plan_catch_up(
        {"status": "awaiting_reflection", "next_eval_date": "2026-06-02"},
        event_name="schedule",
        strategy_changed=False,
        resume_after_reflection=False,
        now=datetime(2026, 7, 14, 7, 0, tzinfo=timezone.utc),
        capacity_available=True,
    )
    assert state["status"] == "awaiting_reflection"
    assert action == "reflect"
    assert eval_date is None


def test_cap_pauses_catch_up_without_losing_state():
    state, action, eval_date = plan_catch_up(
        {"status": "running", "next_eval_date": "2026-06-01"},
        event_name="schedule",
        strategy_changed=False,
        resume_after_reflection=False,
        now=datetime(2026, 7, 14, 7, 0, tzinfo=timezone.utc),
        capacity_available=False,
    )
    assert state["status"] == "running"
    assert action == "noop"
    assert eval_date is None