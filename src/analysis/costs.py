"""
Sleep costs computed from the activity index y [a.u.] only (no physical constants).

A cell sleeps when its estimate (point forecast or conformal upper bound) is below a threshold
theta [a.u.]. Its traffic is then assumed to be absorbed by the neighbouring cells, so sleeping
is wrong only when the true y reaches theta. theta is the single assumption and is meant to be swept.

    sleep_%       cell-steps asleep / all cell-steps (first-order energy saving)
    violation_%   sleeping cell-steps with y >= theta / sleeping cell-steps
    unmet_%       sum(max(y - theta, 0)) over sleeping cell-steps / sum(y)
    wakeups/day   asleep -> awake transitions per cell per day (switching cost)

All arrays are [time steps, cells]; the time axis must be in order.
"""
import numpy as np

from analyse_result import STEPS_PER_DAY


def sleep_decision(est, theta):
    """True where the cell sleeps: its estimate is below theta."""
    return est < theta


def sleep_cost(truth, asleep, theta):
    """Sleep rate and QoS cost of a sleep decision against the true activity."""
    violated = asleep & (truth >= theta)
    return {
        "sleep_%": 100 * asleep.mean(),
        "violation_%": 100 * violated.sum() / max(asleep.sum(), 1),
        "unmet_%": 100 * np.maximum(truth - theta, 0)[asleep].sum() / truth.sum(),
    }


def wakeups_per_cell_day(asleep, steps):
    """Asleep -> awake transitions between consecutive steps, per cell and per day.
    steps: absolute step index of each row (gaps between rows are not counted as transitions)."""
    consecutive = np.diff(steps) == 1
    woke = asleep[:-1] & ~asleep[1:] & consecutive[:, None]
    return woke.sum() / asleep.shape[1] / (len(steps) / STEPS_PER_DAY)
