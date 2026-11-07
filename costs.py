"""
Cost model of the reference paper: energy + QoS degradation + switching.

Conventions
- D: traffic per cell in one slot, shape [N]; a: 1 = active, 0 = sleep, shape [N].
- Terms are summed as in the paper, c_t = sum_i p_i + c_d + c_s (no unit conversion).
- The paper uses 30-min slots, MILAN here has 10-min steps: aggregate 3 steps
  (or rescale beta_s) before comparing numbers with the paper.
- f_ser = 1 / (C - rho) depends on the traffic unit: with raw MILAN values it is tiny and
  the QoS term is negligible next to energy. Calibrate the unit (or beta_d) before use.

Everything the paper text does not define is marked ASSUMPTION or TODO.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostParams:
    p_fixed: float = 160.0  # P^f [W]
    p_load: float = 216.0   # P^l at full load [W]
    beta_d: float = 50.0    # QoS penalty [W/s]
    beta_s: float = 100.0   # cost of a sleep -> active switch [Wh]
    eps: float = 1e-9       # avoids division by zero when rho -> C


def estimate_capacity(train, q=0.99, headroom=1.2):
    """ASSUMPTION: C_i = headroom * q-quantile of the training traffic of cell i. Returns [N]."""
    return headroom * np.quantile(train, q, axis=0)


def dispatch(D, a):
    """
    ASSUMPTION (placeholder for the operator-defined dispatching of [24], no topology):
    the traffic of sleeping cells is moved to active cells, proportionally to their own
    traffic. Returns the load rho [N]. With no active cell the traffic is unserved.
    """
    active = a.astype(bool)
    rho = np.where(active, D, 0.0)
    moved, base = D[~active].sum(), rho.sum()
    if moved > 0 and base > 0:
        rho = rho * (1 + moved / base)
    elif moved > 0 and active.any():  # active cells carry no traffic: split evenly
        rho = np.where(active, moved / active.sum(), 0.0)
    return rho


def energy(a, rho, C, p):
    """
    p_i = P^f + P^l(rho_i) for active cells, 0 for sleeping ones.
    ASSUMPTION: P^l grows linearly with utilization, up to P^l at rho = C.
    """
    u = np.minimum(rho / C, 1.0)
    return np.where(a.astype(bool), p.p_fixed + p.p_load * u, 0.0)


def f_ser(rho, C, p):
    """Average queueing delay 1 / (C - rho). The paper ignores rho >= C; here it is clipped."""
    return 1.0 / np.maximum(C - rho, p.eps)


def f_tran(D, a):
    """TODO: transmission delay of the dispatching in [24] (needs the cell topology)."""
    return 0.0


def qos_cost(D, a, rho, C, p):
    active = a.astype(bool)
    return p.beta_d * (f_tran(D, a) + f_ser(rho[active], C[active], p).sum())


def switching_cost(a_prev, a, p):
    """Only sleep -> active transitions are charged."""
    return p.beta_s * np.maximum(a.astype(int) - a_prev.astype(int), 0).sum()


def slot_cost(D, a, a_prev, C, p):
    """Cost of one slot. 'violations' counts active cells with rho >= C."""
    active = a.astype(bool)
    rho = dispatch(D, a)
    out = {
        "energy": energy(a, rho, C, p).sum(),
        "qos": qos_cost(D, a, rho, C, p),
        "switching": switching_cost(a_prev, a, p),
        "violations": int((active & (rho >= C)).sum()),
        # traffic with no active cell to serve it (otherwise "all asleep" looks free)
        "unserved": float(D.sum()) if not active.any() else 0.0,
    }
    out["total"] = out["energy"] + out["qos"] + out["switching"]
    return out


def evaluate_policy(D, A, C, p=CostParams(), a0=None):
    """
    D, A: [T, N] true traffic and actions (1 = active). a0: state before the first slot
    (default: all active). Returns a dict of [T] arrays with the per-slot costs.
    """
    a_prev = np.ones(D.shape[1], dtype=int) if a0 is None else a0
    rows = []
    for t in range(D.shape[0]):
        rows.append(slot_cost(D[t], A[t], a_prev, C, p))
        a_prev = A[t]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}
