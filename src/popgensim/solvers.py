"""Fixed-step ODE solvers written from scratch (Euler and classical RK4).

All solvers integrate dx/dt = f(t, x), where ``f`` takes a scalar time and a
NumPy state vector and returns a NumPy vector of the same shape. That is
exactly what ``models.make_replicator(W)`` returns.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

RHS = Callable[[float, np.ndarray], np.ndarray]


# --------------------------------------------------------------------------
# Single steps
# --------------------------------------------------------------------------
def euler_step(f: RHS, t: float, x: np.ndarray, dt: float) -> np.ndarray:
    """Forward Euler: x_{n+1} = x_n + dt * f(t_n, x_n).  Global error O(dt)."""
    return x + dt * f(t, x)


def rk4_step(f: RHS, t: float, x: np.ndarray, dt: float) -> np.ndarray:
    """Classical Runge-Kutta 4th order.  Global error O(dt^4).

        k1 = f(t,        x)
        k2 = f(t + dt/2, x + dt/2 * k1)
        k3 = f(t + dt/2, x + dt/2 * k2)
        k4 = f(t + dt,   x + dt   * k3)
        x_{n+1} = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    """
    k1 = f(t, x)
    k2 = f(t + dt / 2, x + dt / 2 * k1)
    k3 = f(t + dt / 2, x + dt / 2 * k2)
    k4 = f(t + dt, x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


STEPPERS = {"euler": euler_step, "rk4": rk4_step}


# --------------------------------------------------------------------------
# Integrator
# --------------------------------------------------------------------------
def integrate(
    f: RHS,
    x0: np.ndarray,
    t_span: Tuple[float, float],
    dt: float,
    method: str = "rk4",
    normalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate dx/dt = f(t, x) from t_span[0] to t_span[1] with fixed step dt."""
    if method not in STEPPERS:
        raise ValueError(f"Unknown method {method!r}; choose from {list(STEPPERS)}.")
    t0, t1 = t_span
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if t1 <= t0:
        raise ValueError("t_span must satisfy t1 > t0.")

    step = STEPPERS[method]
    n_full = int(np.floor((t1 - t0) / dt + 1e-12))
    remainder = (t1 - t0) - n_full * dt
    has_tail = remainder > 1e-12 * max(1.0, abs(t1 - t0))
    n = n_full + (1 if has_tail else 0)

    x = np.asarray(x0, dtype=float).copy()
    t = np.empty(n + 1)
    X = np.empty((n + 1, x.size))
    t[0], X[0] = t0, x

    cur_t = t0
    for i in range(1, n + 1):
        h = dt if i <= n_full else remainder
        x = step(f, cur_t, x, h)
        if normalize:
            x = np.clip(x, 0.0, None)
            x = x / x.sum()
        cur_t += h
        t[i], X[i] = cur_t, x

    t[-1] = t1
    return t, X