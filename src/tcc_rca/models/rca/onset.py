"""Onset (root-cause occurrence time) estimators over a per-minute anomaly signal (ADR 0020).

Every estimator works on positions of a 1-D signal (one value per matrix row of the query's
day span) and returns candidate positions in order of preference; the caller maps positions
back to rows/times. ``query`` is the boolean mask of the positions inside the query window.

* threshold crossing (T1/T2): starts of runs at or above the validation threshold, in time
  order; fallback: the signal maximum inside the query;
* change point (T3/T4): univariate Bayesian online change-point detection (Adams & MacKay,
  2007) with a normal model of unknown mean and variance, inspired by BARO (Pham et al., FSE
  2024), which runs a multivariate BOCPD on the raw metrics.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def robust_standardize(x: FloatArray) -> FloatArray:
    """(x - median) / (1.4826 MAD) over the finite values; a zero MAD becomes 1."""
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x)
    med = float(np.median(finite))
    mad = 1.4826 * float(np.median(np.abs(finite - med)))
    out: FloatArray = (x - med) / (mad if mad > 0 else 1.0)
    return out


def _student_t_logpdf(x: float, mu: FloatArray, var: FloatArray, df: FloatArray) -> FloatArray:
    z = (x - mu) ** 2 / (df * var)
    lg = np.vectorize(math.lgamma)
    out: FloatArray = (
        lg((df + 1.0) / 2.0)
        - lg(df / 2.0)
        - 0.5 * np.log(np.pi * df * var)
        - (df + 1.0) / 2.0 * np.log1p(z)
    )
    return out


def bocpd_run_length(
    x: FloatArray,
    hazard: float,
    mu0: float = 0.0,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> FloatArray:
    """Run-length posterior ``R[t, r] = P(run length r | x[0..t])``, shape ``(T, T + 1)``.

    The run length counts the observations of the current run, ``x[t]`` included (``r = 0``:
    a new run starts after ``t``).

    Normal-gamma prior (``mu0, kappa0, alpha0, beta0``), constant hazard. ``x`` must be finite.
    """
    if not 0.0 < hazard < 1.0:
        raise ValueError("hazard must be in (0, 1)")
    if not np.all(np.isfinite(x)):
        raise ValueError("BOCPD needs a finite signal")
    n = x.size
    out = np.zeros((n, n + 1), dtype=np.float64)
    log_r = np.array([0.0])  # log P(r_{t-1}, x[:t]) up to a constant; r = 0 before any data
    mu, kappa = np.array([mu0]), np.array([kappa0])
    alpha, beta = np.array([alpha0]), np.array([beta0])
    log_h, log_1mh = math.log(hazard), math.log1p(-hazard)
    for t in range(n):
        xt = float(x[t])
        var = beta * (kappa + 1.0) / (alpha * kappa)
        log_pred = _student_t_logpdf(xt, mu, var, 2.0 * alpha)
        growth = log_r + log_pred + log_1mh
        change = np.logaddexp.reduce(log_r + log_pred + log_h)
        log_r = np.concatenate([[change], growth])
        log_r -= np.logaddexp.reduce(log_r)
        out[t, : t + 2] = np.exp(log_r)
        # posterior parameters for every run length (r = 0 restarts from the prior)
        beta_new = beta + kappa * (xt - mu) ** 2 / (2.0 * (kappa + 1.0))
        mu_new = (kappa * mu + xt) / (kappa + 1.0)
        mu = np.concatenate([[mu0], mu_new])
        kappa = np.concatenate([[kappa0], kappa + 1.0])
        alpha = np.concatenate([[alpha0], alpha + 0.5])
        beta = np.concatenate([[beta0], beta_new])
    return out


def bocpd_onsets(signal: FloatArray, query: BoolArray, hazard: float) -> list[int]:
    """Candidate onset positions from the MAP run length, best first.

    A drop of the MAP run length at position t (``map[t] < map[t-1]``) says a new regime began
    at ``t - map[t] + 1`` (the run length counts the observations of the current run).
    Candidates whose start lies inside the query are ordered by the size of the drop (then
    earlier first); the signal maximum inside the query closes the list.
    Non-finite values (minutes without a complete window) are dropped before the run.
    """
    valid = np.flatnonzero(np.isfinite(signal))
    fallback = _query_argmax(signal, query)
    if valid.size < 2:
        return [fallback]
    # the valid positions of a day span are contiguous (complete windows start W - 1 minutes
    # into the day); use the longest contiguous tail to be safe
    breaks = np.flatnonzero(np.diff(valid) != 1)
    if breaks.size:
        valid = valid[breaks[-1] + 1 :]
    x = robust_standardize(signal[valid])
    r = bocpd_run_length(x, hazard)
    map_rl = r.argmax(axis=1)
    found: list[tuple[int, int]] = []  # (-drop, start position)
    for t in range(1, map_rl.size):
        if map_rl[t] < map_rl[t - 1]:
            # run length r after observing position t = positions t - r + 1 .. t
            start = int(valid[min(t - int(map_rl[t]) + 1, map_rl.size - 1)])
            if query[start]:
                found.append((int(map_rl[t]) - int(map_rl[t - 1]), start))
    ordered = [p for _, p in sorted(found)]
    out: list[int] = []
    for p in [*ordered, fallback]:
        if p not in out:
            out.append(p)
    return out


def threshold_onsets(signal: FloatArray, query: BoolArray, threshold: float) -> list[int]:
    """Starts of the runs at or above ``threshold`` inside the query (time order), then the
    signal maximum inside the query."""
    above = np.isfinite(signal) & (signal >= threshold) & query
    starts = [int(i) for i in np.flatnonzero(above) if i == 0 or not above[i - 1]]
    fallback = _query_argmax(signal, query)
    return starts + ([fallback] if fallback not in starts else [])


def _query_argmax(signal: FloatArray, query: BoolArray) -> int:
    inside = np.flatnonzero(query)
    if inside.size == 0:
        raise ValueError("the query window has no matrix row")
    vals = np.where(np.isfinite(signal[inside]), signal[inside], -np.inf)
    return int(inside[int(np.argmax(vals))])  # all -inf -> first query minute


def pick_onsets(
    candidates: list[int],
    signal: FloatArray,
    query: BoolArray,
    n: int,
    min_separation: int,
) -> list[int]:
    """``n`` onsets: the first candidate, then for each further failure the first candidate at
    least ``min_separation`` positions from every chosen one, else the signal maximum among
    such query positions, else the first onset again."""
    chosen = [candidates[0]]
    while len(chosen) < n:
        far = [c for c in candidates if all(abs(c - o) >= min_separation for o in chosen)]
        if far:
            chosen.append(far[0])
            continue
        allowed = query.copy()
        for o in chosen:
            lo, hi = max(0, o - min_separation + 1), o + min_separation
            allowed[lo:hi] = False
        chosen.append(_query_argmax(signal, allowed) if allowed.any() else chosen[0])
    return chosen
