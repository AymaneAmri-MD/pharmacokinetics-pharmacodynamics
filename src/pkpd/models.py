"""Analytical PK/PD models.

Conventions
-----------
- time in hours (h), dose in mg, volume in L, clearance in L/h
- concentration in mg/L (numerically equal to ug/mL)
- every single-dose function returns 0 for t < 0, so doses can be
  superposed (see ``multiple_dose``)
"""
from __future__ import annotations

from typing import Callable

import numpy as np


# --------------------------------------------------------------------------- #
# Basic parameters
# --------------------------------------------------------------------------- #
def elimination_rate(CL: float, V: float) -> float:
    """First-order elimination rate constant k = CL / V (1/h)."""
    return CL / V


def half_life(CL: float, V: float) -> float:
    """Elimination half-life t1/2 = ln(2) * V / CL (h)."""
    return np.log(2) * V / CL


def auc_inf(dose: float, CL: float, F: float = 1.0) -> float:
    """Area under the curve from 0 to infinity: AUC = F * Dose / CL (mg.h/L)."""
    return F * dose / CL


# --------------------------------------------------------------------------- #
# One-compartment models
# --------------------------------------------------------------------------- #
def one_comp_iv(t, dose: float, V: float, k: float) -> np.ndarray:
    """One-compartment model, IV bolus: C(t) = (Dose / V) * exp(-k t)."""
    t = np.asarray(t, dtype=float)
    tp = np.clip(t, 0.0, None)
    return np.where(t >= 0, (dose / V) * np.exp(-k * tp), 0.0)


def one_comp_oral(t, dose: float, V: float, k: float, ka: float, F: float = 1.0) -> np.ndarray:
    """One-compartment model, first-order oral absorption (Bateman function).

    C(t) = F*Dose*ka / (V*(ka-k)) * (exp(-k t) - exp(-ka t))
    The limit ka -> k is handled separately to avoid a division by zero.
    """
    t = np.asarray(t, dtype=float)
    tp = np.clip(t, 0.0, None)
    if np.isclose(ka, k):
        c = (F * dose / V) * k * tp * np.exp(-k * tp)
    else:
        c = F * dose * ka / (V * (ka - k)) * (np.exp(-k * tp) - np.exp(-ka * tp))
    return np.where(t >= 0, c, 0.0)


def tmax_oral(k: float, ka: float) -> float:
    """Time of peak concentration after oral dosing: ln(ka/k) / (ka - k) (h)."""
    if np.isclose(ka, k):
        return 1.0 / k
    return np.log(ka / k) / (ka - k)


def cmax_oral(dose: float, V: float, k: float, ka: float, F: float = 1.0) -> float:
    """Peak concentration after a single oral dose (mg/L)."""
    return float(one_comp_oral(tmax_oral(k, ka), dose, V, k, ka, F))


# --------------------------------------------------------------------------- #
# Two-compartment model
# --------------------------------------------------------------------------- #
def two_comp_iv(t, dose: float, V1: float, k10: float, k12: float, k21: float) -> np.ndarray:
    """Two-compartment model, IV bolus (central concentration).

    C(t) = A exp(-alpha t) + B exp(-beta t)
    where alpha and beta are the hybrid (macro) rate constants.
    """
    t = np.asarray(t, dtype=float)
    tp = np.clip(t, 0.0, None)
    s = k10 + k12 + k21
    disc = np.sqrt(s**2 - 4.0 * k10 * k21)
    alpha = (s + disc) / 2.0
    beta = (s - disc) / 2.0
    A = (dose / V1) * (alpha - k21) / (alpha - beta)
    B = (dose / V1) * (k21 - beta) / (alpha - beta)
    return np.where(t >= 0, A * np.exp(-alpha * tp) + B * np.exp(-beta * tp), 0.0)


# --------------------------------------------------------------------------- #
# Repeated dosing
# --------------------------------------------------------------------------- #
def multiple_dose(
    single_dose_fn: Callable[..., np.ndarray],
    t,
    tau: float,
    n_doses: int,
    **params,
) -> np.ndarray:
    """Superposition principle for repeated doses given every ``tau`` hours."""
    t = np.asarray(t, dtype=float)
    total = np.zeros_like(t)
    for i in range(n_doses):
        total += single_dose_fn(t - i * tau, **params)
    return total


def accumulation_ratio(k: float, tau: float) -> float:
    """Accumulation ratio R = 1 / (1 - exp(-k * tau))."""
    return 1.0 / (1.0 - np.exp(-k * tau))


def css_average(dose: float, CL: float, tau: float, F: float = 1.0) -> float:
    """Average steady-state concentration: Css,avg = F * Dose / (CL * tau) (mg/L)."""
    return F * dose / (CL * tau)


# --------------------------------------------------------------------------- #
# Pharmacodynamics
# --------------------------------------------------------------------------- #
def emax_model(C, Emax: float, EC50: float, hill: float = 1.0, E0: float = 0.0) -> np.ndarray:
    """Sigmoid Emax model: E = E0 + Emax * C^n / (EC50^n + C^n)."""
    C = np.asarray(C, dtype=float)
    return E0 + Emax * C**hill / (EC50**hill + C**hill)
