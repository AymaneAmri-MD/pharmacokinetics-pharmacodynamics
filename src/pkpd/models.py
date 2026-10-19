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


def two_comp_derived(dose: float, V1: float, k10: float, k12: float, k21: float) -> dict:
    """Hybrid constants and derived parameters of the two-compartment IV bolus model.

    Returns alpha, beta (1/h), A, B (mg/L), CL (L/h), V2 and Vss (L),
    the half-lives of both phases (h) and the AUC (mg.h/L).
    """
    s = k10 + k12 + k21
    disc = np.sqrt(s**2 - 4.0 * k10 * k21)
    alpha = (s + disc) / 2.0
    beta = (s - disc) / 2.0
    A = (dose / V1) * (alpha - k21) / (alpha - beta)
    B = (dose / V1) * (k21 - beta) / (alpha - beta)
    CL = k10 * V1
    V2 = V1 * k12 / k21
    return {
        "alpha (1/h)": alpha,
        "beta (1/h)": beta,
        "A (mg/L)": A,
        "B (mg/L)": B,
        "t1/2 alpha (h)": np.log(2) / alpha,
        "t1/2 beta (h)": np.log(2) / beta,
        "CL (L/h)": CL,
        "V2 (L)": V2,
        "Vss (L)": V1 + V2,
        "AUC (mg.h/L)": dose / CL,
    }


def two_comp_ode(t, dose: float, V1: float, k10: float, k12: float, k21: float):
    """Numerical solution of the two-compartment model (IV bolus).

    Returns (t, C_central, C_peripheral) in mg/L. Useful to validate the
    analytical solution and to look at the peripheral compartment.
    """
    from scipy.integrate import solve_ivp

    t = np.asarray(t, dtype=float)
    V2 = V1 * k12 / k21

    def rhs(_, a):
        a1, a2 = a
        return [-(k10 + k12) * a1 + k21 * a2, k12 * a1 - k21 * a2]

    sol = solve_ivp(rhs, (t[0], t[-1]), [dose, 0.0], t_eval=t, rtol=1e-10, atol=1e-12)
    return sol.t, sol.y[0] / V1, sol.y[1] / V2


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


def ec_fraction(EC50: float, hill: float, fraction: float) -> float:
    """Concentration giving a given fraction of Emax: EC50 * (f / (1 - f))^(1/n).

    Example: ec_fraction(EC50, n, 0.9) is the EC90.
    """
    return EC50 * (fraction / (1.0 - fraction)) ** (1.0 / hill)


def effect_site_concentration(t, dose: float, V: float, k: float, ke0: float) -> np.ndarray:
    """Effect-site (biophase) concentration after a one-compartment IV bolus.

    The effect compartment follows dCe/dt = ke0 * (Cp - Ce), which gives
    Ce(t) = C0 * ke0 / (ke0 - k) * (exp(-k t) - exp(-ke0 t)).
    A small ke0 means a long delay between plasma concentration and effect.
    """
    t = np.asarray(t, dtype=float)
    tp = np.clip(t, 0.0, None)
    c0 = dose / V
    if np.isclose(ke0, k):
        ce = c0 * k * tp * np.exp(-k * tp)
    else:
        ce = c0 * ke0 / (ke0 - k) * (np.exp(-k * tp) - np.exp(-ke0 * tp))
    return np.where(t >= 0, ce, 0.0)


def fraction_time_above(c, threshold: float) -> float:
    """Fraction of time a concentration profile spends above a threshold.

    Assumes ``c`` is sampled on a uniform time grid.
    """
    c = np.asarray(c, dtype=float)
    return float(np.mean(c > threshold))
