import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pkpd import models as m  # noqa: E402
from pkpd.drugs import DRUGS  # noqa: E402

_trapz = getattr(np, "trapezoid", None) or np.trapz


def test_iv_initial_concentration():
    assert np.isclose(m.one_comp_iv(0.0, 100, 10, 0.1), 10.0)


def test_iv_halves_after_one_half_life():
    d = DRUGS["paracetamol"]
    c0 = m.one_comp_iv(0.0, d.iv_dose, d.V, d.k)
    c_half = m.one_comp_iv(d.t_half, d.iv_dose, d.V, d.k)
    assert np.isclose(c_half, c0 / 2)


def test_oral_auc_matches_f_dose_over_cl():
    d = DRUGS["amoxicillin"]
    t = np.linspace(0, 200, 200001)
    c = m.one_comp_oral(t, d.oral_dose, d.V, d.k, d.ka, d.F)
    assert np.isclose(_trapz(c, t), m.auc_inf(d.oral_dose, d.CL, d.F), rtol=1e-3)


def test_tmax_is_the_peak():
    d = DRUGS["paracetamol"]
    t = np.linspace(0, 12, 120001)
    c = m.one_comp_oral(t, d.oral_dose, d.V, d.k, d.ka, d.F)
    assert np.isclose(t[np.argmax(c)], m.tmax_oral(d.k, d.ka), atol=1e-3)


def test_two_comp_initial_concentration():
    assert np.isclose(m.two_comp_iv(0.0, 100, 10, 0.2, 0.3, 0.2), 10.0)


def test_accumulation_matches_simulation():
    d = DRUGS["amoxicillin"]
    n = 60  # many doses -> steady state
    t_peak = np.array([(n - 1) * d.tau])
    c_last_peak = m.multiple_dose(m.one_comp_iv, t_peak, d.tau, n,
                                  dose=d.iv_dose, V=d.V, k=d.k)[0]
    c_single_peak = d.iv_dose / d.V
    assert np.isclose(c_last_peak / c_single_peak, m.accumulation_ratio(d.k, d.tau), rtol=1e-2)


def test_emax_at_ec50_is_half_of_emax():
    assert np.isclose(m.emax_model(5.0, Emax=100, EC50=5.0), 50.0)


def test_two_comp_hybrid_constants_identities():
    k10, k12, k21 = 0.35, 0.80, 0.30
    p = m.two_comp_derived(500, 15, k10, k12, k21)
    alpha, beta = p["alpha (1/h)"], p["beta (1/h)"]
    assert np.isclose(alpha + beta, k10 + k12 + k21)
    assert np.isclose(alpha * beta, k10 * k21)
    assert np.isclose(p["A (mg/L)"] + p["B (mg/L)"], 500 / 15)


def test_two_comp_auc_matches_a_over_alpha_plus_b_over_beta():
    p = m.two_comp_derived(500, 15, 0.35, 0.80, 0.30)
    auc = p["A (mg/L)"] / p["alpha (1/h)"] + p["B (mg/L)"] / p["beta (1/h)"]
    assert np.isclose(auc, p["AUC (mg.h/L)"])


def test_two_comp_ode_matches_analytical():
    t = np.linspace(0, 48, 500)
    c_ana = m.two_comp_iv(t, 500, 15, 0.35, 0.80, 0.30)
    _, c1, _ = m.two_comp_ode(t, 500, 15, 0.35, 0.80, 0.30)
    assert np.allclose(c1, c_ana, rtol=1e-6, atol=1e-8)


def test_ec_fraction_gives_expected_effect():
    EC50, hill, Emax = 2.0, 1.7, 100.0
    for f in (0.1, 0.5, 0.9):
        c = m.ec_fraction(EC50, hill, f)
        assert np.isclose(m.emax_model(c, Emax, EC50, hill), f * Emax)


def test_effect_site_starts_at_zero_and_peaks_when_it_crosses_plasma():
    dose, V, k, ke0 = 100.0, 50.0, 0.2, 0.3
    t = np.linspace(0, 40, 400001)
    ce = m.effect_site_concentration(t, dose, V, k, ke0)
    cp = m.one_comp_iv(t, dose, V, k)
    assert ce[0] == 0.0
    i = np.argmax(ce)
    assert np.isclose(ce[i], cp[i], rtol=1e-4)  # classic property of the effect compartment
    assert np.isclose(t[i], np.log(ke0 / k) / (ke0 - k), atol=1e-3)


def test_fraction_time_above():
    c = np.array([0.0, 1.0, 2.0, 3.0])
    assert m.fraction_time_above(c, 1.5) == 0.5
