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
