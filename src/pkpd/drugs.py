"""Illustrative PK parameters for a few common drugs.

IMPORTANT: values are rounded, textbook-style approximations for a ~70 kg
adult, meant for teaching and demonstration only. They are NOT clinical
dosing guidance. Always check primary sources (drug labels, population PK
studies) before using any number for real work.
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import elimination_rate, half_life


@dataclass(frozen=True)
class Drug:
    name: str
    V: float          # volume of distribution (L)
    CL: float         # clearance (L/h)
    F: float          # oral bioavailability (fraction)
    ka: float         # first-order absorption rate constant (1/h)
    iv_dose: float    # example IV dose (mg)
    oral_dose: float  # example oral dose (mg)
    tau: float        # example dosing interval (h)

    @property
    def k(self) -> float:
        return elimination_rate(self.CL, self.V)

    @property
    def t_half(self) -> float:
        return half_life(self.CL, self.V)


DRUGS = {
    "paracetamol": Drug("Paracetamol", V=65.0, CL=20.0, F=0.9, ka=2.0,
                        iv_dose=1000.0, oral_dose=1000.0, tau=6.0),
    "amoxicillin": Drug("Amoxicillin", V=21.0, CL=13.0, F=0.9, ka=2.0,
                        iv_dose=500.0, oral_dose=500.0, tau=8.0),
    "morphine": Drug("Morphine", V=245.0, CL=80.0, F=0.3, ka=1.5,
                     iv_dose=10.0, oral_dose=30.0, tau=4.0),
}
