# PK/PD Simulator

A small, readable **pharmacokinetic / pharmacodynamic (PK/PD) simulator** in Python.
It covers the classic analytical models taught in pharmacology and turns them into
clean, tested code and visual notebooks.

> **Part of a pharmacy / computational-medicine series.** Project 1: ADMET / toxicity
> prediction. Project 2 (this repo): PK/PD simulation.

## What it does

- **One-compartment models**: IV bolus and first-order oral absorption (Bateman function)
- **Two-compartment model**: IV bolus with distribution and elimination phases
- **Repeated dosing**: superposition, accumulation ratio, steady-state average concentration
- **Pharmacodynamics**: sigmoid Emax model (dose-effect relationship)
- **Worked examples**: paracetamol, amoxicillin, morphine (illustrative parameters)
- **Interactive app**: Streamlit sliders to explore parameters live *(coming soon)*

## Repository structure

```
pharmacokinetics-pharmacodynamics/
├── src/pkpd/
│   ├── models.py        # analytical PK/PD models
│   └── drugs.py         # illustrative drug parameters
├── notebooks/
│   ├── 01_one_compartment.ipynb
│   └── 02_two_compartment.ipynb
├── tests/
│   └── test_models.py   # sanity checks (half-life, AUC, tmax, accumulation...)
├── figures/
├── environment.yml
└── requirements.txt
```

## Quick start

```bash
git clone https://github.com/AymaneAmri-MD/pharmacokinetics-pharmacodynamics.git
cd pharmacokinetics-pharmacodynamics

# with conda
conda env create -f environment.yml
conda activate pkpd-simulator

# or with pip
pip install -r requirements.txt

pytest                 # run the tests
jupyter lab            # open the notebooks
```

## Key equations

| Quantity | Formula |
|---|---|
| Elimination rate constant | `k = CL / V` |
| Half-life | `t1/2 = ln(2) * V / CL` |
| IV bolus | `C(t) = (Dose / V) * exp(-k t)` |
| Oral (1-compartment) | `C(t) = F*Dose*ka / (V*(ka-k)) * (exp(-k t) - exp(-ka t))` |
| Time of peak | `tmax = ln(ka/k) / (ka - k)` |
| AUC (0 to infinity) | `AUC = F * Dose / CL` |
| Accumulation ratio | `R = 1 / (1 - exp(-k * tau))` |
| Average steady state | `Css,avg = F * Dose / (CL * tau)` |
| Sigmoid Emax | `E = E0 + Emax * C^n / (EC50^n + C^n)` |

## Notebooks

| # | Notebook | Topic |
|---|---|---|
| 01 | `01_one_compartment.ipynb` | IV bolus, oral absorption, repeated dosing |
| 02 | `02_two_compartment.ipynb` | Biexponential decay, ODE validation, peeling method |
| 03 | *planned* | PD Emax and PK/PD link |

## Roadmap

- [x] Core analytical models + tests
- [x] Notebook 01: one-compartment models
- [x] Notebook 02: two-compartment model
- [ ] Notebook 03: PD Emax / dose-response
- [ ] Streamlit interactive app
- [ ] Results figures in `figures/`

## Disclaimer

Drug parameters are rounded, textbook-style approximations for a typical adult. They are
for **education and demonstration only** and must not be used for clinical dosing decisions.

## Author

**Aymane Amri Amine**, Pre-Med / Pharmacy student with a background in biomedical
engineering and AI. [GitHub](https://github.com/AymaneAmri-MD)

## License

MIT
