# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

React (Vite, JSX) — `frontend/` directory. Python/Streamlit back-end (`app/streamlit_app.py`) replaced by this React front-end; the ML inference layer (PyTorch, src/) is retained as-is.

## Users

**Primary:** The researcher who builds and runs experiments day-to-day — needs fast prediction feedback, clean metrics visibility, and the ability to probe model behaviour under sensor faults.

**Secondary:** Technical evaluators (ML researchers, engineers) reviewing the research work — need architectural credibility, benchmark numbers, and calibration quality signals.

**Tertiary:** Non-ML stakeholders (demo viewers, academics) — need the domain story and operational significance without needing to know PyTorch internals.

## Product Purpose

SHIFT-TS predicts the Remaining Useful Life (RUL) of turbofan aircraft engines from multivariate sensor time-series, in the presence of operating-condition shifts, sensor noise, and limited labels. It extends a GRU-based RUL predictor with self-supervised pre-training, few-shot adaptation, and Monte Carlo Dropout uncertainty estimation. Success means a system that is both honest (calibrated uncertainty) and adaptive (works on unseen engine fleets).

## Positioning

The only engine prognostics demo that quantifies *when its own prediction cannot be trusted* — calibrated uncertainty that identifies unreliable predictions, not just a point estimate.

## Operating Context

- Runs locally against NASA C-MAPSS FD002 data; data must be present in `CMAPSSData/`
- Users select a sample engine from train / validation / target cohorts or upload a raw telemetry file
- Model checkpoint loaded from `models/baseline_best.pt`; absent checkpoint still loads (random weights, for dev)
- Multi-page application: Predict, Model Info, Robustness, About
- Target interaction: run a prediction, inspect RUL + uncertainty, optionally explore model metrics and fault tolerance

## Capabilities and Constraints

- **Model:** 2-layer GRU, hidden dim 64, MC Dropout with T=20 stochastic passes
- **Dataset:** NASA C-MAPSS FD002, 6 operating conditions, multivariate sensor sequences
- **Uncertainty:** 95% prediction intervals via ±1.96σ from MC Dropout variance
- **Robustness tests:** Gaussian noise, sensor dropout, systematic drift, extreme operating state
- **Input format:** 26-column NASA C-MAPSS .txt or .csv
- **Terminology to translate for non-ML users:** RUL, GRU, MC Dropout, self-supervised, InfoNCE, RMSE/MAE, ECE, risk-coverage curve

## Brand Commitments

Name: **SHIFT-TS** — preserve this exact spelling.
No logo, no colour brand constraints confirmed. Visual direction: dark, technical, minimal — not a generic SaaS dashboard.

## Evidence on Hand

- Trained checkpoint: `models/baseline_best.pt` (may or may not exist depending on run state)
- Benchmark numbers shown in UI: RMSE 20.35 cycles (zero-shot target), SSL loss 5.17 to 4.42, rho = 0.224 error-uncertainty correlation
- Dataset: NASA C-MAPSS FD002 in `CMAPSSData/`
- No real testimonials, no fabricated benchmarks permitted

## Product Principles

1. **Honest over impressive.** Uncertainty is a first-class output — never hide or downplay it.
2. **Legible at two depths.** Any non-ML visitor should understand the operational story; any ML engineer should find the architecture and numbers without digging.
3. **Data drives layout.** Charts and metrics are the product; chrome and decoration recede.
4. **Adaptation is the differentiator.** Distribution shift and few-shot adaptation must be explained, not buried in a tooltip.
5. **No invented claims.** All benchmark numbers must be real or explicitly labelled as placeholders.

## Accessibility & Inclusion

WCAG AA minimum. All interactive elements keyboard-navigable. Charts need text alternatives or captions sufficient for screen readers.
