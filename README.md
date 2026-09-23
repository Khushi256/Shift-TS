# SHIFT-TS

### Self-Supervised and Uncertainty-Calibrated Adaptation for Time-Series under Limited Labels

> A research-oriented ML system for **Remaining Useful Life (RUL) prediction** that adapts to changing operating conditions and identifies unreliable predictions.

## Problem

Traditional RUL models assume that training and deployment data come from similar conditions. In real-world systems, operating conditions can change, sensors can become noisy or missing, and new environments may have very few labeled samples.

This can cause models to make **confident but incorrect predictions**.

SHIFT-TS investigates:

* How to learn useful representations from unlabeled sensor data
* How to adapt to unseen operating conditions with few labels
* How to estimate prediction uncertainty under distribution shift
* Whether uncertainty can identify unreliable predictions

## Dataset

**NASA C-MAPSS — FD002**

FD002 contains multivariate turbofan engine sensor time-series under multiple operating conditions, making it suitable for studying RUL prediction and environment shifts.

**Task:** Predict the Remaining Useful Life of an engine from historical sensor sequences.

**Run Frontend Locally**

```bash
cd frontend
npm run dev
```
> Runs at `http://localhost:5173`

## Approach

```text
C-MAPSS FD002
      ↓
Preprocessing & Temporal Windows
      ↓
GRU / Time-Aware GRU
      ↓
Self-Supervised Learning
 ┌────┴─────────────┐
 │ Masked           │ Contrastive
 │ Reconstruction   │ Learning
 └────┬─────────────┘
      ↓
Few-Shot Adaptation
      ↓
Uncertainty Estimation
      ↓
RUL + Reliability
```

### 1. Baseline

Build a GRU-based RUL predictor and establish MAE/RMSE baselines.

### 2. Self-Supervised Learning

Compare:

* Masked temporal reconstruction
* Contrastive learning
* Combined objectives

### 3. Few-Shot Adaptation

Train on known operating environments and adapt to an unseen environment using limited labeled data such as **1%, 5%, and 20%**.

### 4. Uncertainty

Estimate prediction uncertainty and investigate whether higher uncertainty corresponds to larger errors and distribution shifts.

### 5. Robustness

Evaluate the model under:

* Sensor noise
* Missing values
* Sensor dropout
* Changed operating conditions

## Evaluation

**Prediction**

* MAE
* RMSE

**Adaptation**

* Performance before/after adaptation
* Label efficiency
* Adaptation cost

**Uncertainty**

* NLL
* ECE
* Prediction interval coverage
* Risk–coverage analysis

## Tech Stack

| Category            | Technologies        |
| ------------------- | ------------------- |
| Language            | Python              |
| Deep Learning       | PyTorch             |
| ML                  | Scikit-learn        |
| Data                | NumPy, Pandas       |
| Visualization       | Matplotlib, Seaborn |
| Experiment Tracking | TensorBoard         |
| Web Application     | React (Vite)        |

## Project Structure

```text
SHIFT-TS/
├── CMAPSSData/
├── data/
├── experiments/
├── frontend/
├── models/
├── notebooks/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   └── training/
├── tests/
├── requirements.txt
└── README.md
```

## Development Roadmap

* [ ] Obtain C-MAPSS FD002
* [ ] Dataset exploration & preprocessing
* [ ] GRU baseline
* [ ] Time-aware modeling
* [ ] Masked reconstruction
* [ ] Contrastive learning
* [ ] Few-shot adaptation
* [ ] Uncertainty estimation
* [ ] Robustness experiments
* [ ] Streamlit demo
* [ ] Final experimental analysis

## Core Research Question

> **Can self-supervised temporal representations improve few-shot adaptation to unseen operating environments while maintaining useful uncertainty estimates under distribution shift?**

## Goal

SHIFT-TS aims to move beyond:

**"Can the model predict RUL accurately?"**

toward:

**"Can the model adapt when conditions change and recognize when its prediction may not be trustworthy?"**
