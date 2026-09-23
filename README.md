# SHIFT-TS

### Self-Supervised and Uncertainty-Calibrated Adaptation for Time-Series under Limited Labels

> A research-grade ML framework and interactive application for **Remaining Useful Life (RUL) prediction** in aircraft turbofan engines. SHIFT-TS investigates how models can adapt to unseen operational conditions with minimal labeled data and reliably estimate prediction uncertainty under distribution shift.

---

## The Core Problem

Traditional Remaining Useful Life (RUL) models assume that training data and operational deployment conditions match. In mission-critical aerospace systems, operating regimes fluctuate, sensor channels degrade or experience noise, and newly introduced engine fleets may have little or no labeled failure trajectories.

Standard deep learning models frequently produce **confident but catastrophically incorrect predictions** under distribution shift.

**SHIFT-TS addresses three core research questions:**
1. **Self-Supervised Representation:** Can models learn rich degradation dynamics directly from raw, unlabelled multivariate sensor streams?
2. **Few-Shot Adaptation:** Can an agent adapt to completely unseen operational regimes using only **1% to 5%** of labeled target fleet data?
3. **Calibrated Epistemic Uncertainty:** Can Monte Carlo Dropout uncertainty quantify when a prediction cannot be trusted, catching high-error predictions before failure?

---

## System Architecture

```text
NASA C-MAPSS FD002 (21 Sensors, 3 Flight Settings, 6 Operating Regimes)
                             ↓
              Sliding Window Extraction (30 Cycles)
                             ↓
                 2-Layer Recurrent GRU Encoder
                             ↓
             ┌───────────────────────────────┐
             │   Self-Supervised Learning    │
             ├───────────────┬───────────────┤
             │ Masked Recon  │ Contrastive   │
             │ (Channel Loss)│ InfoNCE Loss  │
             └───────────────┴───────────────┘
                             ↓
       Few-Shot Domain Adaptation (1%, 5%, 20% Target Labels)
                             ↓
       Monte Carlo Dropout Inference (T=20 Stochastic Passes)
                             ↓
       RUL Point Estimate + 95% Credible Interval (±1.96σ)
```

---

## Key Benchmark Results

* **Dataset:** NASA C-MAPSS FD002 (6 operating regimes, 260 training engines, 259 test engines).
* **Target Fleet RMSE:** `20.35` cycles zero-shot under operational regime shift.
* **Uncertainty Calibration:** Spearman rank correlation $\rho \approx 0.224$ between predictive variance and true error (high uncertainty reliably signals high error).
* **Selective Prediction:** Risk–coverage analysis demonstrates that deferring high-uncertainty predictions reduces MAE to under `2.0` cycles.
* **Label Efficiency:** Few-shot adaptation restores baseline predictive fidelity using as little as 1% labeled target trajectories.

---

## Quickstart: How to Reproduce & Run

### Prerequisites
* **Python 3.9+**
* **Node.js 18+** & **npm**

---

### 1. Repository Setup

Clone the repository and enter the directory:
```bash
git clone https://github.com/Khushi256/Shift-TS.git
cd Shift-TS
```

---

### 2. Frontend Web Application (React + Vite)

The interactive workstation runs a client-side interface bundled with realistic C-MAPSS FD002 model trajectories, uncertainty bands, and robustness benchmarks:

```bash
# Enter frontend directory
cd frontend

# Install dependencies
npm install

# Start local development server
npm run dev
```

Open your browser at **`http://localhost:5173`**.

To test the production build:
```bash
npm run build
npm run preview
```

---

### 3. ML Pipeline & Experiment Reproduction (Python)

To run the PyTorch models, SSL pretraining, few-shot adaptation, or robustness suites:

#### Step A: Install Python Dependencies
From the repository root:
```bash
pip install -r requirements.txt
```

#### Step B: Run Verification & Smoke Tests
Verify all 6 core modules (data ingestion, baseline GRU, SSL heads, few-shot adaptation, MC dropout uncertainty, and 8-fault robustness suite):
```bash
python tests/smoke_test_all_cores.py
```

#### Step C: Run Experimental Pipelines
```bash
# 1. Train Baseline GRU model on training fleet
python experiments/run_baseline.py

# 2. Run Self-Supervised Learning (Masked Reconstruction + InfoNCE)
python experiments/run_ssl.py

# 3. Run Few-Shot Adaptation across 1%, 5%, and 20% label regimes
python experiments/run_fewshot.py

# 4. Run Monte Carlo Dropout uncertainty calibration & coverage analysis
python experiments/run_uncertainty.py

# 5. Run sensor fault resilience tests (Gaussian noise, dropouts, drift)
python experiments/run_robustness.py
```

## Project Structure

```text
SHIFT-TS/
├── CMAPSSData/                 # NASA C-MAPSS FD002 sensor telemetry files
├── data/                       # Cached tensors and operational splits
├── experiments/                # Research experiment scripts
│   ├── run_baseline.py         # Standard GRU training
│   ├── run_ssl.py              # Self-supervised pretraining
│   ├── run_fewshot.py          # 1%, 5%, 20% adaptation experiments
│   ├── run_uncertainty.py      # MC Dropout epistemic uncertainty & calibration
│   └── run_robustness.py       # Sensor noise and channel dropout testing
├── frontend/                   # React + Vite interactive research workstation
│   ├── src/
│   │   ├── api/                # Model inference data & telemetry feeds
│   │   ├── components/         # Workstation UI components & charts
│   │   ├── pages/              # Landing, Predict, Performance, Robustness, About
│   │   └── tokens/             # Research dark-mode design system tokens
│   ├── package.json
│   ├── vercel.json             # Vercel SPA routing configuration
│   └── vite.config.js
├── models/                     # PyTorch checkpoints (baseline_best.pt)
├── notebooks/                  # Exploratory data analysis notebooks
├── src/                        # Core ML library source code
│   ├── data/                   # Sequence windowing & normalization
│   ├── evaluation/             # Metrics: RMSE, MAE, ECE, Risk-Coverage, Rho
│   ├── models/                 # GRU encoder, SSL heads, MC Dropout
│   └── training/               # Losses, few-shot fine-tuning loops
├── tests/                      # Validation and smoke test suites
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Development Milestones

- [x] NASA C-MAPSS FD002 dataset pipeline & operational condition clustering
- [x] Sliding-window temporal feature extraction (30-cycle windows, 24 channels)
- [x] 2-Layer GRU baseline architecture with MC Dropout
- [x] Self-supervised pretraining (Masked sensor recovery + InfoNCE contrastive head)
- [x] Few-shot domain adaptation (1%, 5%, and 20% label regimes)
- [x] Calibrated epistemic uncertainty bounds ($\pm 1.96\sigma$, 95% CI)
- [x] Sensor perturbation test suite (8 synthetic noise & dropout modes)
- [x] React + Vite research workstation with real-time trajectory visualization
- [x] Production build verification & Vercel deployment setup
