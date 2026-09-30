# SHIFT-TS Frontend

Interactive research workstation for the SHIFT-TS turbofan engine RUL prediction framework.

## Stack

- **React 18** + **Vite** (JSX, no TypeScript)
- **Vanilla CSS** with a custom dark-mode design token system (`src/tokens/`)
- **Recharts** for time-series and risk-coverage visualisation
- **Lucide React** for iconography

## Pages

| Route | Description |
|-------|-------------|
| `/` | Overview — Prognostics pipeline, system workstations, dataset facts |
| `/predict` | Engine selector, RUL prediction, MC Dropout uncertainty bands |
| `/performance` | Benchmark metrics, few-shot ablation table, risk-coverage chart |
| `/robustness` | Sensor perturbation results (noise, dropout, drift, extreme op-state) |
| `/about` | Architecture dossier, design decisions, dataset provenance |

## Data Source

All displayed metrics are loaded from `src/api/model_real_data.json` — a single source of truth populated from real model runs (`models/uncertainty_target.npz`, `models/robustness_results.npz`). **No numbers are hardcoded in components.**

Key empirical values:
- Target MAE: **18.77** / RMSE: **23.91** (zero-shot, all 39 target engines)
- Nominal 95% PI coverage: **55.2%** (MC Dropout, T=20)
- Spearman ρ: **0.135** (uncertainty vs. absolute error, target set)
- Few-shot: **17.91 MAE** at 1%, **17.71 MAE** at 20% (fully fine-tuned)

## Development

```bash
# Install dependencies
npm install

# Start dev server (http://localhost:5173)
npm run dev

# Production build
npm run build
```
