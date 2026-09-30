import PillarSection from '../components/about/PillarSection';
import Disclosure from '../components/shared/Disclosure';
import { MC_PASSES, summaryMetrics } from '../api/inference';
import './About.css';

const PILLARS = [
  {
    plainTitle:     'Learn without labels',
    technicalTitle: 'Self-Supervised Learning — InfoNCE + Masked Reconstruction',
    layExplainer:   'The model learns temporal representations of engine sensor behaviour and degradation patterns from raw sensor data, before ever seeing a single RUL label.',
    technicalNote:  'Pre-training uses two complementary losses: InfoNCE (contrastive) and masked reconstruction. InfoNCE brings representations of two augmented views of the same temporal window closer while separating representations from other windows. Masked reconstruction forces the encoder to model temporal structure by recovering corrupted sensor channels. Together these initialise the GRU encoder in a latent space that transfers effectively to unseen engines.',
  },
  {
    plainTitle:     'Few-shot adaptation — diagnostic investigation',
    technicalTitle: 'Few-Shot Domain Adaptation — Regime Analysis',
    layExplainer:   'We evaluated whether adapting to unfamiliar engine fleets using a small number of target-domain labels outperforms zero-shot transfer. Investigating the pipeline revealed that head-only adaptation overfits small samples, whereas full model fine-tuning with engine-level partitioning successfully improves performance over zero-shot transfer.',
    technicalNote:  `We investigated three adaptation regimes: frozen encoder + head, partially unfrozen top GRU layer, and fully fine-tuned encoder. When partitioned strictly by engine and warm-started from the supervised model with AdamW regularization: Full-target zero-shot MAE is ${summaryMetrics.targetMae.toFixed(2)} (with held-out split baselines of 18.86 for 1%, 18.77 for 5%, and 18.56 for 20%), while 1% adaptation achieves 17.91 MAE (fully fine-tuned), 5% achieves 19.15 MAE (partially unfrozen), and 20% achieves 17.71 MAE (fully fine-tuned). This diagnostic demonstrated that the earlier ~80 MAE collapse was an implementation artifact (uninitialized head predicting ~0 cycles against target mean RUL ~80.8), while full fine-tuning provides genuine adaptation gains on held-out engines.`,
  },
  {
    plainTitle:     'Know what it doesn\'t know',
    technicalTitle: `Uncertainty Estimation — Monte Carlo Dropout (T=${MC_PASSES})`,
    layExplainer:   'Every prediction includes a nominal predictive interval. When the interval is wide, the model indicates elevated uncertainty — either the engine is in an unfamiliar operating state, or sensor noise is high.',
    technicalNote:  `MC Dropout performs T=${MC_PASSES} stochastic forward passes with dropout rate p=0.2 active at inference. The mean across passes is the point estimate; the standard deviation is an uncertainty-aware heuristic proxy — not a statistically calibrated Bayesian posterior. The Spearman rank correlation between uncertainty and absolute error (ρ ≈ ${summaryMetrics.spearmanRho}) shows a positive association between uncertainty and prediction error: lower uncertainty → lower error at lower coverage under selective prediction.`,
  },
];

export default function About() {
  return (
    <div className="about-page">
      {/* Opening statement IS the h1 */}
      <h1 className="about-page__statement">
        SHIFT-TS predicts how long an aircraft engine will last — and tells you when that prediction might be wrong.
      </h1>

      <div className="about-page__section">
        <p>
          Traditional models assume the deployment environment looks like training data.
          In turbofan prognostics, operating regimes vary widely. Different operating conditions
          can produce sensor patterns that differ from those seen during training.
        </p>
        <p style={{ marginTop: 'var(--space-4)' }}>
          SHIFT-TS addresses this with three things working together: representation learning
          that generalises across conditions, rigorous zero-shot transfer evaluation on unseen engines, and honest
          uncertainty estimation so a maintenance planner knows when the prediction should not be trusted.
        </p>
      </div>

      <div className="about-page__pillars">
        {PILLARS.map((p, i) => (
          <PillarSection key={p.technicalTitle} index={i + 1} {...p} />
        ))}
      </div>

      <div className="about-page__section about-page__section--dataset">
        <h2 className="about-page__section-title">Dataset</h2>
        <p>
          All results are on{' '}
          <strong>NASA C-MAPSS FD002</strong> — a multi-condition C-MAPSS subset with six operating conditions.
          It contains 260 training engines and 259 test engines, each observed from
          healthy state to failure across 21 multivariate sensor channels and 3 operational settings (24 model input features total).
        </p>
        <div className="about-page__dataset-facts">
          <Fact label="Operating conditions" value="6" />
          <Fact label="Sensors + settings" value="21 + 3" />
          <Fact label="Training engines" value="260" />
          <Fact label="Target engines (unseen)" value="~39" />
        </div>
      </div>

      <div className="about-page__section">
        <Disclosure summary="Model architecture">
          <div>
            <p style={{ margin: 0, marginBottom: 'var(--space-4)' }}>
              The encoder is a 2-layer GRU with hidden dimension 64, followed by a dropout layer
              (p=0.2) and a two-layer MLP regression head. Pre-training uses a dual-loss objective
              (InfoNCE + masked reconstruction) on sliding windows of length 30. Adaptation evaluates
              multiple fine-tuning regimes, including frozen-encoder, partially unfrozen, and fully
              fine-tuned configurations. Total parameters: ~142K.
              Training hardware: single GPU (CUDA, ≥8GB VRAM). Inference: CPU-compatible.
            </p>
          </div>
        </Disclosure>
      </div>
    </div>
  );
}

function Fact({ label, value }) {
  return (
    <div className="about-fact">
      <div className="about-fact__value">{value}</div>
      <div className="about-fact__label">{label}</div>
    </div>
  );
}
