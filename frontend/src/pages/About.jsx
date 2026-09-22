import PillarSection from '../components/about/PillarSection';
import Disclosure from '../components/shared/Disclosure';
import './About.css';

const PILLARS = [
  {
    plainTitle:     'Learn without labels',
    technicalTitle: 'Self-Supervised Learning — InfoNCE + Masked Reconstruction',
    layExplainer:   'The model learns the shape of healthy and degrading engine behaviour from raw sensor data, before ever seeing a single RUL label. Like learning to read a person\'s health from their gait before checking their chart.',
    technicalNote:  'Pre-training uses two complementary losses: InfoNCE (contrastive) and masked reconstruction. InfoNCE aligns representations of the same engine at nearby timesteps while pushing apart representations from different operating phases. Masked reconstruction forces the encoder to model temporal structure by recovering corrupted sensor channels. Together these initialise the GRU encoder in a basin that transfers across cohorts.',
  },
  {
    plainTitle:     'Adapt to new conditions',
    technicalTitle: 'Few-Shot Domain Adaptation — 1% Label Efficiency',
    layExplainer:   'When deployed on an unfamiliar engine fleet, the model recalibrates using as few as 1% of available labelled examples — roughly 1–2 engines out of 100. It does not retrain from scratch.',
    technicalNote:  'Adaptation fine-tunes only the regression head and the final GRU layer using a ProtoNet-style few-shot objective. The pre-trained encoder weights are frozen. This preserves the learned temporal representations while correcting for shift in the RUL regression surface. The target cohort in FD002 uses 6 operating conditions not present in the training split, simulating a genuine distribution shift.',
  },
  {
    plainTitle:     'Know what it doesn\'t know',
    technicalTitle: 'Uncertainty Estimation — Monte Carlo Dropout (T=20)',
    layExplainer:   'Every prediction includes a confidence range. When the range is wide, the model is telling you something — either the engine is in an unusual operating state, or the measurement is noisy.',
    technicalNote:  'MC Dropout performs T=20 stochastic forward passes with dropout rate p=0.2 active at inference. The mean across passes is the point estimate; the standard deviation is the uncertainty. This approximates the posterior predictive distribution of a deep Gaussian Process with the same architecture. The Spearman rank correlation between uncertainty and absolute error (ρ ≈ 0.22) confirms the uncertainty is informative. The risk-coverage curve demonstrates that retaining only high-confidence predictions drives MAE below 2 cycles.',
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
          In aircraft maintenance, it rarely does. Engines from different fleets, maintenance
          histories, or operating theatres produce sensor patterns a training-time model
          never encountered.
        </p>
        <p style={{ marginTop: 'var(--space-4)' }}>
          SHIFT-TS addresses this with three things working together: representation learning
          that generalises across conditions, fast adaptation to new fleets, and honest
          uncertainty so a maintenance planner knows when the prediction should not be trusted.
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
          <strong>NASA C-MAPSS FD002</strong> — the hardest of the four C-MAPSS subsets,
          with 6 operating conditions and the largest train/test split asymmetry.
          It contains 260 training engines and 259 test engines, each observed from
          healthy state to failure across up to 26 multivariate sensor channels.
        </p>
        <div className="about-page__dataset-facts">
          <Fact label="Operating conditions" value="6" />
          <Fact label="Sensor channels" value="26" />
          <Fact label="Training engines" value="260" />
          <Fact label="Target engines (unseen)" value="~39" />
        </div>
      </div>

      <div className="about-page__section">
        <Disclosure summary="Model architecture">
          <div>
            <p style={{ margin: 0, marginBottom: 'var(--space-4)' }}>
              The encoder is a single-layer GRU with hidden dimension 64, followed by a dropout layer
              (p=0.2) and a two-layer MLP regression head. Pre-training uses a dual-loss objective
              (InfoNCE + masked reconstruction) on sliding windows of length 30. Adaptation fine-tunes
              only the head and final GRU layer. Total parameters: ~142K. Training hardware: single GPU
              (CUDA, ≥8GB VRAM). Inference: CPU-compatible.
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
