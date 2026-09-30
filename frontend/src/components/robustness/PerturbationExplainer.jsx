import './PerturbationExplainer.css';

const PERTURBATIONS = [
  {
    key:     'gaussian_noise',
    label:   'Gaussian Noise (σ = 0.05)',
    analogy: 'Random sensor jitter with standard deviation σ = 0.05 relative to normalized feature range.',
    impact:  'Tests whether the model can ignore low-level noise without losing predictive accuracy.',
  },
  {
    key:     'sensor_dropout',
    label:   'Sensor Dropout (1 channel zeroed)',
    analogy: 'A random sensor channel goes offline and returns zeros — simulating a physical telemetry failure.',
    impact:  'Tests whether the model degrades gracefully when input channels disappear.',
  },
  {
    key:     'systematic_drift',
    label:   'Systematic Drift (10% ramp)',
    analogy: 'A sensor linearly drifts upward by +0.10 across the window — simulating calibration loss over time.',
    impact:  'Tests whether the model is robust to biased sensors, common in ageing hardware.',
  },
  {
    key:     'extreme_ops',
    label:   'Extreme Operating State (Regime: 1.0)',
    analogy: 'Flight operating conditions forced to upper boundary extremes (1.0 in normalized space).',
    impact:  'Simulates an extreme operating-state shift beyond the observed normalized range.',
  },
];

export default function PerturbationExplainer() {
  return (
    <div className="pert-explainer">
      <p className="pert-explainer__intro">
        Select an engine below to inject each fault type and measure how the prediction degrades.
        Each perturbation simulates a realistic failure mode.
      </p>
      <div className="pert-explainer__grid">
        {PERTURBATIONS.map(p => (
          <div key={p.key} className="pert-card">
            <div className="pert-card__label">{p.label}</div>
            <div className="pert-card__analogy">{p.analogy}</div>
            <div className="pert-card__impact">{p.impact}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
