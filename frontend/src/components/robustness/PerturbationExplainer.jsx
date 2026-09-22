import './PerturbationExplainer.css';

const PERTURBATIONS = [
  {
    key:     'gaussian_noise',
    label:   'Gaussian Noise',
    analogy: 'Random sensor jitter — like electrical interference on a measurement cable.',
    impact:  'Tests whether the model can ignore low-level noise without losing predictive accuracy.',
  },
  {
    key:     'sensor_dropout',
    label:   'Sensor Dropout',
    analogy: 'A sensor goes offline and returns zeros — simulating a physical sensor failure.',
    impact:  'Tests whether the model degrades gracefully when input channels disappear.',
  },
  {
    key:     'systematic_drift',
    label:   'Systematic Drift',
    analogy: 'A sensor reads consistently high or low — like a calibration error that accumulates over time.',
    impact:  'Tests whether the model is robust to biased sensors, common in ageing hardware.',
  },
  {
    key:     'extreme_ops',
    label:   'Extreme Operating State',
    analogy: 'Operating conditions fall outside the range seen during training.',
    impact:  'Simulates a genuine distribution shift — the core challenge SHIFT-TS is designed to handle.',
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
