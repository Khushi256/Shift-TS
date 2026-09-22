import './DistributionDiagram.css';

const STEPS = [
  {
    label: 'Training cohort',
    pct:   '70%',
    detail: 'Engines seen during self-supervised pre-training and supervised RUL regression.',
    color: 'info',
  },
  {
    label: 'Validation cohort',
    pct:   '15%',
    detail: 'Used for early stopping and checkpoint selection. Not used for gradient updates.',
    color: 'warning',
  },
  {
    label: 'Target cohort',
    pct:   '15%',
    detail: 'Completely unseen engines. Used only for few-shot adaptation and zero-shot transfer evaluation.',
    color: 'healthy',
    highlight: true,
  },
];

export default function DistributionDiagram() {
  return (
    <div className="dist-diagram" style={{ marginTop: 'var(--space-12)' }}>
      <h3 style={{ fontSize: 'var(--text-md)', fontWeight: 'var(--weight-medium)', marginBottom: 'var(--space-2)', color: 'var(--color-text-primary)' }}>
        How distribution shift is tested
      </h3>
      <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)', marginBottom: 'var(--space-6)', lineHeight: 'var(--line-height-prose)' }}>
        NASA C-MAPSS FD002 contains 6 discrete operating conditions. Engines are split so that
        the target cohort is never visible to the model during training — simulating a real deployment
        on an unfamiliar engine fleet.
      </p>

      <div className="dist-diagram__steps">
        {STEPS.map((s, i) => (
          <div key={s.label} className={`dist-step ${s.highlight ? 'dist-step--highlight' : ''}`}>
            <div className="dist-step__pct" style={{ color: `var(--color-${s.color})` }}>{s.pct}</div>
            <div className="dist-step__label">{s.label}</div>
            <div className="dist-step__detail">{s.detail}</div>
            {i < STEPS.length - 1 && (
              <div className="dist-step__arrow" aria-hidden="true">→</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
