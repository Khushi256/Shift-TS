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
    detail: 'Held-out fleet engines. Evaluated under zero-shot transfer (0% target labels). Also tested in few-shot ablation.',
    color: 'healthy',
    highlight: true,
  },
];

export default function DistributionDiagram() {
  return (
    <div className="dist-diagram" style={{ marginTop: 'var(--space-12)' }}>
      <h3 style={{ fontSize: 'var(--text-md)', fontWeight: 'var(--weight-medium)', marginBottom: 'var(--space-2)', color: 'var(--color-text-primary)' }}>
        Cohort Split & Zero-Shot Transfer Evaluation
      </h3>
      <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)', marginBottom: 'var(--space-6)', lineHeight: 'var(--line-height-prose)' }}>
        NASA C-MAPSS FD002 features complex sensor dynamics across 6 operating conditions. The dataset is partitioned
        at the engine level into training (70%), validation (15%), and held-out target engines (15%, ~39 engines).
        Evaluation on the target cohort measures <strong>zero-shot transfer to unseen engines</strong> operating across dynamic multi-condition profiles.
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

      <div style={{ marginTop: 'var(--space-6)', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 'var(--space-4)', fontSize: 'var(--text-xs)', color: 'var(--color-text-secondary)', lineHeight: '1.6' }}>
        <div style={{ padding: 'var(--space-3)', background: 'var(--color-surface-subtle)', borderRadius: 6, border: '1px solid var(--color-border)' }}>
          <strong style={{ color: 'var(--color-healthy)' }}>Zero-Shot Transfer (Deployment Mode):</strong>
          <p style={{ margin: 'var(--space-1) 0 0 0' }}>
            Target engines are completely unseen by the model; evaluated directly without any target-domain label adaptation.
          </p>
        </div>
        <div style={{ padding: 'var(--space-3)', background: 'var(--color-surface-subtle)', borderRadius: 6, border: '1px solid var(--color-border)' }}>
          <strong style={{ color: 'var(--color-warning)' }}>Few-Shot Adaptation (Ablation Study):</strong>
          <p style={{ margin: 'var(--space-1) 0 0 0' }}>
            A small fraction (1%, 5%, 20%) of labeled target-domain engine trajectories is used for adaptation under different fine-tuning regimes: fully fine-tuned, partially unfrozen, and frozen-encoder configurations.
          </p>
        </div>
      </div>
    </div>
  );
}
