import Disclosure from '../shared/Disclosure';

export default function TechnicalDrawer({ rul, std, lower, upper, mcPasses }) {
  return (
    <div style={{ marginTop: 'var(--space-8)' }}>
      <Disclosure summary="Statistical detail">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 'var(--space-5)' }}>
          <Stat label="Point estimate (μ)" value={rul.toFixed(2)} unit="cycles" />
          <Stat label="Std deviation (σ)" value={std.toFixed(3)} unit="cycles" />
          <Stat label="Lower bound (2.5%)" value={Math.max(0, lower).toFixed(2)} unit="cycles" />
          <Stat label="Upper bound (97.5%)" value={upper.toFixed(2)} unit="cycles" />
          <Stat label="MC Dropout passes (T)" value={mcPasses} unit="passes" />
          <Stat label="Interval formula" value="μ ± 1.96σ" unit="" />
        </div>
        <p style={{ marginTop: 'var(--space-4)', fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)', lineHeight: 'var(--line-height-prose)' }}>
          Uncertainty is estimated via Monte Carlo Dropout: T stochastic forward passes are
          performed with dropout active at inference time. The standard deviation across these
          passes approximates the predictive uncertainty of a deep Gaussian Process posterior,
          without requiring expensive Bayesian MCMC sampling.
        </p>
      </Disclosure>
    </div>
  );
}

function Stat({ label, value, unit }) {
  return (
    <div>
      <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)', marginBottom: 2 }}>{label}</div>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-sm)', color: 'var(--color-text-primary)', fontVariantNumeric: 'tabular-nums' }}>
        {value} <span style={{ color: 'var(--color-text-muted)', fontFamily: 'var(--font-ui)' }}>{unit}</span>
      </div>
    </div>
  );
}
