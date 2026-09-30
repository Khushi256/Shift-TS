import Disclosure from '../shared/Disclosure';
import { MC_PASSES, summaryMetrics } from '../../api/inference';

export default function TechnicalDrawer({ rul, std, lower, upper }) {
  return (
    <div style={{ marginTop: 'var(--space-8)' }}>
      <Disclosure summary="Statistical detail">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 'var(--space-5)' }}>
          <Stat label="Point estimate (μ)" value={rul.toFixed(2)} unit="cycles" />
          <Stat label="Std deviation (σ)" value={std.toFixed(3)} unit="cycles" />
          <Stat label="Lower bound (2.5%)" value={Math.max(0, lower).toFixed(2)} unit="cycles" />
          <Stat label="Upper bound (97.5%)" value={upper.toFixed(2)} unit="cycles" />
          <Stat label={`MC Dropout passes (T = ${MC_PASSES})`} value={MC_PASSES} unit="passes" />
          <Stat label="Interval formula" value="μ ± 1.96σ" unit="" />
        </div>
        <p style={{ marginTop: 'var(--space-4)', fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)', lineHeight: 'var(--line-height-prose)' }}>
          Uncertainty is estimated via <strong>Monte Carlo Dropout</strong>: T&nbsp;=&nbsp;{MC_PASSES} stochastic
          forward passes are run at inference time with dropout layers kept active. The mean across
          passes is the point estimate; the standard deviation is a heuristic proxy for epistemic
          uncertainty. The displayed interval (μ&nbsp;±&nbsp;1.96σ) is a <em>nominal</em> 95%
          interval — it is not a statistically calibrated Bayesian credible interval, and empirical
          coverage is {summaryMetrics.targetCoverage95}% on held-out target engines (below the nominal 95% target).
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
