import { Table, Thead, Tbody, Th, Td, Tr } from '../shared/Table';

export default function MetricsTable({ rmse, mae, coverage95, rho }) {
  const rows = [
    {
      metric:  'RMSE',
      value:   rmse != null ? rmse.toFixed(2) : '—',
      meaning: 'Root Mean Squared Error — average prediction error in cycles. Lower is better.',
      context: 'Full-target zero-shot evaluation across all 39 unseen target engines (6,939 windows).',
    },
    {
      metric:  'MAE',
      value:   mae != null ? mae.toFixed(2) : '—',
      meaning: 'Mean Absolute Error in cycles. Less sensitive to large outliers than RMSE.',
      context: 'Full-target zero-shot evaluation across all 39 unseen target engines (6,939 windows).',
    },
    {
      metric:  'Nominal 95% Interval Coverage',
      value:   coverage95 != null ? `${coverage95}%` : '—',
      meaning: 'Fraction of true RUL values that fall inside the nominal 95% interval (μ ± 1.96σ).',
      context: `Nominal target is 95%. Measured ${coverage95 ?? '—'}% confirms the interval is an uncertainty-aware heuristic spread, not an empirically calibrated guarantee.`,
    },
    {
      metric:  'Spearman ρ',
      value:   rho != null ? rho.toFixed(3) : '—',
      meaning: 'Rank correlation between MC Dropout uncertainty (σ) and absolute prediction error.',
      context: 'Positive ρ reflects that lower uncertainty → lower error at lower coverage under selective prediction.',
    },
  ];

  return (
    <div>
      <h3 style={{ fontSize: 'var(--text-md)', fontWeight: 'var(--weight-medium)', marginBottom: 'var(--space-3)', color: 'var(--color-text-primary)' }}>
        Benchmark Results — Target Cohort (Zero-Shot)
      </h3>
      <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)', marginBottom: 'var(--space-5)', lineHeight: 'var(--line-height-prose)' }}>
        All metrics are on the <strong>target cohort</strong> — engines the model never saw during training,
        evaluated without any target-label adaptation (<em>zero-shot transfer</em>).
        Few-shot head fine-tuning was also tested but did not improve on the zero-shot baseline.
      </p>
      <Table caption="Benchmark metrics on target cohort (zero-shot, unseen engines)">
        <Thead>
          <Tr>
            <Th>Metric</Th>
            <Th align="right">Value</Th>
            <Th>What it means</Th>
            <Th>Context</Th>
          </Tr>
        </Thead>
        <Tbody>
          {rows.map(row => (
            <Tr key={row.metric}>
              <Td>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-sm)', color: 'var(--color-text-primary)', fontVariantNumeric: 'tabular-nums' }}>
                  {row.metric}
                </span>
              </Td>
              <Td align="right" mono>{row.value}</Td>
              <Td>{row.meaning}</Td>
              <Td style={{ color: 'var(--color-text-muted)' }}>{row.context}</Td>
            </Tr>
          ))}
        </Tbody>
      </Table>
    </div>
  );
}
