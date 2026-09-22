import { Table, Thead, Tbody, Th, Td, Tr } from '../shared/Table';

const METRICS = [
  {
    metric:  'RMSE',
    value:   '20.35',
    meaning: 'Average prediction error in cycles. Lower is better.',
    context: 'On completely unseen target engines (zero-shot transfer).',
  },
  {
    metric:  'MAE',
    value:   '15.80',
    meaning: 'Mean absolute error in cycles. Measures average magnitude of prediction errors.',
    context: 'Target cohort (unseen engines withheld from training).',
  },
  {
    metric:  'NLL',
    value:   '3.47',
    meaning: 'Negative Log-Likelihood — measures how well the uncertainty matches actual errors. Lower is better.',
    context: 'Well-calibrated models score lower; overconfident models score high.',
  },
  {
    metric:  'ECE',
    value:   '0.082',
    meaning: 'Expected Calibration Error — how often the model\'s confidence matches observed accuracy. Closer to 0 is better.',
    context: 'A value of 0.082 means the model\'s confidence intervals are reasonably trustworthy.',
  },
];

export default function MetricsTable() {
  return (
    <div>
      <h3 style={{ fontSize: 'var(--text-md)', fontWeight: 'var(--weight-medium)', marginBottom: 'var(--space-3)', color: 'var(--color-text-primary)' }}>
        Benchmark Results — Target Cohort
      </h3>
      <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)', marginBottom: 'var(--space-5)', lineHeight: 'var(--line-height-prose)' }}>
        These results are on the target cohort — engines the model never saw during training.
      </p>
      <Table caption="Benchmark metrics on target cohort (unseen engines)">
        <Thead>
          <Tr>
            <Th>Metric</Th>
            <Th align="right">Value</Th>
            <Th>What it means</Th>
            <Th>Context</Th>
          </Tr>
        </Thead>
        <Tbody>
          {METRICS.map(row => (
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
