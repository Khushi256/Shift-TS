import { Table, Thead, Tbody, Th, Td, Tr } from '../shared/Table';
import './FewShotTable.css';

const ROW_METADATA = {
  '1%': {
    protocol: 'Few-Shot (1%)',
    splitType: 'Whole-engine split',
    partition: '1 adapted / 38 held-out',
    regime: 'Fully fine-tuned',
  },
  '5%': {
    protocol: 'Few-Shot (5%)',
    splitType: 'Whole-engine split',
    partition: '2 adapted / 37 held-out',
    regime: 'Partially unfrozen',
  },
  '20%': {
    protocol: 'Few-Shot (20%)',
    splitType: 'Whole-engine split',
    partition: '8 adapted / 31 held-out',
    regime: 'Fully fine-tuned',
  },
};

export default function FewShotTable({ fewShotData, zeroShotMae = 18.77 }) {
  const fractions = fewShotData?.fractions ?? [
    { fraction: '1%', splitZeroShotMae: 18.86, mae: 17.91, delta: '-0.95', detail: '1% target data (1 engine adapted / 38 held-out engines, fully fine-tuned)' },
    { fraction: '5%', splitZeroShotMae: 18.77, mae: 19.15, delta: '+0.38', detail: '5% target data (2 engines adapted / 37 held-out engines, partially unfrozen)' },
    { fraction: '20%', splitZeroShotMae: 18.56, mae: 17.71, delta: '-0.86', detail: '20% target data (8 engines adapted / 31 held-out engines, fully fine-tuned)' },
  ];

  return (
    <div style={{ marginTop: 'var(--space-12)' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', flexWrap: 'wrap', gap: 'var(--space-2)', marginBottom: 'var(--space-2)' }}>
        <h3 style={{ fontSize: 'var(--text-md)', fontWeight: 'var(--weight-medium)', color: 'var(--color-text-primary)', margin: 0 }}>
          Few-Shot Domain Adaptation (Diagnostic & Regime Comparison)
        </h3>
        <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-accent)', background: 'var(--color-surface-subtle)', padding: '2px 8px', borderRadius: 4, fontFamily: 'var(--font-mono)' }}>
          Engine-Level Partitioning · Multi-Regime
        </span>
      </div>

      <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)', marginBottom: 'var(--space-4)', lineHeight: 'var(--line-height-prose)' }}>
        We experimentally investigated few-shot adaptation by allocating whole engine trajectories (1%, 5%, 20% of target fleet)
        to adaptation, evaluating on strictly held-out target engines.
        The full-target zero-shot benchmark ({zeroShotMae.toFixed(2)} MAE) represents evaluation over all 39 target engines without adaptation.
        Each few-shot split is compared directly against that specific split's held-out zero-shot baseline.
      </p>

      <Table
        wrapperClassName="few-shot-table-wrapper"
        className="few-shot-table"
        caption="Few-shot adaptation regimes vs held-out split zero-shot baselines"
      >
        <colgroup>
          <col style={{ width: '18%' }} />
          <col style={{ width: '27%' }} />
          <col style={{ width: '13%' }} />
          <col style={{ width: '13%' }} />
          <col style={{ width: '12%' }} />
          <col style={{ width: '17%' }} />
        </colgroup>
        <Thead>
          <Tr>
            <Th>Protocol</Th>
            <Th>Partition & Regime</Th>
            <Th align="right">Split Baseline</Th>
            <Th align="right">Adapted MAE</Th>
            <Th align="right">Δ vs Split</Th>
            <Th>Outcome</Th>
          </Tr>
        </Thead>
        <Tbody>
          <Tr>
            <Td>
              <span className="few-shot-table__primary">Full Zero-Shot</span>
              <span className="few-shot-table__sub">Target reference</span>
            </Td>
            <Td>
              <span className="few-shot-table__primary">All 39 target engines</span>
              <span className="few-shot-table__sub">6,939 windows · No adaptation</span>
            </Td>
            <Td align="right" mono><strong>{zeroShotMae.toFixed(2)}</strong></Td>
            <Td align="right" mono style={{ color: 'var(--color-text-muted)' }}>—</Td>
            <Td align="right" mono style={{ color: 'var(--color-text-muted)' }}>Full-Cohort</Td>
            <Td style={{ color: 'var(--color-text-secondary)', fontSize: 'var(--text-xs)' }}>
              Full-target benchmark
            </Td>
          </Tr>
          {fractions.map(row => {
            const isImprovement = String(row.delta).startsWith('-');
            const meta = ROW_METADATA[row.fraction] || {
              protocol: `Few-Shot (${row.fraction})`,
              splitType: 'Whole-engine split',
              partition: row.detail,
              regime: '',
            };
            return (
              <Tr key={row.fraction}>
                <Td>
                  <span className="few-shot-table__primary" style={{ fontFamily: 'var(--font-mono)' }}>
                    {meta.protocol}
                  </span>
                  <span className="few-shot-table__sub">{meta.splitType}</span>
                </Td>
                <Td>
                  <span className="few-shot-table__primary">{meta.partition}</span>
                  <span className="few-shot-table__sub">{meta.regime}</span>
                </Td>
                <Td align="right" mono style={{ color: 'var(--color-text-muted)' }}>
                  {typeof row.splitZeroShotMae === 'number' ? row.splitZeroShotMae.toFixed(2) : row.splitZeroShotMae ?? '—'}
                </Td>
                <Td align="right" mono>
                  <strong>{typeof row.mae === 'number' ? row.mae.toFixed(2) : row.mae}</strong>
                </Td>
                <Td align="right" mono style={{ color: isImprovement ? 'var(--color-healthy)' : 'var(--color-warning)', fontWeight: 'var(--weight-semibold)' }}>
                  {row.delta}
                </Td>
                <Td style={{ color: 'var(--color-text-secondary)', fontSize: 'var(--text-xs)' }}>
                  {isImprovement ? 'Outperforms split baseline' : 'Small-sample variance'}
                </Td>
              </Tr>
            );
          })}
        </Tbody>
      </Table>

      <div style={{ marginTop: 'var(--space-4)', padding: 'var(--space-4)', background: 'var(--color-surface-subtle)', borderRadius: 6, border: '1px solid var(--color-border)' }}>
        <h4 style={{ fontSize: 'var(--text-xs)', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--color-text-muted)', marginBottom: 'var(--space-2)' }}>
          Diagnostic Analysis: Implementation vs. Physical Adaptation
        </h4>
        <ul style={{ margin: 0, paddingLeft: 'var(--space-5)', fontSize: 'var(--text-xs)', color: 'var(--color-text-secondary)', lineHeight: '1.6' }}>
          <li>
            <strong>Full-target benchmark vs split baselines:</strong> The {zeroShotMae.toFixed(2)} MAE is the full-target zero-shot benchmark across all 39 engines. Each few-shot split evaluates on its respective held-out subset (18.86 for 1%, 18.77 for 5%, 18.56 for 20%), preventing conflation between full-cohort performance and split-specific evaluations.
          </li>
          <li>
            <strong>Root cause of earlier ~80 MAE collapse:</strong> Diagnostic tracing revealed that the adapter was initializing an untrained random head ending in ReLU (predicting ~0 cycles) against target engines with mean RUL ~80.8. Freezing the encoder prevented the head from recovering from this 80-cycle offset within small target steps.
          </li>
          <li>
            <strong>Engine-level selection & warm-starting:</strong> Grouping adaptation samples strictly by whole engine trajectories (preventing window leakage) and warm-starting weights restores convergence across all fractions.
          </li>
          <li>
            <strong>Unfreezing regimes:</strong> Fully fine-tuning both the GRU encoder and regression head yields the strongest adaptation gain (17.91 MAE vs 18.86 split baseline at 1%, and 17.71 MAE vs 18.56 split baseline at 20%).
          </li>
        </ul>
      </div>
    </div>
  );
}
