import { Table, Thead, Tbody, Th, Td, Tr } from '../shared/Table';

const LABELS = {
  clean:            'Clean (baseline)',
  gaussian_noise:   'Gaussian noise',
  sensor_dropout:   'Sensor dropout',
  systematic_drift: 'Systematic drift',
  extreme_ops:      'Extreme operating state',
};

function getVerdict(delta) {
  if (delta < 0.5)  return { label: 'Resilient',  color: 'var(--color-healthy)' };
  if (delta < 5)    return { label: 'Moderate',   color: 'var(--color-warning)' };
  return              { label: 'Vulnerable', color: 'var(--color-critical)' };
}

export default function RobustnessTable({ results }) {
  const baseMae = results['clean']?.mae ?? 0;
  const entries = Object.entries(results);

  return (
    <div style={{ marginTop: 'var(--space-8)' }}>
      <h3 style={{ fontSize: 'var(--text-md)', fontWeight: 'var(--weight-medium)', marginBottom: 'var(--space-3)', color: 'var(--color-text-primary)' }}>
        Fault Impact Breakdown
      </h3>
      <Table caption="Fault tolerance metrics by perturbation type">
        <Thead>
          <Tr>
            <Th>Perturbation</Th>
            <Th align="right">MAE</Th>
            <Th align="right">RMSE</Th>
            <Th align="right">vs. clean</Th>
            <Th>Resilience</Th>
          </Tr>
        </Thead>
        <Tbody>
          {entries.map(([key, v]) => {
            const delta = v.mae - baseMae;
            const verdict = key === 'clean' ? null : getVerdict(delta);
            return (
              <Tr key={key}>
                <Td style={{ color: 'var(--color-text-primary)', fontWeight: 'var(--weight-medium)' }}>
                  {LABELS[key] ?? key}
                </Td>
                <Td align="right" mono>{v.mae.toFixed(2)}</Td>
                <Td align="right" mono>{v.rmse.toFixed(2)}</Td>
                <Td align="right" mono>
                  {key === 'clean'
                    ? <span style={{ color: 'var(--color-text-muted)' }}>—</span>
                    : <span style={{ color: delta > 5 ? 'var(--color-critical)' : delta > 2 ? 'var(--color-warning)' : 'var(--color-text-secondary)' }}>
                        +{delta.toFixed(2)}
                      </span>
                  }
                </Td>
                <Td>
                  {key === 'clean'
                    ? <span style={{ color: 'var(--color-text-muted)', fontSize: 'var(--text-xs)' }}>Baseline</span>
                    : <span style={{ color: verdict.color, fontSize: 'var(--text-xs)', fontWeight: 'var(--weight-semibold)' }}>
                        {verdict.label}
                      </span>
                  }
                </Td>
              </Tr>
            );
          })}
        </Tbody>
      </Table>
    </div>
  );
}
