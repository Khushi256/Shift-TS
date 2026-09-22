import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, ReferenceLine, Cell,
} from 'recharts';

const LABELS = {
  clean:            'Clean (baseline)',
  gaussian_noise:   'Gaussian noise',
  sensor_dropout:   'Sensor dropout',
  systematic_drift: 'Systematic drift',
  extreme_ops:      'Extreme ops',
};

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: 'var(--color-bg-elevated)',
      border: '1px solid var(--color-border)',
      borderRadius: 'var(--radius-md)',
      padding: 'var(--space-3) var(--space-4)',
      fontSize: 'var(--text-sm)',
      boxShadow: 'var(--shadow-md)',
    }}>
      <div style={{ color: 'var(--color-text-muted)', fontSize: 'var(--text-xs)', marginBottom: 4 }}>{label}</div>
      <div style={{ color: 'var(--color-text-primary)' }}>MAE: <strong>{payload[0]?.value?.toFixed(2)} cycles</strong></div>
    </div>
  );
};

export default function RobustnessChart({ results }) {
  const entries = Object.entries(results);
  const baseMae = results['clean']?.mae ?? 0;

  const data = entries.map(([key, v]) => ({
    name: LABELS[key] ?? key,
    key,
    mae: parseFloat(v.mae.toFixed(2)),
  }));

  return (
    <div style={{ marginTop: 'var(--space-8)' }}>
      <h3 style={{ fontSize: 'var(--text-md)', fontWeight: 'var(--weight-medium)', marginBottom: 'var(--space-2)', color: 'var(--color-text-primary)' }}>
        Prediction Error by Fault Type
      </h3>
      <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)', marginBottom: 'var(--space-5)', lineHeight: 'var(--line-height-prose)' }}>
        Mean Absolute Error (MAE) in cycles. The dashed line marks the clean baseline — lower is better.
      </p>
      <div style={{ border: '1px solid var(--color-border)', borderRadius: 'var(--radius-lg)', padding: 'var(--space-5) var(--space-4) var(--space-3)' }}>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
            <title>Robustness bar chart — MAE per fault type</title>
            <desc>Bar chart comparing prediction error across sensor fault types. The clean baseline is shown with a reference line.</desc>

            <CartesianGrid strokeDasharray="4 4" stroke="var(--color-border-subtle)" vertical={false} />

            <XAxis
              dataKey="name"
              tick={{ fill: 'var(--color-text-muted)', fontSize: 10 }}
              tickLine={false}
              axisLine={{ stroke: 'var(--color-border)' }}
            />

            <YAxis
              tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              label={{ value: 'MAE (cycles)', angle: -90, position: 'insideLeft', offset: 12, fill: 'var(--color-text-muted)', fontSize: 11 }}
            />

            <Tooltip content={<CustomTooltip />} />

            <ReferenceLine
              y={baseMae}
              stroke="var(--color-data-baseline)"
              strokeDasharray="4 3"
              strokeWidth={1}
              label={{ value: 'Clean baseline', position: 'insideTopRight', fill: 'var(--color-text-muted)', fontSize: 10 }}
            />

            <Bar dataKey="mae" radius={[3, 3, 0, 0]}>
              {data.map(d => (
                <Cell
                  key={d.key}
                  fill={d.key === 'clean' ? 'var(--color-data-baseline)' : 'var(--color-signal)'}
                  fillOpacity={d.key === 'clean' ? 0.5 : 0.75}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
