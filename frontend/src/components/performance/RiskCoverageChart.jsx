import {
  ResponsiveContainer, ComposedChart, Line, ReferenceLine,
  XAxis, YAxis, CartesianGrid, Tooltip,
} from 'recharts';
import './RiskCoverageChart.css';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="rc-chart__tooltip">
      <div className="rc-chart__tooltip-title">Coverage: {label}%</div>
      <div className="rc-chart__tooltip-row">
        MAE retained: <strong>{payload[0]?.value?.toFixed(2)} cycles</strong>
      </div>
    </div>
  );
};

export default function RiskCoverageChart({ data, baselineMae }) {
  if (!data?.length) return null;

  return (
    <div className="rc-chart">
      <div className="rc-chart__header">
        <h3 className="rc-chart__title">Uncertainty actually works.</h3>
        <p className="rc-chart__subtitle">
          When the model keeps only its most confident predictions (left side of the chart),
          error drops sharply — proving the uncertainty score is informative, not decorative.
          At 20% coverage, MAE falls below 2 cycles; the unfiltered baseline is{' '}
          {baselineMae?.toFixed(1)} cycles.
        </p>
      </div>

      <div className="rc-chart__container">
        <ResponsiveContainer width="100%" height={220}>
          <ComposedChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
            <title>Risk-Coverage Curve — selective prediction performance by uncertainty threshold</title>
            <desc>Line chart showing that retaining only high-confidence predictions dramatically reduces error. The x-axis is the fraction of predictions retained; the y-axis is mean absolute error on retained predictions.</desc>

            <CartesianGrid strokeDasharray="4 4" stroke="var(--color-border-subtle)" vertical={false} />

            <XAxis
              dataKey="coverage"
              tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: 'var(--color-border)' }}
              tickFormatter={v => `${v}%`}
              label={{ value: 'Predictions retained (%)', position: 'insideBottomRight', offset: -4, fill: 'var(--color-text-muted)', fontSize: 11 }}
            />

            <YAxis
              tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              label={{ value: 'MAE (cycles)', angle: -90, position: 'insideLeft', offset: 12, fill: 'var(--color-text-muted)', fontSize: 11 }}
            />

            <Tooltip content={<CustomTooltip />} />

            {baselineMae && (
              <ReferenceLine
                y={baselineMae}
                stroke="var(--color-critical)"
                strokeDasharray="4 3"
                strokeWidth={1}
                label={{ value: `Baseline ${baselineMae.toFixed(1)}`, position: 'insideTopRight', fill: 'var(--color-critical)', fontSize: 10 }}
              />
            )}

            <Line
              dataKey="mae"
              stroke="var(--color-data-primary)"
              strokeWidth={2}
              dot={false}
              name="Selective MAE"
              activeDot={{ r: 4, fill: 'var(--color-data-primary)', stroke: 'var(--color-bg-base)', strokeWidth: 2 }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
