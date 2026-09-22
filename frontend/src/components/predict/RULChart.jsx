import {
  ResponsiveContainer, ComposedChart, Line, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from 'recharts';
import './RULChart.css';

function buildChartData(cycles, means, stds, ruls) {
  return cycles.map((c, i) => {
    const lower = Math.max(0, means[i] - 1.96 * stds[i]);
    const upper = means[i] + 1.96 * stds[i];
    return {
      cycle: c,
      predicted: parseFloat(means[i].toFixed(1)),
      lower:     parseFloat(lower.toFixed(1)),
      upper:     parseFloat(upper.toFixed(1)),
      trueRul:   ruls && ruls[i] != null ? parseFloat(ruls[i].toFixed(1)) : undefined,
    };
  });
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  return (
    <div className="rul-chart__tooltip">
      <div className="rul-chart__tooltip-title">Cycle {label}</div>
      <div className="rul-chart__tooltip-row">
        <span className="rul-chart__tooltip-dot rul-chart__tooltip-dot--predicted" />
        Predicted: <strong>{d?.predicted} cycles</strong>
      </div>
      <div className="rul-chart__tooltip-row rul-chart__tooltip-row--muted">
        95% interval: [{d?.lower} – {d?.upper}]
      </div>
      {d?.trueRul != null && (
        <div className="rul-chart__tooltip-row rul-chart__tooltip-row--true">
          <span className="rul-chart__tooltip-dot rul-chart__tooltip-dot--true" />
          True RUL: {d.trueRul}
        </div>
      )}
    </div>
  );
};

export default function RULChart({ cycles, means, stds, ruls }) {
  const data = buildChartData(cycles, means, stds, ruls);
  const hasTrue = ruls && ruls.some(v => v != null);

  return (
    <div className="rul-chart">
      <div className="rul-chart__header">
        <h3 className="rul-chart__title">RUL Degradation Trajectory</h3>
        <span className="rul-chart__subtitle">
          Predicted remaining cycles over the engine's operating history · 95% credible interval
        </span>
      </div>

      <div className="rul-chart__container">
        <ResponsiveContainer width="100%" height={260}>
          <ComposedChart data={data} margin={{ top: 12, right: 24, left: 6, bottom: 16 }}>
            <title>RUL Degradation Trajectory — Predicted remaining useful life with 95% confidence interval</title>
            <desc>Line chart showing predicted remaining cycles over operating history. The shaded band shows the 95% prediction interval from Monte Carlo Dropout uncertainty estimation.</desc>

            <CartesianGrid
              strokeDasharray="4 4"
              stroke="var(--color-border-subtle)"
              vertical={false}
            />

            <XAxis
              dataKey="cycle"
              tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: 'var(--color-border)' }}
              dy={4}
              label={{ value: 'Operating Cycle', position: 'insideBottomRight', offset: -8, fill: 'var(--color-text-muted)', fontSize: 11 }}
            />

            <YAxis
              domain={[0, 'auto']}
              tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              label={{ value: 'Cycles Remaining', angle: -90, position: 'insideLeft', offset: 12, fill: 'var(--color-text-muted)', fontSize: 11 }}
            />

            <Tooltip content={<CustomTooltip />} />

            {/* 95% PI band */}
            <Area
              dataKey="upper"
              stroke="none"
              fill="var(--color-data-band)"
              legendType="none"
              name="95% interval upper"
              activeDot={false}
            />
            <Area
              dataKey="lower"
              stroke="none"
              fill="var(--color-bg-base)"
              legendType="none"
              name="95% interval lower"
              activeDot={false}
            />

            {/* True RUL */}
            {hasTrue && (
              <Line
                dataKey="trueRul"
                stroke="var(--color-data-true)"
                strokeWidth={1.5}
                strokeDasharray="6 4"
                dot={false}
                name="True RUL"
                activeDot={false}
              />
            )}

            {/* Predicted mean */}
            <Line
              dataKey="predicted"
              stroke="var(--color-data-primary)"
              strokeWidth={2}
              dot={false}
              name="Predicted RUL"
              activeDot={{ r: 4, fill: 'var(--color-data-primary)', stroke: 'var(--color-bg-base)', strokeWidth: 2 }}
            />

            <Legend
              wrapperStyle={{ fontSize: 11, color: 'var(--color-text-muted)', paddingTop: 8 }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
