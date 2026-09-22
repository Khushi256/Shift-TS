import { useEffect, useRef } from 'react';
import ConfidenceStrip from './ConfidenceStrip';
import OperationalVerdict from './OperationalVerdict';
import RULChart from './RULChart';
import TechnicalDrawer from './TechnicalDrawer';
import './PredictResult.css';

export default function PredictResult({ data, onReset }) {
  const { rul, lower, upper, std, cycles, means, stds, ruls, sourceLabel, timestamp, mcPasses } = data;
  const panelRef = useRef(null);

  // Move focus here after prediction completes
  useEffect(() => {
    panelRef.current?.focus();
  }, []);

  return (
    <section
      ref={panelRef}
      className="predict-result"
      tabIndex={-1}
      aria-label="Prediction results"
    >
      {/* Result header row */}
      <div className="predict-result__meta">
        <span className="predict-result__source">{sourceLabel}</span>
        <span className="predict-result__timestamp">{timestamp}</span>
        <button className="predict-result__reset" onClick={onReset}>
          Change engine →
        </button>
      </div>

      {/* Primary reading: RUL number + interval */}
      <div className="predict-result__primary">
        <div>
          <div className="predict-result__rul-label">Remaining Useful Life</div>
          <div className="predict-result__rul" aria-label={`${Math.round(rul)} cycles remaining`}>
            {Math.round(rul)}
            <span className="predict-result__rul-unit">cycles</span>
          </div>
          <div className="predict-result__interval" aria-label={`95% interval: ${Math.round(lower)} to ${Math.round(upper)} cycles`}>
            95% interval&nbsp;
            <span className="predict-result__interval-range">
              [{Math.round(lower)} – {Math.round(upper)}]
            </span>
          </div>
        </div>
      </div>

      {/* Confidence strip */}
      <ConfidenceStrip std={std} mcPasses={mcPasses} />

      {/* Operational verdict */}
      <OperationalVerdict rul={rul} />

      {/* Chart */}
      <RULChart cycles={cycles} means={means} stds={stds} ruls={ruls} />

      {/* Technical drawer */}
      <TechnicalDrawer rul={rul} std={std} lower={lower} upper={upper} mcPasses={mcPasses} />
    </section>
  );
}
