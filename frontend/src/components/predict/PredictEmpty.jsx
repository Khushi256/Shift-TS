import './PredictEmpty.css';

export default function PredictEmpty() {
  return (
    <div className="predict-empty" role="region" aria-label="Prediction guide">
      <p className="predict-empty__statement">
        Select an engine above and predict its remaining useful life.
      </p>
      <p className="predict-empty__explainer">
        <strong>Remaining Useful Life (RUL)</strong> is the number of operating cycles before an engine
        is expected to reach its failure threshold — the point at which maintenance
        can no longer be deferred. Every prediction here includes a confidence range
        that tells you how much to trust the number.
      </p>
      <div className="predict-empty__hint">
        <span className="predict-empty__hint-label">Try</span>
        Target cohort — Engine #15 — Predict RUL
      </div>
    </div>
  );
}
