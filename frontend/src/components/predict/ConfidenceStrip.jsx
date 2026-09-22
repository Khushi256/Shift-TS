import './ConfidenceStrip.css';

/**
 * Confidence strip: visual representation of prediction uncertainty.
 * A narrow fill = HIGH confidence (tight interval).
 * A wide fill = LOW confidence (spread interval).
 * Inverted from a typical progress bar.
 */
export default function ConfidenceStrip({ std, mcPasses = 20 }) {
  // Map std → confidence score 0–1
  // std < 5 → very high confidence, std > 20 → low confidence
  const maxStd = 22;
  const confidence = Math.max(0, Math.min(1, 1 - std / maxStd));
  const pct = Math.round(confidence * 100);

  const tier =
    confidence > 0.72 ? { label: 'High', color: 'healthy' } :
    confidence > 0.44 ? { label: 'Moderate', color: 'warning' } :
                        { label: 'Low', color: 'critical' };

  const tooltip = `Based on ${mcPasses} prediction passes. σ = ${std.toFixed(2)} cycles. ${
    tier.label === 'High'     ? 'Tight spread — prediction is reliable.' :
    tier.label === 'Moderate' ? 'Moderate spread — treat with some caution.' :
                                 'Wide spread — model is uncertain; interpret carefully.'
  }`;

  return (
    <div className="confidence-strip" title={tooltip}>
      <div className="confidence-strip__header">
        <span className="confidence-strip__label">Model confidence</span>
        <span className={`confidence-strip__tier confidence-strip__tier--${tier.color}`}>
          {tier.label}
        </span>
      </div>

      <div className="confidence-strip__track" role="meter" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100} aria-label={`Model confidence: ${pct}%`}>
        <div
          className={`confidence-strip__fill confidence-strip__fill--${tier.color}`}
          style={{ width: `${pct}%` }}
        />
      </div>

      <p className="confidence-strip__description">
        {tier.label === 'High'
          ? `The model ran ${mcPasses} passes and found consistent results. This is a reliable estimate.`
          : tier.label === 'Moderate'
          ? `The model ran ${mcPasses} passes with moderate variation. Consider the 95% interval when planning.`
          : `The model ran ${mcPasses} passes with high variation. The prediction interval is wide — this engine may be in an unusual operating state.`}
      </p>
    </div>
  );
}
