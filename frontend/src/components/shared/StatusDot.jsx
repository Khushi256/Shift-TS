import './StatusDot.css';

const STATUS_MAP = {
  healthy:  { color: 'healthy',  label: 'Healthy' },
  warning:  { color: 'warning',  label: 'Warning' },
  critical: { color: 'critical', label: 'Critical' },
  info:     { color: 'info',     label: 'Info' },
  trained:  { color: 'healthy',  label: 'Trained model' },
  dev:      { color: 'warning',  label: 'Development mode — no checkpoint' },
};

export default function StatusDot({ status = 'info', label, showLabel = true }) {
  const resolved = STATUS_MAP[status] || STATUS_MAP.info;
  const displayLabel = label ?? resolved.label;

  return (
    <span className={`status-dot status-dot--${resolved.color}`} title={displayLabel}>
      <span className="status-dot__dot" aria-hidden="true" />
      {showLabel && (
        <span className="status-dot__label">{displayLabel}</span>
      )}
    </span>
  );
}
