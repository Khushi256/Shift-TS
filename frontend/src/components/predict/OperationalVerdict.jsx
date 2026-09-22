import './OperationalVerdict.css';

const VERDICTS = {
  healthy: {
    label: 'Healthy',
    action: 'No immediate inspection required. Continue routine flight scheduling.',
    color: 'healthy',
  },
  warning: {
    label: 'Schedule Maintenance',
    action: 'Plan a depot overhaul within the next 20–30 operating cycles.',
    color: 'warning',
  },
  critical: {
    label: 'Ground Immediately',
    action: 'Engine is approaching its failure threshold. Take unit offline for overhaul.',
    color: 'critical',
  },
};

function getVerdict(rul) {
  if (rul > 80) return 'healthy';
  if (rul > 30) return 'warning';
  return 'critical';
}

export default function OperationalVerdict({ rul }) {
  const key = getVerdict(rul);
  const v   = VERDICTS[key];

  return (
    <div
      className={`verdict verdict--${v.color}`}
      role="status"
      aria-live="polite"
      aria-label={`Operational verdict: ${v.label}`}
    >
      <div className="verdict__header">
        <span className={`verdict__dot verdict__dot--${v.color}`} aria-hidden="true" />
        <span className="verdict__label">{v.label}</span>
      </div>
      <p className="verdict__action">{v.action}</p>
    </div>
  );
}
