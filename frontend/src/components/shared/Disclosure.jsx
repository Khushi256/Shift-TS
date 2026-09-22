import './Disclosure.css';

export default function Disclosure({ summary, children, defaultOpen = false }) {
  return (
    <details className="disclosure" open={defaultOpen}>
      <summary className="disclosure__summary">
        {summary}
        <span className="disclosure__icon" aria-hidden="true">›</span>
      </summary>
      <div className="disclosure__body">
        {children}
      </div>
    </details>
  );
}
