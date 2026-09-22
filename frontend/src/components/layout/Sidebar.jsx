import { NavLink } from 'react-router-dom';
import { Activity, Home, Gauge, BarChart2, Shield, Info, X } from 'lucide-react';
import StatusDot from '../shared/StatusDot';
import './Sidebar.css';

const NAV_ITEMS = [
  { to: '/',            icon: Home,      label: 'Overview' },
  { to: '/predict',     icon: Gauge,     label: 'Predict' },
  { to: '/performance', icon: BarChart2, label: 'Performance' },
  { to: '/robustness',  icon: Shield,    label: 'Robustness' },
  { to: '/about',       icon: Info,      label: 'About' },
];

export default function Sidebar({
  collapsed,
  mobileOpen,
  onMobileClose,
  modelStatus = 'dev',
}) {
  // Mobile drawer mode must never display as a collapsed rail; it should always show full labels and brand
  const isRail = collapsed && !mobileOpen;

  return (
    <nav
      className={[
        'sidebar',
        isRail ? 'sidebar--rail' : '',
        mobileOpen ? 'sidebar--open' : '',
      ].filter(Boolean).join(' ')}
      aria-label="Main navigation"
    >
      {/* Close button (mobile only) */}
      <button
        className="sidebar__close"
        onClick={onMobileClose}
        aria-label="Close navigation"
      >
        <X size={18} />
      </button>

      {/* Wordmark */}
      <div className="sidebar__header">
        <NavLink
          to="/"
          onClick={onMobileClose}
          className="sidebar__wordmark"
          aria-label="SHIFT-TS Home"
        >
          <span className="sidebar__logo" aria-hidden="true">
            <Activity size={20} strokeWidth={1.8} />
          </span>
          {!isRail && (
            <div className="sidebar__brand">
              <span className="sidebar__name">SHIFT-TS</span>
              <span className="sidebar__tagline">RUL Prediction</span>
            </div>
          )}
        </NavLink>
      </div>

      <div className="sidebar__divider" />

      {/* Nav items */}
      <ul className="sidebar__nav" role="list">
        {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
          <li key={to}>
            <NavLink
              to={to}
              end={to === '/'}
              onClick={onMobileClose}
              title={label}
              className={({ isActive }) =>
                ['sidebar__item', isActive ? 'sidebar__item--active' : ''].filter(Boolean).join(' ')
              }
            >
              <span className="sidebar__item-icon" aria-hidden="true">
                <Icon size={18} strokeWidth={1.8} />
              </span>
              {!isRail && (
                <span className="sidebar__item-label">{label}</span>
              )}
            </NavLink>
          </li>
        ))}
      </ul>

      {/* Footer: model status */}
      {!isRail && (
        <div className="sidebar__footer">
          <StatusDot status={modelStatus} />
        </div>
      )}
    </nav>
  );
}
