import { Link } from 'react-router-dom';
import { Menu, Activity } from 'lucide-react';
import './TopBar.css';

export default function TopBar({ onMenuOpen }) {
  return (
    <header className="topbar" role="banner">
      <button
        className="topbar__menu"
        onClick={onMenuOpen}
        aria-label="Open navigation menu"
        aria-haspopup="true"
      >
        <Menu size={20} />
      </button>
      <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: '8px', textDecoration: 'none' }} aria-label="SHIFT-TS Home">
        <Activity size={18} style={{ color: 'var(--color-signal)' }} aria-hidden="true" />
        <span className="topbar__wordmark">SHIFT-TS</span>
      </Link>
    </header>
  );
}
