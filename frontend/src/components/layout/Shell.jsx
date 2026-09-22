import { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import Sidebar from './Sidebar';
import TopBar from './TopBar';
import './Shell.css';

export default function Shell({ children, modelStatus = 'dev' }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();

  // Close mobile drawer on route change
  useEffect(() => {
    setSidebarOpen(false);
  }, [location.pathname]);

  // Collapse sidebar on tablet (769px to 1024px)
  useEffect(() => {
    const mq = window.matchMedia('(min-width: 769px) and (max-width: 1024px)');
    const handler = (e) => setCollapsed(e.matches);
    setCollapsed(mq.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  return (
    <div className="shell">
      {/* Mobile top bar */}
      <TopBar onMenuOpen={() => setSidebarOpen(true)} />

      {/* Sidebar */}
      <Sidebar
        collapsed={collapsed}
        mobileOpen={sidebarOpen}
        onMobileClose={() => setSidebarOpen(false)}
        modelStatus={modelStatus}
      />

      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="shell__overlay"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Content */}
      <main className={`shell__main ${collapsed ? 'shell__main--rail' : ''}`} id="main-content">
        <div className="shell__content">
          {children}
        </div>
      </main>
    </div>
  );
}
