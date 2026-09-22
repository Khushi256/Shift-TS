import './SkeletonBar.css';

export default function SkeletonBar({ height = 20, width = '100%', style = {} }) {
  return (
    <div
      className="skeleton-bar"
      style={{ height, width, ...style }}
      aria-hidden="true"
    />
  );
}

export function SkeletonBlock({ children }) {
  return <div className="skeleton-block" aria-busy="true" aria-label="Loading">{children}</div>;
}
