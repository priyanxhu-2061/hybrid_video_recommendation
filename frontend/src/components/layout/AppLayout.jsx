import { Link, Outlet } from 'react-router-dom';

export default function AppLayout() {
  return (
    <div className="shell">
      <header className="topbar">
        <Link to="/" className="brand">Hybrid Recommender</Link>
        <span className="topbar__note">MovieLens 20M · 5,000 users</span>
      </header>
      <Outlet />
    </div>
  );
}