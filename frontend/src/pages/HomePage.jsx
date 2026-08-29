import { useEffect, useState } from 'react';

import VideoCard from '@/components/video/VideoCard';
import { getFeed, getHistory, getUsers } from '@/api/recommendations';

export default function HomePage() {
  const [users, setUsers] = useState([]);
  const [userId, setUserId] = useState(null);
  const [diversify, setDiversify] = useState(true);

  const [feed, setFeed] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Pull the valid user ids once at startup. Without this you would be
  // guessing which of MovieLens's 138,000 ids happen to be in the sampled
  // subset the model was trained on.
  useEffect(() => {
    getUsers(50)
      .then((ids) => {
        setUsers(ids);
        setUserId(ids[0]);
      })
      .catch(() =>
        setError('Cannot reach the API. Is uvicorn running on port 8000?')
      );
  }, []);

  // Refetch whenever the user or the diversify toggle changes. Both the feed
  // and the history are needed together - the whole point is reading one
  // against the other.
  useEffect(() => {
    if (!userId) return;

    setLoading(true);
    setError(null);

    Promise.all([getFeed(userId, { diversify }), getHistory(userId, 12)])
      .then(([feedData, historyData]) => {
        setFeed(feedData);
        setHistory(historyData);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [userId, diversify]);

  return (
    <main className="page">
      <header className="controls">
        <label>
          User
          <select
            value={userId ?? ''}
            onChange={(e) => setUserId(Number(e.target.value))}
          >
            {users.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
        </label>

        {/* The demo's centrepiece. Flipping this re-runs MMR and the category
            caps, and the feed reorders in front of you. */}
        <label className="toggle">
          <input
            type="checkbox"
            checked={diversify}
            onChange={(e) => setDiversify(e.target.checked)}
          />
          Diversify (MMR + category caps)
        </label>

        {feed && (
          <span className="meta">
            {feed.strategy} · {feed.history_size} rated · model{' '}
            {feed.model_version}
          </span>
        )}
      </header>

      {history.length > 0 && (
        <section className="history">
          <h2>What this user has watched</h2>
          <p className="history__list">
            {history
              .map((h) => h.title)
              .filter(Boolean)
              .join(' · ')}
          </p>
          <p className="history__note">
            Compare this against the feed below. Metrics tell you the average is
            fine; only reading one user's history next to their recommendations
            tells you whether it is sensible for a real person.
          </p>
        </section>
      )}

      <h2>Recommended</h2>

      {error && <p className="error">{error}</p>}
      {loading && <p className="muted">Loading…</p>}

      {feed && !loading && (
        <section className="grid">
          {feed.items.map((item, i) => (
            <VideoCard key={item.video_id} video={item} position={i} />
          ))}
        </section>
      )}
    </main>
  );
}