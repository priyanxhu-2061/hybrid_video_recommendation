import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';

import VideoCard from '@/components/video/VideoCard';
import { getSimilar, getVideo } from '@/api/recommendations';

export default function WatchPage() {
  const { videoId } = useParams();
  const [video, setVideo] = useState(null);
  const [similar, setSimilar] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    const id = Number(videoId);
    Promise.all([getVideo(id), getSimilar(id, 8)])
      .then(([v, s]) => {
        setVideo(v);
        setSimilar(s.items);
      })
      .catch(() => setError('This film is not in the content model.'));
  }, [videoId]);

  if (error) {
    return (
      <main className="page">
        <Link to="/" className="back">Back to feed</Link>
        <p className="error">{error}</p>
      </main>
    );
  }

  if (!video) return <main className="page"><p className="muted">Loading…</p></main>;

  return (
    <main className="page">
      <Link to="/" className="back">Back to feed</Link>

      <h1 className="detail__title">{video.title}</h1>
      <p className="detail__category">{video.category}</p>
      {video.tags?.length > 0 && (
        <p className="detail__tags">{video.tags.join(' · ')}</p>
      )}

      <h2>Similar films</h2>
      <p className="muted">
        Content-based only — no user, no history. These come from tag and genre
        similarity alone, which is why the model finds things like other films
        by the same director without ever being told who directed what.
      </p>

      <section className="grid">
        {similar.map((item, i) => (
          <VideoCard key={item.video_id} video={item} position={i} />
        ))}
      </section>
    </main>
  );
}