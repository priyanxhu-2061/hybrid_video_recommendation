import { Link } from 'react-router-dom';

const SOURCE_LABELS = {
  content: 'content',
  collaborative: 'CF',
  item_knn: 'co-watch',
  trending: 'popular',
};

export default function VideoCard({ video, position }) {
  const sources = video.sources || [];

  return (
    <article className="card">
      <Link to={`/watch/${video.video_id}`} className="card__link">
        <span className="card__rank">{position + 1}</span>
        <h3 className="card__title">{video.title || `#${video.video_id}`}</h3>
      </Link>

      {video.category && <span className="card__category">{video.category}</span>}

      {/* Multiple sources means several retrievers independently proposed this
          film. That agreement is why rank fusion ranked it where it did, so
          showing it makes the ranking legible rather than magical. */}
      <div className="card__sources">
        {sources.map((s) => (
          <span key={s} className={`chip chip--${s}`}>
            {SOURCE_LABELS[s] || s}
          </span>
        ))}
      </div>

      {video.explanation && <p className="card__reason">{video.explanation}</p>}

      {video.tags?.length > 0 && (
        <p className="card__tags">{video.tags.slice(0, 4).join(' · ')}</p>
      )}
    </article>
  );
}