import VideoCard from '@/components/video/VideoCard';
import { useFeed } from '@/hooks/useRecommendations';

export default function FeedGrid({ sessionId }) {
  const { data, isLoading, isError, refetch } = useFeed(20);

  if (isLoading) return <div className="feed-grid feed-grid--loading">Loading your feed</div>;

  if (isError) {
    return (
      <div className="feed-empty">
        <p>The feed did not load.</p>
        <button type="button" onClick={() => refetch()}>Try again</button>
      </div>
    );
  }

  if (!data.items.length) {
    return (
      <div className="feed-empty">
        <p>Nothing to show yet. Watch a few videos and this fills up.</p>
      </div>
    );
  }

  return (
    <section className="feed-grid" aria-label="Recommended for you">
      {data.items.map((item, index) => (
        <VideoCard
          key={item.video_id}
          video={item}
          position={index}
          sessionId={sessionId}
        />
      ))}
    </section>
  );
}
