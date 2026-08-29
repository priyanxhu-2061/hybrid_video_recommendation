import { useEffect, useRef } from 'react';
import { logImpression } from '@/api/interactions';

/**
 * Fires one impression per card, once it has been at least half visible for a
 * second. The backend needs these negatives - without them the reranker trains
 * on clicks only and learns that everything shown was good.
 */
export function useImpressionTracking({ videoId, position, source, sessionId }) {
  const ref = useRef(null);
  const sent = useRef(false);

  useEffect(() => {
    const node = ref.current;
    if (!node || sent.current) return undefined;

    let timer;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !sent.current) {
          timer = setTimeout(() => {
            sent.current = true;
            logImpression({ videoId, position, source, sessionId });
            observer.disconnect();
          }, 1000);
        } else {
          clearTimeout(timer);
        }
      },
      { threshold: 0.5 }
    );

    observer.observe(node);
    return () => {
      clearTimeout(timer);
      observer.disconnect();
    };
  }, [videoId, position, source, sessionId]);

  return ref;
}
