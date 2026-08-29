import { useRef } from 'react';

/**
 * A session id groups impressions into one displayed list, which is what the
 * reranker trains on. Kept in memory only.
 */
export function useSessionId() {
  const id = useRef(null);
  if (!id.current) id.current = crypto.randomUUID();
  return id.current;
}
