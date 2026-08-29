import { client } from './client';

// Fire-and-forget: never block playback on telemetry.
export const logEvent = (payload) =>
  client.post('/interactions', payload).catch(() => {});

export const logImpression = ({ videoId, position, source, sessionId }) =>
  logEvent({
    video_id: videoId,
    event_type: 'impression',
    position,
    source,
    session_id: sessionId,
  });

export const logWatchProgress = ({ videoId, watchSeconds, completionRatio, sessionId }) =>
  logEvent({
    video_id: videoId,
    event_type: 'view',
    watch_seconds: watchSeconds,
    completion_ratio: completionRatio,
    session_id: sessionId,
  });
