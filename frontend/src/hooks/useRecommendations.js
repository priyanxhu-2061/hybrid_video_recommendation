import { useQuery } from '@tanstack/react-query';
import { getFeed, getSimilar } from '@/api/recommendations';

export const useFeed = (topK = 20) =>
  useQuery({ queryKey: ['feed', topK], queryFn: () => getFeed(topK) });

export const useSimilar = (videoId, topK = 10) =>
  useQuery({
    queryKey: ['similar', videoId, topK],
    queryFn: () => getSimilar(videoId, topK),
    enabled: Boolean(videoId),
  });
