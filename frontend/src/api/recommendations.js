import { client } from './client';

// Every call takes user_id explicitly - this backend has no auth, and the
// demo needs to switch users freely anyway.

export const getUsers = async (limit = 50) => {
  const { data } = await client.get('/users', { params: { limit } });
  return data.user_ids;
};

export const getHistory = async (userId, limit = 12) => {
  const { data } = await client.get(`/users/${userId}/history`, { params: { limit } });
  return data.items;
};

export const getFeed = async (userId, { topK = 20, diversify = true } = {}) => {
  const { data } = await client.get('/recommendations/feed', {
    params: { user_id: userId, top_k: topK, diversify },
  });
  return data;
};

export const getSimilar = async (videoId, topK = 10) => {
  const { data } = await client.get(`/recommendations/similar/${videoId}`, {
    params: { top_k: topK },
  });
  return data;
};

export const getVideo = async (videoId) => {
  const { data } = await client.get(`/videos/${videoId}`);
  return data;
};