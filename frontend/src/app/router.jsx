import { createBrowserRouter } from 'react-router-dom';

import AppLayout from '@/components/layout/AppLayout';
import HomePage from '@/pages/HomePage';
import WatchPage from '@/pages/WatchPage';

export const router = createBrowserRouter([
  {
    element: <AppLayout />,
    children: [
      { path: '/', element: <HomePage /> },
      { path: '/watch/:videoId', element: <WatchPage /> },
    ],
  },
]);