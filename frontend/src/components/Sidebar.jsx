/**
 * Sidebar — Video library, upload button, and chat history.
 */
import { useState, useEffect } from 'react';
import { getVideos } from '../api/client';
import StatusBadge from './StatusBadge';
import './Sidebar.css';

export default function Sidebar({
  selectedVideoId,
  onSelectVideo,
  onUploadClick,
  refreshKey,
  collapsed,
  onToggleCollapse,
}) {
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const fetchVideos = async () => {
      setLoading(true);
      try {
        const data = await getVideos();
        if (!cancelled) setVideos(data);
      } catch (err) {
        console.error('Failed to fetch videos:', err);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    fetchVideos();
    return () => { cancelled = true; };
  }, [refreshKey]);

  const formatDuration = (seconds) => {
    if (!seconds) return '--:--';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  if (collapsed) {
    return (
      <div className="sidebar sidebar--collapsed">
        <button className="sidebar__toggle" onClick={onToggleCollapse} title="Expand sidebar">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M9 18l6-6-6-6" />
          </svg>
        </button>
        <button className="sidebar__toggle sidebar__upload-mini" onClick={onUploadClick} title="Upload video">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 5v14M5 12h14" />
          </svg>
        </button>
      </div>
    );
  }

  return (
    <div className="sidebar">
      {/* Header */}
      <div className="sidebar__header">
        <div className="sidebar__brand">
          <div className="sidebar__logo">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <rect x="2" y="4" width="20" height="16" rx="3" stroke="var(--accent-primary)" strokeWidth="2" />
              <polygon points="10,8 10,16 16,12" fill="var(--accent-primary)" />
            </svg>
          </div>
          <div>
            <h1 className="sidebar__title">VideoRAG</h1>
            <p className="sidebar__subtitle">Forensic Analysis</p>
          </div>
        </div>
        <button className="sidebar__toggle" onClick={onToggleCollapse} title="Collapse sidebar">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M15 18l-6-6 6-6" />
          </svg>
        </button>
      </div>

      {/* Upload Button */}
      <div className="sidebar__actions">
        <button className="sidebar__upload-btn" onClick={onUploadClick} id="upload-button">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12" />
          </svg>
          Upload Evidence
        </button>
      </div>

      {/* All Videos button */}
      <div className="sidebar__section">
        <button
          className={`sidebar__all-btn ${!selectedVideoId ? 'sidebar__all-btn--active' : ''}`}
          onClick={() => onSelectVideo(null)}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8" />
            <path d="M21 21l-4.35-4.35" />
          </svg>
          Search All Videos
        </button>
      </div>

      {/* Video Library */}
      <div className="sidebar__section">
        <h3 className="sidebar__section-title">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M4 6h16M4 12h16M4 18h7" />
          </svg>
          Evidence Library
          <span className="sidebar__count">{videos.length}</span>
        </h3>

        <div className="sidebar__video-list">
          {loading ? (
            <div className="sidebar__loading">
              <div className="sidebar__loading-dot" />
              <div className="sidebar__loading-dot" />
              <div className="sidebar__loading-dot" />
            </div>
          ) : videos.length === 0 ? (
            <div className="sidebar__empty">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5">
                <rect x="2" y="4" width="20" height="16" rx="3" />
                <polygon points="10,8 10,16 16,12" fill="none" stroke="var(--text-muted)" />
              </svg>
              <p>No videos uploaded yet</p>
              <span>Click "Upload Evidence" to begin</span>
            </div>
          ) : (
            videos.map((video) => (
              <button
                key={video.video_id}
                className={`sidebar__video-item ${selectedVideoId === video.video_id ? 'sidebar__video-item--active' : ''}`}
                onClick={() => onSelectVideo(video.video_id)}
                id={`video-${video.video_id}`}
              >
                <div className="sidebar__video-thumb">
                  {video.thumbnail ? (
                    <img src={video.thumbnail} alt={video.filename} />
                  ) : (
                    <div className="sidebar__video-thumb-placeholder">
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <polygon points="10,8 10,16 16,12" />
                      </svg>
                    </div>
                  )}
                </div>
                <div className="sidebar__video-info">
                  <span className="sidebar__video-name" title={video.filename}>
                    {video.filename}
                  </span>
                  <div className="sidebar__video-meta">
                    <span>{formatDuration(video.duration)}</span>
                    {video.resolution && <span>{video.resolution}</span>}
                  </div>
                </div>
                <StatusBadge status={video.status} />
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
