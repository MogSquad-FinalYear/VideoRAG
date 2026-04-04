/**
 * VideoPlayer — Lightweight player for clip playback with frame gallery.
 */
import { useState } from 'react';
import './VideoPlayer.css';

export default function VideoPlayer({ clip, onClose }) {
  const [currentFrame, setCurrentFrame] = useState(0);

  if (!clip) return null;

  const formatTime = (seconds) => {
    if (seconds == null) return '0:00';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  const frames = clip.frame_paths || [];

  return (
    <div className="player">
      {/* Header */}
      <div className="player__header">
        <h3 className="player__title">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--accent-primary)" strokeWidth="2">
            <polygon points="10,8 10,16 16,12" />
            <rect x="2" y="4" width="20" height="16" rx="2" />
          </svg>
          Video Clip
        </h3>
        <button className="player__close" onClick={onClose} title="Close player" id="close-player">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      {/* Frame Display */}
      <div className="player__viewport">
        {frames.length > 0 ? (
          <img
            src={frames[currentFrame]}
            alt={`Frame at ${formatTime(clip.start_time + currentFrame)}`}
            className="player__frame"
          />
        ) : (
          <div className="player__placeholder">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5">
              <rect x="2" y="4" width="20" height="16" rx="3" />
              <polygon points="10,8 10,16 16,12" />
            </svg>
            <p>No frames available</p>
          </div>
        )}
      </div>

      {/* Timeline */}
      <div className="player__timeline">
        <span className="player__time">{formatTime(clip.start_time)}</span>
        <div className="player__progress">
          <div
            className="player__progress-fill"
            style={{
              width: frames.length > 1
                ? `${(currentFrame / (frames.length - 1)) * 100}%`
                : '100%',
            }}
          />
        </div>
        <span className="player__time">{formatTime(clip.end_time)}</span>
      </div>

      {/* Controls */}
      {frames.length > 1 && (
        <div className="player__controls">
          <button
            className="player__btn"
            onClick={() => setCurrentFrame((f) => Math.max(0, f - 1))}
            disabled={currentFrame === 0}
            title="Previous frame"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polygon points="19 20 9 12 19 4 19 20" />
              <line x1="5" y1="19" x2="5" y2="5" />
            </svg>
          </button>

          <span className="player__frame-count">
            Frame {currentFrame + 1} / {frames.length}
          </span>

          <button
            className="player__btn"
            onClick={() => setCurrentFrame((f) => Math.min(frames.length - 1, f + 1))}
            disabled={currentFrame === frames.length - 1}
            title="Next frame"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polygon points="5 4 15 12 5 20 5 4" />
              <line x1="19" y1="5" x2="19" y2="19" />
            </svg>
          </button>
        </div>
      )}

      {/* Clip Info */}
      <div className="player__info">
        <div className="player__info-row">
          <span className="player__info-label">Video ID</span>
          <span className="player__info-value">{clip.video_id}</span>
        </div>
        <div className="player__info-row">
          <span className="player__info-label">Time Range</span>
          <span className="player__info-value player__info-value--mono">
            {formatTime(clip.start_time)} – {formatTime(clip.end_time)}
          </span>
        </div>
        {clip.description && (
          <div className="player__info-row">
            <span className="player__info-label">Description</span>
            <span className="player__info-value">{clip.description}</span>
          </div>
        )}
      </div>

      {/* Frame Thumbnails */}
      {frames.length > 1 && (
        <div className="player__thumbstrip">
          {frames.map((framePath, i) => (
            <button
              key={i}
              className={`player__thumb ${i === currentFrame ? 'player__thumb--active' : ''}`}
              onClick={() => setCurrentFrame(i)}
            >
              <img src={framePath} alt={`Frame ${i + 1}`} />
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
