/**
 * VideoPlayer — Lightweight player for clip playback with frame gallery.
 */
import { useRef } from 'react';
import './VideoPlayer.css';

export default function VideoPlayer({ clip, onClose }) {
  const videoRef = useRef(null);

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

      {/* Real Video Player */}
      <div className="player__viewport">
        <video 
          ref={videoRef}
          key={clip.video_id + clip.start_time}
          src={`/videos/${clip.video_id}/play#t=${clip.start_time},${clip.end_time || clip.start_time + 5}`}
          controls
          autoPlay
          className="player__video"
          style={{ width: '100%', height: '100%', objectFit: 'contain' }}
        >
          Your browser does not support the video tag.
        </video>
      </div>

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
      {frames.length > 0 && (
        <div className="player__thumbstrip">
          {frames.map((framePath, i) => {
            const match = framePath.match(/frame_(\d+)/);
            const ts = match ? parseInt(match[1], 10) : clip.start_time;
            return (
              <button
                key={i}
                className="player__thumb"
                title={`Jump to ${formatTime(ts)}`}
                onClick={() => {
                  if (videoRef.current) {
                    videoRef.current.currentTime = ts;
                    videoRef.current.play();
                  }
                }}
              >
                <img src={framePath} alt={`Frame at ${formatTime(ts)}`} />
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
