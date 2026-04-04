/**
 * MessageBubble — Individual message display for user and AI messages.
 */
import './MessageBubble.css';

export default function MessageBubble({ message, onClipClick }) {
  const isUser = message.role === 'user';

  const formatTimestamp = (seconds) => {
    if (seconds == null) return '';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div className={`msg ${isUser ? 'msg--user' : 'msg--ai'}`}>
      {/* Avatar */}
      {!isUser && (
        <div className="msg__avatar">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
            <rect x="2" y="4" width="20" height="16" rx="3" stroke="var(--accent-primary)" strokeWidth="2" />
            <polygon points="10,8 10,16 16,12" fill="var(--accent-primary)" />
          </svg>
        </div>
      )}

      <div className="msg__content">
        {/* Name */}
        <span className="msg__name">{isUser ? 'You' : 'Kubrick'}</span>

        {/* Text */}
        <div className="msg__text">{message.text}</div>

        {/* Source clips */}
        {message.clips && message.clips.length > 0 && (
          <div className="msg__clips">
            <span className="msg__clips-label">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="10,8 10,16 16,12" />
                <rect x="2" y="4" width="20" height="16" rx="2" />
              </svg>
              Referenced Clips
            </span>
            <div className="msg__clips-grid">
              {message.clips.map((clip, i) => (
                <button
                  key={i}
                  className="msg__clip-card"
                  onClick={() => onClipClick?.(clip)}
                  id={`clip-${message.id}-${i}`}
                >
                  {clip.frame_paths && clip.frame_paths[0] ? (
                    <img src={clip.frame_paths[0]} alt={`Clip frame`} className="msg__clip-thumb" />
                  ) : (
                    <div className="msg__clip-thumb msg__clip-thumb--placeholder">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <polygon points="10,8 10,16 16,12" />
                      </svg>
                    </div>
                  )}
                  <div className="msg__clip-info">
                    <span className="msg__clip-time">
                      [{formatTimestamp(clip.start_time)} – {formatTimestamp(clip.end_time)}]
                    </span>
                    {clip.description && (
                      <span className="msg__clip-desc">{clip.description.slice(0, 60)}</span>
                    )}
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Sources */}
        {message.sources && message.sources.length > 0 && (
          <div className="msg__sources">
            <span className="msg__sources-label">Sources</span>
            <div className="msg__source-tags">
              {[...new Set(message.sources.map((s) => s.source_index))].map((idx) => (
                <span key={idx} className={`msg__source-tag msg__source-tag--${idx}`}>
                  {idx === 'caption' && '👁️ Caption Index'}
                  {idx === 'image' && '🖼️ Visual Index'}
                  {idx === 'speech' && '🎤 Speech Index'}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
