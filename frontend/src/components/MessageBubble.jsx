/**
 * MessageBubble — Individual message display for user and AI messages.
 * Supports markdown rendering, image results with descriptions, and source tags.
 */
import './MessageBubble.css';
import { useState } from 'react';

/**
 * Simple markdown-to-HTML converter for AI messages.
 * Handles: newlines, ### headings, **bold**, `code`, [timestamps], bullet lists.
 */
function renderMarkdown(text) {
  if (!text) return '';

  const escaped = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  const lines = escaped.split('\n');
  const htmlLines = lines.map((line) => {
    // Handle ### headings
    const h3Match = line.match(/^###\s+(.+)/);
    if (h3Match) {
      return `<h4 class="msg__heading">${h3Match[1]}</h4>`;
    }

    const withInline = line
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\[(\d{1,2}:\d{2}(?:\s*[-–]\s*\d{1,2}:\d{2})?)\]/g, '<span class="msg__timestamp">[$1]</span>');

    if (/^\s*[-•]\s+/.test(withInline)) {
      return `<li>${withInline.replace(/^\s*[-•]\s+/, '')}</li>`;
    }
    if (/^\s*\d+\.\s+/.test(withInline)) {
      return `<li>${withInline.replace(/^\s*\d+\.\s+/, '')}</li>`;
    }
    if (!withInline.trim()) {
      return '';
    }
    return `<p>${withInline}</p>`;
  });

  const combined = htmlLines.join('\n').replace(/(?:<li>.*?<\/li>\n*)+/gs, (match) => `<ul>${match}</ul>`);
  return combined || '<p></p>';
}

export default function MessageBubble({ message, onClipClick }) {
  const isUser = message.role === 'user';
  const [brokenThumbs, setBrokenThumbs] = useState({});
  const [showMoreResults, setShowMoreResults] = useState(false);
  const validClips = (message.clips || []).filter((clip) => {
    const path = clip?.frame_paths?.[0];
    return path && !brokenThumbs[path];
  });

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

        {/* Text — render markdown for AI, plain text for user */}
        {isUser ? (
          <>
            {/* Show attached reference image in user message */}
            {message.imageUrl && (
              <div className="msg__user-image">
                <img
                  src={message.imageUrl}
                  alt={message.imageName || 'Reference image'}
                  className="msg__user-image-thumb"
                />
                <span className="msg__user-image-label">
                  📸 {message.imageName || 'Reference image'}
                </span>
              </div>
            )}
            <div className="msg__text">{message.text}</div>
          </>
        ) : (
          <div
            className="msg__text msg__text--markdown"
            dangerouslySetInnerHTML={{ __html: renderMarkdown(message.text) }}
          />
        )}

        {/* Image Results - Primary Display with Descriptions */}
        {validClips.length > 0 && (
          <div className="msg__image-results">
            <div className="msg__image-results-label">
              🖼️ Retrieved Frames ({validClips.length} match{validClips.length !== 1 ? 'es' : ''})
            </div>
            <div className="msg__image-cards">
              {validClips.slice(0, 2).map((clip, i) => (
                <div
                  key={`${clip.video_id}-${clip.start_time}-${i}`}
                  className="msg__image-card"
                  onClick={() => onClipClick?.(clip)}
                >
                  <div className="msg__image-card-visual">
                    <img
                      src={clip.frame_paths[0]}
                      alt={`Result ${i + 1}`}
                      className="msg__result-image"
                      onError={() => setBrokenThumbs((prev) => ({ ...prev, [clip.frame_paths[0]]: true }))}
                    />
                    <span className="msg__image-time">
                      {formatTimestamp(clip.start_time)}
                    </span>
                    <span className="msg__image-match-badge">
                      Match {i + 1}
                    </span>
                  </div>
                  {/* Description Panel */}
                  {clip.description && (
                    <div className="msg__image-card-desc">
                      <span className="msg__image-card-desc-icon">📝</span>
                      <p className="msg__image-card-desc-text">{clip.description}</p>
                    </div>
                  )}
                  <div className="msg__image-card-meta">
                    <span className="msg__image-card-vid">
                      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <rect x="2" y="4" width="20" height="16" rx="2" />
                        <polygon points="10,8 10,16 16,12" />
                      </svg>
                      {clip.video_id}
                    </span>
                    <span className="msg__image-card-range">
                      {formatTimestamp(clip.start_time)} – {formatTimestamp(clip.end_time)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Additional results beyond top 2 */}
        {validClips.length > 2 && (
          <div className="msg__more-results">
            <button 
              className="msg__more-toggle"
              onClick={() => setShowMoreResults(!showMoreResults)}
            >
              {showMoreResults ? '▲ Hide' : '▼ Show'} {validClips.length - 2} More Result{validClips.length - 2 !== 1 ? 's' : ''}
            </button>
            <div className="msg__clips-grid" style={{ display: showMoreResults ? 'flex' : 'none' }}>
              {validClips.slice(2).map((clip, i) => (
                <button
                  key={`${clip.video_id}-${clip.start_time}-${i + 2}`}
                  className="msg__clip-card"
                  onClick={() => onClipClick?.(clip)}
                  id={`clip-${message.id}-${i + 2}`}
                >
                  <img
                    src={clip.frame_paths[0]}
                    alt={`Clip frame ${i + 3}`}
                    className="msg__clip-thumb"
                    onError={() => setBrokenThumbs((prev) => ({ ...prev, [clip.frame_paths[0]]: true }))}
                  />
                  <div className="msg__clip-info">
                    <span className="msg__clip-time">
                      [{formatTimestamp(clip.start_time)} – {formatTimestamp(clip.end_time)}]
                    </span>
                    {clip.description && (
                      <span className="msg__clip-desc">{clip.description}</span>
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
