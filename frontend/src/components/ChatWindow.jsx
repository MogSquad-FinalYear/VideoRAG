/**
 * ChatWindow — Main chat interface with message input and history.
 */
import { useState, useRef, useEffect } from 'react';
import { sendMessage, sendMessageWithImage } from '../api/client';
import MessageBubble from './MessageBubble';
import './ChatWindow.css';

let msgIdCounter = 0;

export default function ChatWindow({ videoId, onClipClick }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [referenceImage, setReferenceImage] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const imageInputRef = useRef(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSend = async () => {
    const query = input.trim();
    if (!query || loading) return;

    const userMsg = {
      id: `msg-${++msgIdCounter}`,
      role: 'user',
      text: query,
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const result = referenceImage
        ? await sendMessageWithImage(query, referenceImage, videoId)
        : await sendMessage(query, videoId);

      const aiMsg = {
        id: `msg-${++msgIdCounter}`,
        role: 'assistant',
        text: result.answer,
        clips: result.clips || [],
        sources: result.sources || [],
      };

      setMessages((prev) => [...prev, aiMsg]);
      setReferenceImage(null);
    } catch (err) {
      const errorMsg = {
        id: `msg-${++msgIdCounter}`,
        role: 'assistant',
        text: `⚠️ Error: ${err.message}. Make sure the backend is running on port 8000.`,
        clips: [],
        sources: [],
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  // Fix 13: Clear chat
  const handleClearChat = () => {
    setMessages([]);
    msgIdCounter = 0;
    inputRef.current?.focus();
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat">
      {/* Header */}
      <div className="chat__header">
        <div className="chat__header-info">
          <h2 className="chat__header-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--accent-primary)" strokeWidth="2">
              <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
            </svg>
            Kubrick AI
          </h2>
          <p className="chat__header-subtitle">
            {videoId ? (
              <>Analyzing video <code>{videoId}</code></>
            ) : (
              'Searching across all evidence videos'
            )}
          </p>
        </div>
        <div className="chat__header-actions">
          {videoId && (
            <span className="chat__scope-badge">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="2" y="4" width="20" height="16" rx="2" />
                <polygon points="10,8 10,16 16,12" />
              </svg>
              Scoped to video
            </span>
          )}
          {messages.length > 0 && (
            <button
              className="chat__clear-btn"
              onClick={handleClearChat}
              title="Clear conversation"
              id="clear-chat-button"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="3 6 5 6 21 6" />
                <path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2" />
              </svg>
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="chat__messages" id="chat-messages">
        {messages.length === 0 ? (
          <div className="chat__welcome">
            <div className="chat__welcome-icon">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
                <rect x="2" y="4" width="20" height="16" rx="3" stroke="var(--accent-primary)" strokeWidth="1.5" />
                <polygon points="10,8 10,16 16,12" fill="var(--accent-primary)" opacity="0.6" />
              </svg>
            </div>
            <h3>Welcome to Kubrick</h3>
            <p>Your AI forensic video analyst. Ask questions about your evidence videos using natural language.</p>
            <div className="chat__suggestions">
              <button className="chat__suggestion" onClick={() => setInput('What objects appear in the video?')}>
                👁️ What objects appear in the video?
              </button>
              <button className="chat__suggestion" onClick={() => setInput('When did someone speak about a weapon?')}>
                🎤 When did someone speak about a weapon?
              </button>
              <button className="chat__suggestion" onClick={() => setInput('Show me frames with a person walking')}>
                🖼️ Show me frames with a person walking
              </button>
              <button className="chat__suggestion" onClick={() => setInput('Find scenes with vehicles')}>
                🚗 Find scenes with vehicles
              </button>
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <MessageBubble
              key={msg.id}
              message={msg}
              onClipClick={onClipClick}
            />
          ))
        )}

        {/* Loading indicator */}
        {loading && (
          <div className="chat__typing">
            <div className="chat__typing-avatar">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                <rect x="2" y="4" width="20" height="16" rx="3" stroke="var(--accent-primary)" strokeWidth="2" />
                <polygon points="10,8 10,16 16,12" fill="var(--accent-primary)" />
              </svg>
            </div>
            <div className="chat__typing-dots">
              <span /><span /><span />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="chat__input-area">
        {referenceImage && (
          <div className="chat__reference-image">
            <span className="chat__reference-image-label">Image Reference: {referenceImage.name}</span>
            <button
              className="chat__reference-image-remove"
              onClick={() => setReferenceImage(null)}
              title="Remove reference image"
            >
              Remove
            </button>
          </div>
        )}
        <div className="chat__input-wrapper">
          <input
            ref={imageInputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => {
              const file = e.target.files?.[0] || null;
              setReferenceImage(file);
            }}
          />
          <button
            className="chat__attach-btn"
            onClick={() => imageInputRef.current?.click()}
            title="Attach reference image"
            disabled={loading}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="3" width="18" height="18" rx="2" />
              <circle cx="8.5" cy="8.5" r="1.5" />
              <polyline points="21 15 16 10 5 21" />
            </svg>
          </button>
          <textarea
            ref={inputRef}
            className="chat__input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about your evidence videos..."
            rows={1}
            disabled={loading}
            id="chat-input"
          />
          <button
            className="chat__send-btn"
            onClick={handleSend}
            disabled={!input.trim() || loading}
            id="send-button"
            title="Send message"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </div>
        <p className="chat__input-hint">
          Press <kbd>Enter</kbd> to send · <kbd>Shift+Enter</kbd> for new line · attach image for visual search
        </p>
      </div>
    </div>
  );
}
