/**
 * StatusBadge — Processing status indicator with animations.
 */
import './StatusBadge.css';

export default function StatusBadge({ status, progress, message }) {
  const statusConfig = {
    processing: { label: 'Processing', icon: '⟳', className: 'status-badge--processing' },
    completed: { label: 'Ready', icon: '✓', className: 'status-badge--completed' },
    failed: { label: 'Failed', icon: '✕', className: 'status-badge--failed' },
    missing: { label: 'Missing', icon: '?', className: 'status-badge--failed' },
  };

  const config = statusConfig[status] || statusConfig.processing;

  return (
    <div className={`status-badge ${config.className}`} title={message || config.label}>
      <span className="status-badge__icon">{config.icon}</span>
      <span className="status-badge__label">{config.label}</span>
      {status === 'processing' && progress > 0 && (
        <div className="status-badge__progress">
          <div
            className="status-badge__progress-bar"
            style={{ width: `${Math.round(progress * 100)}%` }}
          />
        </div>
      )}
    </div>
  );
}
