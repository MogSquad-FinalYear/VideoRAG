/**
 * UploadModal — Drag-and-drop video upload dialog with progress tracking.
 * Includes case_id field for cross-session testimony memory.
 */
import { useState, useRef, useCallback } from 'react';
import { uploadVideo, pollTaskStatus, getSpeakerMatches } from '../api/client';
import './UploadModal.css';

const ALLOWED_TYPES = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm', 'video/x-flv', 'video/x-ms-wmv'];
const ALLOWED_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'];

export default function UploadModal({ onClose, onUploadComplete }) {
  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [caseId, setCaseId] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processing, setProcessing] = useState(false);
  const [processStatus, setProcessStatus] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const validateFile = (file) => {
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext) && !ALLOWED_TYPES.includes(file.type)) {
      return `Unsupported file type. Allowed: ${ALLOWED_EXTENSIONS.join(', ')}`;
    }
    if (file.size > 500 * 1024 * 1024) {
      return 'File too large. Maximum size is 500MB.';
    }
    return null;
  };

  const handleFile = (file) => {
    const err = validateFile(file);
    if (err) {
      setError(err);
      return;
    }
    setError(null);
    setSelectedFile(file);
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
  }, []);

  const handleUpload = async () => {
    if (!selectedFile) return;
    setError(null);
    setUploading(true);
    setUploadProgress(0);

    try {
      const result = await uploadVideo(selectedFile, (progress) => {
        setUploadProgress(progress);
      }, caseId.trim() || null);

      setUploading(false);
      setProcessing(true);

      // Poll for processing status
      await pollTaskStatus(result.task_id, (status) => {
        setProcessStatus(status);
      });

      // Processing complete — check whether the cross-session speaker
      // matcher (Novelty 1) left anything awaiting human confirmation.
      const finalCaseId = result.case_id || caseId.trim() || null;
      let hasPendingMatches = false;
      if (finalCaseId) {
        try {
          const matches = await getSpeakerMatches(finalCaseId);
          hasPendingMatches = (matches.pending || []).length > 0;
        } catch {
          // Non-fatal — the case dashboard can still be opened manually.
        }
      }
      onUploadComplete(finalCaseId, hasPendingMatches);
    } catch (err) {
      setError(err.message || 'Upload failed');
      setUploading(false);
      setProcessing(false);
    }
  };

  const formatSize = (bytes) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const isWorking = uploading || processing;

  return (
    <div className="upload-overlay" onClick={!isWorking ? onClose : undefined}>
      <div className="upload-modal animate-scaleIn" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="upload-modal__header">
          <h2 className="upload-modal__title">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--accent-primary)" strokeWidth="2">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12" />
            </svg>
            Upload Evidence Video
          </h2>
          {!isWorking && (
            <button className="upload-modal__close" onClick={onClose} id="close-upload">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          )}
        </div>

        {/* Case ID Input */}
        {!isWorking && (
          <div className="upload-modal__case-id">
            <label className="upload-modal__case-label" htmlFor="case-id-input">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z" />
              </svg>
              Case ID
              <span className="upload-modal__case-hint">(group videos from the same trial for contradiction detection)</span>
            </label>
            <input
              id="case-id-input"
              className="upload-modal__case-input"
              type="text"
              value={caseId}
              onChange={(e) => setCaseId(e.target.value)}
              placeholder="e.g., trial_001 (leave empty for auto-generated)"
            />
          </div>
        )}

        {/* Drop Zone */}
        {!isWorking && !selectedFile && (
          <div
            className={`upload-modal__dropzone ${dragOver ? 'upload-modal__dropzone--active' : ''}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => fileInputRef.current?.click()}
            id="dropzone"
          >
            <input
              ref={fileInputRef}
              type="file"
              accept={ALLOWED_EXTENSIONS.join(',')}
              onChange={(e) => e.target.files[0] && handleFile(e.target.files[0])}
              hidden
            />
            <div className="upload-modal__dropzone-icon">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
            <p className="upload-modal__dropzone-text">
              Drag & drop your video here
            </p>
            <span className="upload-modal__dropzone-subtext">
              or click to browse · MP4, AVI, MOV, MKV · Max 500MB
            </span>
          </div>
        )}

        {/* Selected File */}
        {selectedFile && !isWorking && (
          <div className="upload-modal__file">
            <div className="upload-modal__file-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent-primary)" strokeWidth="1.5">
                <rect x="2" y="4" width="20" height="16" rx="3" />
                <polygon points="10,8 10,16 16,12" />
              </svg>
            </div>
            <div className="upload-modal__file-info">
              <span className="upload-modal__file-name">{selectedFile.name}</span>
              <span className="upload-modal__file-size">{formatSize(selectedFile.size)}</span>
            </div>
            <button
              className="upload-modal__file-remove"
              onClick={() => setSelectedFile(null)}
              title="Remove file"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
        )}

        {/* Upload Progress */}
        {uploading && (
          <div className="upload-modal__progress">
            <div className="upload-modal__progress-header">
              <span>Uploading video...</span>
              <span>{Math.round(uploadProgress * 100)}%</span>
            </div>
            <div className="upload-modal__progress-bar">
              <div
                className="upload-modal__progress-fill"
                style={{ width: `${uploadProgress * 100}%` }}
              />
            </div>
          </div>
        )}

        {/* Processing Status */}
        {processing && processStatus && (
          <div className="upload-modal__processing">
            <div className="upload-modal__processing-spinner" />
            <div className="upload-modal__processing-info">
              <span className="upload-modal__processing-title">Processing Video</span>
              <span className="upload-modal__processing-msg">{processStatus.message}</span>
              <div className="upload-modal__progress-bar">
                <div
                  className="upload-modal__progress-fill upload-modal__progress-fill--processing"
                  style={{ width: `${(processStatus.progress || 0) * 100}%` }}
                />
              </div>
              <span className="upload-modal__processing-pct">
                {Math.round((processStatus.progress || 0) * 100)}%
              </span>
              {processStatus.progress >= 1 && processStatus.captions_ready === false && (
                <span className="upload-modal__captions-note">
                  Captions still indexing in the background — some visual
                  descriptions may be incomplete until this finishes.
                </span>
              )}
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="upload-modal__error">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <line x1="15" y1="9" x2="9" y2="15" />
              <line x1="9" y1="9" x2="15" y2="15" />
            </svg>
            {error}
          </div>
        )}

        {/* Actions */}
        {!isWorking && (
          <div className="upload-modal__actions">
            <button className="upload-modal__cancel" onClick={onClose}>
              Cancel
            </button>
            <button
              className="upload-modal__submit"
              onClick={handleUpload}
              disabled={!selectedFile}
              id="submit-upload"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12" />
              </svg>
              Upload & Process
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
