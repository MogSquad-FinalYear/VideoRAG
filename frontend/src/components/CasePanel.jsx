/**
 * CasePanel — Case-level dashboard: cross-session speaker match confirmation
 * (Novelty 1), testimony ledger, and contradiction flags (Novelty 2).
 *
 * The Speakers tab is the human-in-the-loop confirmation step the
 * implementation plan requires: an auto-matched voiceprint must be
 * confirmed (or corrected) by a user before it's trusted by downstream
 * contradiction detection.
 */
import { useState, useEffect, useCallback } from 'react';
import {
  getSpeakerMatches,
  confirmSpeakerMatch,
  getSpeakerRoles,
  setSpeakerRoles,
  getCaseTestimony,
  getCaseContradictions,
} from '../api/client';
import './CasePanel.css';

const ROLES = ['unknown', 'judge', 'witness', 'counsel'];

function formatTimestamp(seconds) {
  if (seconds == null) return '--:--';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export default function CasePanel({ caseId, initialTab = 'speakers', onClose }) {
  const [activeTab, setActiveTab] = useState(initialTab);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [pending, setPending] = useState([]);
  const [allSpeakers, setAllSpeakers] = useState([]);
  const [testimony, setTestimony] = useState([]);
  const [contradictions, setContradictions] = useState([]);

  const [renamingKey, setRenamingKey] = useState(null);
  const [renameValue, setRenameValue] = useState('');
  const [renameRole, setRenameRole] = useState('unknown');
  const [busyKey, setBusyKey] = useState(null);

  const [roleVideoId, setRoleVideoId] = useState('');
  const [videoRoles, setVideoRoles] = useState(null);
  const [rolesLoading, setRolesLoading] = useState(false);

  const loadAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [matches, testimonyRes, contradictionsRes] = await Promise.all([
        getSpeakerMatches(caseId),
        getCaseTestimony(caseId),
        getCaseContradictions(caseId),
      ]);
      setPending(matches.pending || []);
      setAllSpeakers(matches.all_speakers || []);
      setTestimony(testimonyRes.statements || []);
      setContradictions(contradictionsRes.contradictions || []);
    } catch (err) {
      setError(err.message || 'Failed to load case data');
    } finally {
      setLoading(false);
    }
  }, [caseId]);

  useEffect(() => {
    loadAll();
  }, [loadAll]);

  const videoIds = [...new Set(testimony.map((t) => t.video_id).filter(Boolean))];

  const matchKey = (m) => `${m.video_id}::${m.local_speaker_id}`;

  const handleConfirm = async (match) => {
    setBusyKey(matchKey(match));
    try {
      await confirmSpeakerMatch(caseId, {
        video_id: match.video_id,
        local_speaker_id: match.local_speaker_id,
        action: 'confirm',
      });
      await loadAll();
    } catch (err) {
      setError(err.message || 'Failed to confirm match');
    } finally {
      setBusyKey(null);
    }
  };

  const startRename = (match) => {
    setRenamingKey(matchKey(match));
    setRenameValue(match.canonical_name);
    setRenameRole(match.role || 'unknown');
  };

  const submitRename = async (match) => {
    if (!renameValue.trim()) return;
    setBusyKey(matchKey(match));
    try {
      await confirmSpeakerMatch(caseId, {
        video_id: match.video_id,
        local_speaker_id: match.local_speaker_id,
        action: 'rename',
        corrected_name: renameValue.trim(),
        role: renameRole,
      });
      setRenamingKey(null);
      await loadAll();
    } catch (err) {
      setError(err.message || 'Failed to correct match');
    } finally {
      setBusyKey(null);
    }
  };

  const loadVideoRoles = async (videoId) => {
    if (!videoId) {
      setVideoRoles(null);
      return;
    }
    setRolesLoading(true);
    try {
      const res = await getSpeakerRoles(videoId);
      setVideoRoles(res.speakers || []);
    } catch (err) {
      setError(err.message || 'Failed to load speaker roles');
      setVideoRoles([]);
    } finally {
      setRolesLoading(false);
    }
  };

  const handleRoleVideoChange = (e) => {
    const vid = e.target.value;
    setRoleVideoId(vid);
    loadVideoRoles(vid);
  };

  const updateVideoRole = (speakerId, field, value) => {
    setVideoRoles((prev) =>
      prev.map((r) => (r.speaker_id === speakerId ? { ...r, [field]: value } : r))
    );
  };

  const saveVideoRoles = async () => {
    if (!roleVideoId || !videoRoles) return;
    setRolesLoading(true);
    try {
      await setSpeakerRoles(roleVideoId, videoRoles);
    } catch (err) {
      setError(err.message || 'Failed to save speaker roles');
    } finally {
      setRolesLoading(false);
    }
  };

  return (
    <div className="case-panel-overlay" onClick={onClose}>
      <div className="case-panel animate-scaleIn" onClick={(e) => e.stopPropagation()}>
        <div className="case-panel__header">
          <h2 className="case-panel__title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--accent-primary)" strokeWidth="2">
              <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z" />
            </svg>
            Case: {caseId}
          </h2>
          <button className="case-panel__close" onClick={onClose}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        <div className="case-panel__tabs">
          <button
            className={`case-panel__tab ${activeTab === 'speakers' ? 'case-panel__tab--active' : ''}`}
            onClick={() => setActiveTab('speakers')}
          >
            Speakers {pending.length > 0 && <span className="case-panel__badge">{pending.length}</span>}
          </button>
          <button
            className={`case-panel__tab ${activeTab === 'testimony' ? 'case-panel__tab--active' : ''}`}
            onClick={() => setActiveTab('testimony')}
          >
            Testimony ({testimony.length})
          </button>
          <button
            className={`case-panel__tab ${activeTab === 'contradictions' ? 'case-panel__tab--active' : ''}`}
            onClick={() => setActiveTab('contradictions')}
          >
            Contradictions {contradictions.length > 0 && <span className="case-panel__badge case-panel__badge--warn">{contradictions.length}</span>}
          </button>
        </div>

        <div className="case-panel__body">
          {error && <div className="case-panel__error">{error}</div>}
          {loading ? (
            <div className="case-panel__loading">Loading case data…</div>
          ) : (
            <>
              {activeTab === 'speakers' && (
                <div className="case-panel__section">
                  <h3 className="case-panel__section-title">Pending Voiceprint Matches</h3>
                  <p className="case-panel__hint">
                    A matched speaker must be confirmed before it's trusted for
                    cross-session contradiction detection.
                  </p>
                  {pending.length === 0 ? (
                    <div className="case-panel__empty">No pending matches — everything confirmed.</div>
                  ) : (
                    pending.map((m) => {
                      const key = matchKey(m);
                      const isRenaming = renamingKey === key;
                      const isBusy = busyKey === key;
                      return (
                        <div key={key} className="case-panel__match">
                          <div className="case-panel__match-info">
                            <div className="case-panel__match-name">
                              {m.canonical_name}
                              <span className="case-panel__match-role">{m.role}</span>
                            </div>
                            <div className="case-panel__match-meta">
                              video <code>{m.video_id}</code> · local speaker <code>{m.local_speaker_id}</code> ·{' '}
                              {m.confidence > 0 ? `${Math.round(m.confidence * 100)}% match confidence` : 'newly registered'}
                            </div>
                          </div>
                          {isRenaming ? (
                            <div className="case-panel__rename-form">
                              <input
                                className="case-panel__rename-input"
                                value={renameValue}
                                onChange={(e) => setRenameValue(e.target.value)}
                                placeholder="Correct speaker name"
                                list="case-panel-speaker-names"
                              />
                              <datalist id="case-panel-speaker-names">
                                {allSpeakers.map((s) => (
                                  <option key={s.canonical_name} value={s.canonical_name} />
                                ))}
                              </datalist>
                              <select
                                className="case-panel__rename-role"
                                value={renameRole}
                                onChange={(e) => setRenameRole(e.target.value)}
                              >
                                {ROLES.map((r) => (
                                  <option key={r} value={r}>{r}</option>
                                ))}
                              </select>
                              <button
                                className="case-panel__btn case-panel__btn--primary"
                                disabled={isBusy}
                                onClick={() => submitRename(m)}
                              >
                                Save
                              </button>
                              <button className="case-panel__btn" onClick={() => setRenamingKey(null)}>
                                Cancel
                              </button>
                            </div>
                          ) : (
                            <div className="case-panel__match-actions">
                              <button
                                className="case-panel__btn case-panel__btn--primary"
                                disabled={isBusy}
                                onClick={() => handleConfirm(m)}
                              >
                                Confirm
                              </button>
                              <button className="case-panel__btn" disabled={isBusy} onClick={() => startRename(m)}>
                                Correct…
                              </button>
                            </div>
                          )}
                        </div>
                      );
                    })
                  )}

                  <h3 className="case-panel__section-title case-panel__section-title--spaced">
                    Known Speakers in Case
                  </h3>
                  {allSpeakers.length === 0 ? (
                    <div className="case-panel__empty">No speakers registered yet.</div>
                  ) : (
                    <div className="case-panel__speaker-list">
                      {allSpeakers.map((s) => (
                        <div key={s.canonical_name} className="case-panel__speaker-chip">
                          {s.canonical_name} <span className="case-panel__match-role">{s.role}</span>
                        </div>
                      ))}
                    </div>
                  )}

                  <h3 className="case-panel__section-title case-panel__section-title--spaced">
                    Edit Speaker Roles for a Video
                  </h3>
                  <select className="case-panel__video-select" value={roleVideoId} onChange={handleRoleVideoChange}>
                    <option value="">Select a video…</option>
                    {videoIds.map((vid) => (
                      <option key={vid} value={vid}>{vid}</option>
                    ))}
                  </select>
                  {rolesLoading && <div className="case-panel__loading">Loading…</div>}
                  {videoRoles && !rolesLoading && (
                    <div className="case-panel__roles-editor">
                      {videoRoles.length === 0 ? (
                        <div className="case-panel__empty">No speakers tagged for this video yet.</div>
                      ) : (
                        <>
                          {videoRoles.map((r) => (
                            <div key={r.speaker_id} className="case-panel__role-row">
                              <code>{r.speaker_id}</code>
                              <select
                                value={r.role}
                                onChange={(e) => updateVideoRole(r.speaker_id, 'role', e.target.value)}
                              >
                                {ROLES.map((role) => (
                                  <option key={role} value={role}>{role}</option>
                                ))}
                              </select>
                              <input
                                value={r.label || ''}
                                placeholder="Label (e.g. Jane Doe)"
                                onChange={(e) => updateVideoRole(r.speaker_id, 'label', e.target.value)}
                              />
                            </div>
                          ))}
                          <button className="case-panel__btn case-panel__btn--primary" onClick={saveVideoRoles}>
                            Save Roles
                          </button>
                        </>
                      )}
                    </div>
                  )}
                </div>
              )}

              {activeTab === 'testimony' && (
                <div className="case-panel__section">
                  {testimony.length === 0 ? (
                    <div className="case-panel__empty">No testimony stored for this case yet.</div>
                  ) : (
                    <div className="case-panel__testimony-list">
                      {testimony
                        .slice()
                        .sort((a, b) => (a.timestamp ?? 0) - (b.timestamp ?? 0))
                        .map((t, i) => (
                          <div key={i} className="case-panel__testimony-item">
                            <div className="case-panel__testimony-meta">
                              <strong>{t.role && t.role !== 'unknown' ? t.role : t.speaker_id}</strong>
                              <span className="case-panel__match-role">{t.speaker_id}</span>
                              <span className="msg__timestamp">[{formatTimestamp(t.timestamp)}]</span>
                              <code>{t.video_id}</code>
                            </div>
                            <p className="case-panel__testimony-text">"{t.text}"</p>
                          </div>
                        ))}
                    </div>
                  )}
                </div>
              )}

              {activeTab === 'contradictions' && (
                <div className="case-panel__section">
                  {contradictions.length === 0 ? (
                    <div className="case-panel__empty">No contradictions detected for this case.</div>
                  ) : (
                    contradictions.map((c, i) => (
                      <div key={c.id || i} className="case-panel__contradiction-item">
                        <div className="case-panel__match-name">
                          {c.speaker_id || 'Unknown Speaker'}
                          <span className={`msg__contradiction-conf msg__contradiction-conf--${
                            c.confidence >= 0.8 ? 'high' : c.confidence >= 0.6 ? 'medium' : 'low'
                          }`}>
                            {Math.round((c.confidence || 0) * 100)}% confidence
                          </span>
                        </div>
                        <div className="msg__contradiction-stmts">
                          <div className="msg__contradiction-stmt">
                            <span className="msg__contradiction-label">Session 1</span>
                            <span className="msg__timestamp">[{formatTimestamp(c.stmt_a_timestamp)}]</span>
                            <p>"{c.stmt_a_text}"</p>
                          </div>
                          <div className="msg__contradiction-vs">VS</div>
                          <div className="msg__contradiction-stmt">
                            <span className="msg__contradiction-label">Session 2</span>
                            <span className="msg__timestamp">[{formatTimestamp(c.stmt_b_timestamp)}]</span>
                            <p>"{c.stmt_b_text}"</p>
                          </div>
                        </div>
                        {c.explanation && <div className="msg__contradiction-explain">💡 {c.explanation}</div>}
                      </div>
                    ))
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
