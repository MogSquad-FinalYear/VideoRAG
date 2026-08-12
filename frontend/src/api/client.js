const API_BASE = '';

/**
 * Upload a video file for processing.
 * @param {File} file
 * @param {function} onProgress - optional XHR progress callback
 * @param {string|null} caseId - optional case identifier for cross-session testimony memory
 * @returns {Promise<{task_id: string, status: string, message: string, case_id: string}>}
 */
export async function uploadVideo(file, onProgress, caseId = null) {
  const formData = new FormData();
  formData.append('file', file);
  if (caseId) formData.append('case_id', caseId);

  if (onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${API_BASE}/execute`);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress(e.loaded / e.total);
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText));
        } else {
          reject(new Error(`Upload failed: ${xhr.statusText}`));
        }
      };
      xhr.onerror = () => reject(new Error('Upload failed'));
      xhr.send(formData);
    });
  }

  const res = await fetch(`${API_BASE}/execute`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
  return res.json();
}

/**
 * Poll task status.
 * @param {string} taskId
 * @returns {Promise<{task_id, status, progress, message, video_id}>}
 */
export async function getTaskStatus(taskId) {
  const res = await fetch(`${API_BASE}/status/${taskId}`);
  if (!res.ok) throw new Error(`Status check failed: ${res.statusText}`);
  return res.json();
}

/**
 * Send a chat message to the agent.
 * @param {string} query
 * @param {string|null} videoId
 * @param {string|null} caseId - optional case ID for contradiction checking
 * @returns {Promise<{answer, clips, sources, citations, contradictions}>}
 */
export async function sendMessage(query, videoId = null, caseId = null) {
  const body = { query };
  if (videoId) body.video_id = videoId;
  if (caseId) body.case_id = caseId;

  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`Chat failed: ${res.statusText}`);
  return res.json();
}

/**
 * Send a chat message with a reference image.
 * @param {string} query
 * @param {File} imageFile
 * @param {string|null} videoId
 * @returns {Promise<{answer, clips, sources}>}
 */
export async function sendMessageWithImage(query, imageFile, videoId = null) {
  const formData = new FormData();
  formData.append('query', query);
  formData.append('image', imageFile);
  if (videoId) formData.append('video_id', videoId);

  const res = await fetch(`${API_BASE}/chat/image`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(`Image chat failed: ${res.statusText}`);
  return res.json();
}

/**
 * Get list of all uploaded videos.
 * @returns {Promise<Array<VideoInfo>>}
 */
export async function getVideos() {
  const res = await fetch(`${API_BASE}/videos`);
  if (!res.ok) throw new Error(`Failed to fetch videos: ${res.statusText}`);
  return res.json();
}

/**
 * Get health/stats.
 * @returns {Promise<{status, indexes}>}
 */
export async function getHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

/**
 * Delete a video and all its indexed data.
 * @param {string} videoId
 * @returns {Promise<{status, video_id, deleted}>}
 */
export async function deleteVideo(videoId) {
  const res = await fetch(`${API_BASE}/videos/${videoId}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(`Delete failed: ${res.statusText}`);
  return res.json();
}

/**
 * Get speaker-to-role mappings for a video.
 * @param {string} videoId
 * @returns {Promise<{video_id, speakers}>}
 */
export async function getSpeakerRoles(videoId) {
  const res = await fetch(`${API_BASE}/videos/${videoId}/speakers`);
  if (!res.ok) throw new Error(`Failed to fetch speaker roles: ${res.statusText}`);
  return res.json();
}

/**
 * Set speaker-to-role mappings for a video.
 * @param {string} videoId
 * @param {Array<{speaker_id: string, role: string, label?: string}>} mappings
 * @returns {Promise<{video_id, updated}>}
 */
export async function setSpeakerRoles(videoId, mappings) {
  const res = await fetch(`${API_BASE}/videos/${videoId}/speakers`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mappings }),
  });
  if (!res.ok) throw new Error(`Failed to set speaker roles: ${res.statusText}`);
  return res.json();
}

/**
 * Get pending cross-session voiceprint matches for a case, awaiting human confirmation.
 * @param {string} caseId
 * @returns {Promise<{case_id, pending, all_speakers}>}
 */
export async function getSpeakerMatches(caseId) {
  const res = await fetch(`${API_BASE}/cases/${caseId}/speaker-matches`);
  if (!res.ok) throw new Error(`Failed to fetch speaker matches: ${res.statusText}`);
  return res.json();
}

/**
 * Confirm or correct a cross-session voiceprint match.
 * @param {string} caseId
 * @param {{video_id: string, local_speaker_id: string, action: 'confirm'|'rename', corrected_name?: string, role?: string}} payload
 * @returns {Promise<{case_id, result}>}
 */
export async function confirmSpeakerMatch(caseId, payload) {
  const res = await fetch(`${API_BASE}/cases/${caseId}/speaker-matches/confirm`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Failed to confirm speaker match: ${res.statusText}`);
  return res.json();
}

/**
 * Get all testimony statements for a case across sessions.
 * @param {string} caseId
 * @returns {Promise<{case_id, total_statements, statements}>}
 */
export async function getCaseTestimony(caseId) {
  const res = await fetch(`${API_BASE}/cases/${caseId}/testimony`);
  if (!res.ok) throw new Error(`Failed to fetch testimony: ${res.statusText}`);
  return res.json();
}

/**
 * Get detected contradictions for a case.
 * @param {string} caseId
 * @returns {Promise<{case_id, total_contradictions, contradictions}>}
 */
export async function getCaseContradictions(caseId) {
  const res = await fetch(`${API_BASE}/cases/${caseId}/contradictions`);
  if (!res.ok) throw new Error(`Failed to fetch contradictions: ${res.statusText}`);
  return res.json();
}

/**
 * Poll task status until completion or failure.
 * @param {string} taskId
 * @param {function} onUpdate - called with each status update
 * @param {number} intervalMs
 * @returns {Promise<object>} - the final status
 */
export function pollTaskStatus(taskId, onUpdate, intervalMs = 2000) {
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const status = await getTaskStatus(taskId);
        onUpdate(status);
        if (status.status === 'completed') {
          resolve(status);
        } else if (status.status === 'failed') {
          reject(new Error(status.message || 'Processing failed'));
        } else {
          setTimeout(poll, intervalMs);
        }
      } catch (err) {
        reject(err);
      }
    };
    poll();
  });
}
