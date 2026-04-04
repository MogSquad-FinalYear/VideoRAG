const API_BASE = '';

/**
 * Upload a video file for processing.
 * @param {File} file
 * @param {function} onProgress - optional XHR progress callback
 * @returns {Promise<{task_id: string, status: string, message: string}>}
 */
export async function uploadVideo(file, onProgress) {
  const formData = new FormData();
  formData.append('file', file);

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
 * @returns {Promise<{answer, clips, sources}>}
 */
export async function sendMessage(query, videoId = null) {
  const body = { query };
  if (videoId) body.video_id = videoId;

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
