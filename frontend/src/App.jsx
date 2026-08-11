import { useState } from 'react';
import './App.css';
import Sidebar from './components/Sidebar';
import ChatWindow from './components/ChatWindow';
import VideoPlayer from './components/VideoPlayer';
import UploadModal from './components/UploadModal';

function App() {
  const [selectedVideoId, setSelectedVideoId] = useState(null);
  const [selectedCaseId, setSelectedCaseId] = useState(null);
  const [showUpload, setShowUpload] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [activeClip, setActiveClip] = useState(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const handleUploadComplete = (caseId) => {
    setShowUpload(false);
    setRefreshKey((k) => k + 1);
    if (caseId) setSelectedCaseId(caseId);
  };

  const handleClipClick = (clip) => {
    setActiveClip(clip);
  };

  const handleSelectVideo = (videoId, caseId) => {
    setSelectedVideoId(videoId);
    if (caseId) setSelectedCaseId(caseId);
  };

  return (
    <div className="app">
      {/* Sidebar */}
      <aside className={`app__sidebar ${sidebarCollapsed ? 'app__sidebar--collapsed' : ''}`}>
        <Sidebar
          selectedVideoId={selectedVideoId}
          onSelectVideo={handleSelectVideo}
          onUploadClick={() => setShowUpload(true)}
          refreshKey={refreshKey}
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed((c) => !c)}
          onDeleteVideo={() => setRefreshKey((k) => k + 1)}
        />
      </aside>

      {/* Main chat area */}
      <main className="app__main">
        <ChatWindow
          videoId={selectedVideoId}
          caseId={selectedCaseId}
          onClipClick={handleClipClick}
        />
      </main>

      {/* Video player panel */}
      {activeClip && (
        <aside className="app__player animate-slideInRight">
          <VideoPlayer
            clip={activeClip}
            onClose={() => setActiveClip(null)}
          />
        </aside>
      )}

      {/* Upload modal */}
      {showUpload && (
        <UploadModal
          onClose={() => setShowUpload(false)}
          onUploadComplete={handleUploadComplete}
        />
      )}
    </div>
  );
}

export default App;
