import { useState } from 'react';
import './App.css';
import Sidebar from './components/Sidebar';
import ChatWindow from './components/ChatWindow';
import VideoPlayer from './components/VideoPlayer';
import UploadModal from './components/UploadModal';
import CasePanel from './components/CasePanel';

function App() {
  const [selectedVideoId, setSelectedVideoId] = useState(null);
  const [selectedCaseId, setSelectedCaseId] = useState(null);
  const [showUpload, setShowUpload] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [activeClip, setActiveClip] = useState(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [casePanelId, setCasePanelId] = useState(null);

  const handleUploadComplete = (caseId, hasPendingMatches) => {
    setShowUpload(false);
    setRefreshKey((k) => k + 1);
    if (caseId) {
      setSelectedCaseId(caseId);
      // Novelty 1, Step 5: surface the auto-match confirmation UI immediately
      // instead of leaving it as an inert API a user would never find.
      if (hasPendingMatches) setCasePanelId(caseId);
    }
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
          onOpenCasePanel={(caseId) => setCasePanelId(caseId)}
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

      {/* Case dashboard: speaker match confirmation, testimony, contradictions */}
      {casePanelId && (
        <CasePanel caseId={casePanelId} onClose={() => setCasePanelId(null)} />
      )}
    </div>
  );
}

export default App;
