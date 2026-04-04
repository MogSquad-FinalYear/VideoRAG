# VideoRAG 🔍🎬

**VideoRAG** is a full-stack, multimodal Retrieval-Augmented Generation (RAG) intelligence platform built for forensic video analysis. It allows investigators to upload evidence videos and query them using natural language. 

Behind the scenes, VideoRAG processes video content entirely locally using edge AI models to extract speech transversity, frame-by-frame visual similarity embeddings, and text scene captions. An Agentic LLM (powered by Groq) automatically decides what strategy or collection of strategies to use to query the video, returning rich, playable video clips directly matched to the user's inquiry.

## ✨ Features

- **Multimodal Video Indexing:** 
  - **Speech Index** powered by OpenAI's Whisper (Timeline synced spoken dialog).
  - **Visual Similarity Index** powered by OpenCLIP (Image-to-Video and advanced visual feature matching).
  - **Scene Context Index** powered by Salesforce's BLIP-Image-Captioning (Query visual scenes directly with natural text).
- **Agentic Routing Loop:** Driven by a Groq-hosted Llama-4 Scout 17B LLM. Uses tool-calling (FastMCP-style logic) to independently choose which local video indices to search based on the investigator's prompt.
- **Forensic UI Dashboard:** A dark-themed, glass-morphism dashboard built in React + Vite, equipped with collapsible sidebars, real-time XHR upload progress indicators, LLM response streaming, and interactive frame-by-frame video clip viewing.
- **Local Embedded Database:** Uses ChromaDB embedded on the disk to store and query the generated embeddings. **No external cloud database needed**.

## 🛠 Technology Stack

### Backend
- **Framework**: `FastAPI` + `Uvicorn`
- **Agent & LLM**: `Groq` API (`llama-4-scout-17b-16e-instruct`)
- **Computer Vision Extraction**: `OpenCV`, `imageio-ffmpeg`
- **AI Models (Running Locally)**: 
  - `openai-whisper` (Base/Tiny)
  - `open_clip` (ViT-B-32)
  - `transformers` (BLIP Image Captioning Base)
- **Vector Database**: `ChromaDB` Persistent Client

### Frontend
- **Framework**: `React 19` + `Vite`
- **Styling**: Pure CSS (CSS Variables + Glassmorphism aesthetic)

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.10+**
- **Node.js 18+**
- **NVIDIA GPU** (Optional, but highly recommended for fast Whisper/CLIP processing).
- **Groq API Key**: Go to [console.groq.com](https://console.groq.com/) and grab a free API Key.

### 2. Setting Up the Backend

1. Navigate to the root folder:
   ```bash
   cd VideoRAG
   ```
2. Create and activate a Virtual Environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
   *(Note: if you face issues installing `openai-whisper` regarding `pkg_resources`, ensure you downgrade setup tools with `pip install "setuptools<81"` and try again).*
   
4. Configure Environment Variables:
   Open the `.env` file in the root directory and ensure you paste your GROQ API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. Start the backend:
   ```bash
   python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```

### 3. Setting Up the Frontend

1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd VideoRAG/frontend
   ```
2. Install the JavaScript dependencies:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
4. Access the UI by navigating your browser to `http://localhost:5173/`. 

*(Note: The Vite config includes a proxy redirecting `http://localhost:5173/api/:routes` to `http://localhost:8000/` automatically to avoid CORS issues).*

## 🧠 Usage

1. Open the UI at `http://localhost:5173`.
2. Click **Upload Evidence** on the sidebar and select an `.mp4`, `.mov`, `.mkv`, or `.avi` video.
3. The platform will automatically extract audio, slice frames, embed images with CLIP, transcribe text with Whisper, and store them securely into `ChromaDB`.
4. In the Chat interface, ask natural language questions such as:
   - *"When did someone mention a weapon?"*
   - *"Do you see any red cars driving?"*
   - *"Show me exact frames where the suspect is walking outside."* 

Kubrick (the built-in Agent) will retrieve the exact clips directly correlated with your request.
