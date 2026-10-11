# JobApp Helper 🚀

An autonomous multi-agent AI assistant for researching tech startups, evaluating candidate-role fit, and generating high-converting cold emails and LinkedIn connection notes with local ChromaDB vector memory.

---

## ✨ Features

- **Multi-Agent LangGraph Pipeline**:
  - `company_researcher`: Web intelligence via DuckDuckGo (`ddgs`) with local vector caching.
  - `fit_assessor`: Evaluates candidate skills against the company stack and provides a 1-10 match score.
  - `email_writer`: Drafts tailored, project-referenced cold emails (<150 words).
  - `linkedin_writer`: Writes punchy, character-limited connection notes (<300 characters).
- **ChromaDB Vector Memory**:
  - Instant cache hits on repeat lookups to save LLM tokens and bypass web searches.
  - Semantic startup discovery (`find_similar_companies`) to find similar companies from past research.
- **Full-Stack Web Studio**:
  - FastAPI backend with live Server-Sent Events (SSE) streaming progress.
  - Interactive Outreach Studio with one-click copy, `mailto:` link export, and real-time character counters.
  - Dynamic candidate profile switching.
  - Applications archive dashboard.

---

## 🛠️ Quick Start

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/your-username/job_app_helper.git
cd job_app_helper

# Create and activate virtual environment
python -m venv job_agent_env
# On Windows:
job_agent_env\Scripts\activate
# On Linux/macOS:
source job_agent_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```env
GEMINI_API_KEY="your_gemini_api_key"
GROQ_API_KEY="your_groq_api_key"
```

### 3. Run the Web Studio

```bash
python server.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

### 4. (Optional) Run CLI Pipeline

```bash
python job_application_agent_v2.py
```

---

## 📁 Project Structure

```
job_app_helper/
├── server.py                   # FastAPI REST & SSE Backend
├── pipeline_service.py         # Streaming Multi-Agent Pipeline Service
├── job_application_agent_v2.py # Standalone LangGraph CLI Agent
├── memory.py                   # ChromaDB Vector Store & Semantic Similarity
├── static/                     # Web Application UI (HTML/CSS/JS)
├── applications/               # Saved outreach packages (.json)
├── chroma_memory/              # Local ChromaDB vector database
├── requirements.txt            # Python dependencies
└── profiles.json               # Saved candidate profiles
```

---

## 🔒 Security

API keys and virtual environments are ignored via `.gitignore`. Never commit your `.env` file.
