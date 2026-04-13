# System Architecture Analysis
**AI-Driven Admission & Exit Interview Analysis System**

---

## High-Level Overview

This is a **full-stack web application** that enables real-time video interview sessions between interviewers and students, with AI-powered analysis of the student's speech. The system performs **live sentiment analysis, emotion detection, topic classification, and engagement scoring** on student responses during interviews.

```mermaid
graph TB
    subgraph "Client Layer"
        A["🎓 Student Browser<br/>(Mobile/Desktop)"]
        B["👔 Interviewer Browser<br/>(Desktop)"]
    end

    subgraph "Communication Layer"
        C["WebRTC P2P<br/>(Video/Audio)"]
        D["Socket.IO<br/>(Signaling + Events)"]
    end

    subgraph "Application Layer"
        E["Flask Server<br/>(app.py - 1,148 lines)"]
        F["REST API<br/>(/api/*)"]
    end

    subgraph "AI/ML Pipeline"
        G["Sentiment Analysis<br/>(mBERT Taglish)"]
        H["Emotion Detection<br/>(DistilBERT)"]
        I["Topic Classification<br/>(RoBERTa-Tagalog)"]
        J["Key Phrase Extraction<br/>(KeyBERT)"]
        K["Engagement Scoring<br/>(Custom Heuristics)"]
        L["Audio Emotion<br/>(wav2vec2)"]
    end

    subgraph "Storage Layer"
        M[("SQLite<br/>interviews.db")]
        N["File System<br/>(recordings/, transcripts/)"]
    end

    A <--> C
    B <--> C
    A <--> D
    B <--> D
    D <--> E
    E --> F
    E --> G & H & I & J & K
    F --> M
    E --> N
    L -.-> E
```

---

## Architecture Layers

### 1. Client Layer (Frontend)

| File | Purpose | Size |
|------|---------|------|
| `templates/index.html` | Student/Admin home — join room, create room | 7.6 KB |
| `templates/interview_room.html` | Live video call interface | 8.5 KB |
| `templates/dashboard.html` | Analytics dashboard (charts, tables, export) | 33.5 KB |
| `templates/interview_details.html` | Single interview deep-dive | 15 KB |
| `templates/login.html` / `register.html` | Authentication pages | 1.9 / 3.8 KB |
| `static/js/webrtc.js` | WebRTC, speech recognition, recording | 39 KB (1,103 lines) |
| `static/css/` | 5 stylesheets (style, dashboard, interview, details, auth) | ~45 KB total |

**Key frontend technologies:**
- **WebRTC** — peer-to-peer video/audio between student and interviewer
- **Web Speech API** — browser-based speech recognition (student only)
- **Chart.js** — dashboard visualizations (sentiment, trend, topic, emotion, type charts)
- **Socket.IO client** — real-time bidirectional events

### 2. Communication Layer

```mermaid
sequenceDiagram
    participant S as Student Browser
    participant Srv as Flask + SocketIO
    participant I as Interviewer Browser

    S->>Srv: join-room {room_id, role: "student"}
    Srv->>I: participant-joined {count: 2}
    I->>Srv: offer {SDP}
    Srv->>S: offer {SDP}
    S->>Srv: answer {SDP}
    Srv->>I: answer {SDP}
    Note over S,I: WebRTC P2P established (video/audio direct)

    I->>Srv: interview-started
    Srv->>S: interview-started
    Note over S: Speech Recognition begins

    S->>Srv: transcript-line {text}
    Srv->>I: transcript-line {text}
    Srv->>Srv: NLP analysis (async)
    Srv->>I: analysis-result {sentiment, emotion, engagement}

    I->>Srv: interview-ended
    Srv->>S: interview-ended
    Note over S: Auto-redirect to /home with modal
```

**Socket.IO events:**

| Event | Direction | Purpose |
|-------|-----------|---------|
| `join-room` | Client → Server | Join interview room |
| `participant-joined` | Server → Client | Notify when someone joins |
| `offer` / `answer` / `ice-candidate` | Bidirectional | WebRTC signaling |
| `interview-started` | Bidirectional | Start recording/transcription |
| `interview-ended` | Bidirectional | End call, cleanup, redirect |
| `transcript-line` | Client → Server → Client | Live transcript broadcast |
| `analysis-result` | Server → Client | Push NLP results to interviewer |

### 3. Application Layer (`app.py` — 1,148 lines)

Flask server with Eventlet WSGI, serving both HTTP routes and WebSocket events.

**REST API Endpoints:**

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/home` | Student/admin landing page |
| GET | `/dashboard` | Admin analytics dashboard |
| GET | `/create-interview` | Create new room |
| GET | `/join/<room_id>` | Join existing room |
| GET | `/interview/<id>` | Interview detail page |
| POST | `/api/transcript` | Save transcript + trigger analysis |
| GET | `/api/dashboard/stats` | Aggregated dashboard data |
| GET | `/api/dashboard/recent` | Paginated recent interviews |
| POST | `/api/room/<id>/metadata` | Update interview metadata |
| POST | `/api/upload-recording` | Save recording file |
| POST | `/api/warmup` | Pre-load NLP models |
| GET | `/api/export-report` | Export Excel report |
| POST | `/login` / `/register` / `/logout` | Authentication |

**Security features:**
- CSRF protection (Flask-WTF)
- Rate limiting (Flask-Limiter)
- Password hashing (Werkzeug)
- Role-based access (admin, interviewer, student)
- Self-signed SSL certificate generation

### 4. AI/ML Pipeline

```mermaid
graph LR
    A["Student Speech<br/>(Web Speech API)"] --> B["Transcript Text"]
    B --> C["clean_transcript()<br/>Remove fillers, fix ASR"]
    C --> D["Sentiment Analysis<br/>mBERT (Taglish)"]
    C --> E["Emotion Detection<br/>DistilBERT"]
    C --> F["Key Phrases<br/>KeyBERT"]
    C --> G["Engagement Score<br/>Custom heuristics"]
    D & E & F & G --> H["Combined Result"]
    H --> I["Save to DB"]
    H --> J["Push via Socket.IO"]

    K["End of Interview"] --> L["Topic Classification<br/>RoBERTa-Tagalog"]
    L --> I
```

#### Module Details

| Module | Model | Purpose | Fallback |
|--------|-------|---------|----------|
| `nlp_utils.py` (658 lines) | **mBERT** (`taglish_sentiment_model_full.pth`, 711 MB) | Sentiment: Positive/Neutral/Negative | Rule-based TextBlob |
| `nlp_utils.py` | **DistilBERT** (`bhadresh-savani/distilbert-base-uncased-emotion`) | 6 emotions: joy, sadness, love, anger, fear, surprise | — |
| `nlp_utils.py` | **KeyBERT** | Key phrase extraction from transcripts | — |
| `nlp_utils.py` | Custom heuristics | Engagement score (0-10) based on word count, keywords, response length | — |
| `topic_classifier.py` (169 lines) | **RoBERTa-Tagalog** (`model/` folder, 436 MB) | 7 topics: Academics, Career, Faculty, Infrastructure, Mental health, Social, Technology | — |
| `topic_modeling.py` (350 lines) | **BERTopic** + Sentence Transformers | Unsupervised topic discovery with institutional category mapping | Keyword matching |
| `audio_emotion.py` (440 lines) | **wav2vec2** (HuggingFace) | 8 audio emotions from recordings | Feature-based (librosa) |

### 5. Storage Layer

#### Database Schema (`database.py` — 910 lines)

```mermaid
erDiagram
    users {
        int id PK
        text username UK
        text email UK
        text password_hash
        text role
        text full_name
        timestamp created_at
    }

    interviews {
        int id PK
        text room_id UK
        text interview_type
        text student_name
        text interviewer_name
        text program
        text cohort
        text status
        timestamp created_at
        int duration_seconds
    }

    transcripts {
        int id PK
        int interview_id FK
        text speaker
        text text
        timestamp timestamp
    }

    analysis_results {
        int id PK
        int interview_id FK
        int transcript_id FK
        text sentiment_label
        real sentiment_confidence
        text emotions_json
        text keyphrases_json
        real engagement_score
    }

    interview_summary {
        int id PK
        int interview_id FK
        int total_words
        text dominant_sentiment
        text dominant_emotion
        text top_topics_json
        real avg_engagement_score
    }

    topic_classifications {
        int id PK
        int interview_id FK
        real academics_percent
        real career_percent
        real faculty_percent
        real infrastructure_percent
        real mental_health_percent
        real social_percent
        real technology_percent
    }

    interviews ||--o{ transcripts : has
    interviews ||--o{ analysis_results : has
    interviews ||--o| interview_summary : has
    interviews ||--o| topic_classifications : has
    transcripts ||--o| analysis_results : analyzed_by
```

#### File Storage

| Directory | Contents |
|-----------|----------|
| `recordings/` | WebM video/audio recordings |
| `transcripts/` | Exported transcript files |
| `model/` | Trained RoBERTa-Tagalog model (436 MB safetensors + config) |

---

## Technology Stack Summary

| Layer | Technology | Version |
|-------|-----------|---------|
| **Backend** | Flask + Flask-SocketIO | 3.1.3 / 5.6.1 |
| **WSGI** | Eventlet (async) | — |
| **Database** | SQLite (WAL mode) | — |
| **Auth** | Flask-WTF (CSRF) + Flask-Limiter | 1.2.2 / 4.1.1 |
| **ML Framework** | PyTorch | 2.10.0 |
| **NLP** | Transformers (HuggingFace) | 5.3.0 |
| **Embeddings** | Sentence Transformers | 5.2.3 |
| **Keyphrases** | KeyBERT | 0.9.0 |
| **Audio** | Librosa + SoundFile + Pydub | 0.11.0 |
| **Frontend** | Vanilla HTML/CSS/JS + Chart.js | — |
| **Real-time** | WebRTC + Socket.IO + Web Speech API | — |
| **SSL** | pyOpenSSL (self-signed) | 25.3.0 |

---

## Data Flow Summary

1. **Interview starts** → Interviewer creates room → Student joins via room code
2. **During interview** → WebRTC handles video; Web Speech API transcribes student speech
3. **Real-time analysis** → Each transcript line is sent to the server, processed through the NLP pipeline (sentiment + emotion + keyphrases + engagement), and results are pushed back to the interviewer's sidebar
4. **Interview ends** → Topic classification runs on the full transcript; interview summary is calculated and saved
5. **Dashboard** → Admins view aggregated statistics, charts (sentiment distribution, trend over time, topic breakdown, emotion analysis), and recent interviews with search + pagination
