# AI-Driven Interview Analysis System - Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Frontend Layer (Web Browser)                     │
│  ┌─────────────────┬──────────────────┬─────────────────────────┐   │
│  │  Interview Room │   Dashboard      │   Management Portal     │   │
│  │  (WebRTC Video) │   (Analytics)    │   (Admin/Auth)          │   │
│  └────────┬────────┴────────┬─────────┴────────────┬────────────┘   │
└───────────┼─────────────────┼────────────────────────┼────────────────┘
            │                 │                        │
    WebSocket & HTTP API Calls (Flask-SocketIO, REST)
            │                 │                        │
┌───────────▼─────────────────▼────────────────────────▼────────────────┐
│                     Application Layer (Flask)                         │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                   app.py (Main Application)                      │ │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │ │
│  │  │ WebSocket   │  │ REST API     │  │ Authentication         │ │ │
│  │  │ Endpoints   │  │ Endpoints    │  │ & Authorization        │ │ │
│  │  └──────┬──────┘  └──────┬───────┘  └────────┬───────────────┘ │ │
│  │         │                │                   │                  │ │
│  │  • Room Management  • Interview CRUD   • Login/Register      │ │
│  │  • Recording Stream  • Analysis Results  • Role-based Access  │ │
│  │  • Real-time Updates • Dashboard Stats  • Session Management  │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                  Core Processing Pipeline                        │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │ │
│  │  │ nlp_utils.py │  │topic_classifier │ │ audio_emotion.py   │ │ │
│  │  │              │  │.py             │ │                      │ │ │
│  │  ├──────────────┤  ├──────────────┤  ├──────────────────────┤ │ │
│  │  │• Transcript  │  │• RoBERTa      │  │• Audio Feature      │ │ │
│  │  │  Cleaning    │  │  Classification│  │  Extraction (MFCC)  │ │ │
│  │  │• Sentiment   │  │• 7 Topics:    │  │• Emotion Detection  │ │ │
│  │  │  Analysis    │  │  - Academics  │  │  (8 emotions)       │ │ │
│  │  │  (mBERT)     │  │  - Career     │  │• Confidence Scores  │ │ │
│  │  │• Emotion     │  │  - Faculty    │  │                      │ │ │
│  │  │  Detection   │  │  - Mental     │  │                      │ │ │
│  │  │• Engagement  │  │  - Social     │  │                      │ │ │
│  │  │  Scoring     │  │  - Technology │  │                      │ │ │
│  │  │• Key Phrase  │  │  - Infrastructure                      │ │ │
│  │  │  Extraction  │  │                                         │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘ │ │
│  │                                                                  │ │
│  │  ┌──────────────┐  ┌──────────────────────────────────────┐   │ │
│  │  │topic_modeling │  │          Analysis Engine            │   │ │
│  │  │.py             │  │  ┌────────────────────────────────┐ │   │ │
│  │  ├──────────────┤  │  │ Process Interview Flow:        │ │   │ │
│  │  │• BERTopic    │  │  │ 1. Record & Transcribe        │ │   │ │
│  │  │  Model       │  │  │ 2. Clean Transcript            │ │   │ │
│  │  │• Keyword     │  │  │ 3. Extract Sentiment           │ │   │ │
│  │  │  Matching    │  │  │ 4. Classify Topics             │ │   │ │
│  │  │• Theme       │  │  │ 5. Detect Emotions (Audio)     │ │   │ │
│  │  │  Discovery   │  │  │ 6. Calculate Engagement        │ │   │ │
│  │  │• Category    │  │  │ 7. Store Results               │ │   │ │
│  │  │  Mapping     │  │  │ 8. Generate Insights           │ │   │ │
│  │  │              │  │  └────────────────────────────────┘ │   │ │
│  │  └──────────────┘  └──────────────────────────────────────┘   │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────┬─────────────────────────────────┘
                                     │
┌────────────────────────────────────▼─────────────────────────────────┐
│                     Data Persistence Layer                            │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                   database.py (SQLite)                           │ │
│  │  ┌──────────────────────────────────────────────────────────┐   │ │
│  │  │  Database Schema (interviews.db)                         │   │ │
│  │  │                                                           │   │ │
│  │  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐ │   │ │
│  │  │  │ interviews  │  │ transcripts  │  │analysis_results  │ │   │ │
│  │  │  ├─────────────┤  ├──────────────┤  ├──────────────────┤ │   │ │
│  │  │  │ id (PK)     │  │ id (PK)      │  │ id (PK)          │ │   │ │
│  │  │  │ room_id     │  │ interview_id │  │ interview_id (FK)│ │   │ │
│  │  │  │ type        │  │ speaker      │  │ analysis_type    │ │   │ │
│  │  │  │ student     │  │ text         │  │ result_data      │ │   │ │
│  │  │  │ interviewer │  │ timestamp    │  │ confidence       │ │   │ │
│  │  │  │ program     │  │              │  │ created_at       │ │   │ │
│  │  │  │ status      │  │              │  │                  │ │   │ │
│  │  │  │ recording   │  │              │  │ ┌──────────────┐ │ │   │ │
│  │  │  │ duration    │  │              │  │ │ Subtypes:    │ │ │   │ │
│  │  │  │ created_at  │  │              │  │ │• sentiment   │ │ │   │ │
│  │  │  │             │  │              │  │ │• topic       │ │ │   │ │
│  │  │  └─────────────┘  └──────────────┘  │ │• emotion     │ │ │   │ │
│  │  │                                      │ │• engagement  │ │ │   │ │
│  │  │                                      │ └──────────────┘ │ │   │ │
│  │  │                                      └──────────────────┘ │   │ │
│  │  └──────────────────────────────────────────────────────────┘   │ │
│  │                                                                    │ │
│  │  Database Features:                                               │ │
│  │  • WAL (Write-Ahead Logging) for concurrent access               │ │
│  │  • Foreign key constraints for referential integrity             │ │
│  │  • JSON storage for flexible analysis results                    │ │
│  │  • Timestamp tracking for all records                            │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. **Frontend Layer**
- **Technology**: HTML5, CSS3, JavaScript
- **Templates**:
  - `interview_room.html` - WebRTC video interface
  - `dashboard.html` - Analytics & statistics
  - `login.html` / `register.html` - Authentication
  - `interview_details.html` - Interview review

- **Features**:
  - Real-time video conferencing via WebRTC
  - Live transcript display
  - Interactive dashboards
  - Role-based UI rendering

---

### 2. **Application Layer (Flask)**

#### **Main App (`app.py`)**
- **Framework**: Flask + Flask-SocketIO
- **Security**:
  - CSRF protection
  - Rate limiting (200/day, 50/hour)
  - Session management
  - File upload validation

- **Core Endpoints**:
  ```
  WebSocket Events:
  • /socket.io/join_room - Join interview room
  • /socket.io/stream_audio - Stream audio for transcription
  • /socket.io/end_session - Complete interview
  
  REST API:
  • POST /api/register - User registration
  • POST /api/login - User authentication
  • GET /api/dashboard - Dashboard statistics
  • POST /api/interview/create - Create new interview
  • GET /api/interview/<id> - Retrieve interview details
  • POST /api/analyze-sentiment - Sentiment analysis
  • POST /api/analyze-topics - Topic extraction
  • POST /api/analyze-emotion - Emotion detection
  ```

---

### 3. **NLP Processing Pipeline**

#### **3a. Sentiment Analysis (`nlp_utils.py`)**
- **Model**: mBERT (Multilingual BERT)
- **Location**: `./model/taglish_sentiment_model_full.pth`
- **Labels**: Positive, Neutral, Negative
- **Features**:
  - Taglish support (English + Tagalog)
  - Confidence scores (0-1)
  - Transcript cleaning
  - ASR error correction

#### **3b. Topic Classification (`topic_classifier.py`)**
- **Model**: RoBERTa for Sequence Classification
- **Location**: `./model/`
- **Topics** (7 classes):
  1. Academics
  2. Career
  3. Faculty
  4. Infrastructure
  5. Mental Health
  6. Social
  7. Technology
- **Features**:
  - Multi-label classification
  - Configurable thresholds per topic
  - Confidence scoring

#### **3c. Topic Modeling (`topic_modeling.py`)**
- **Method**: BERTopic (unsupervised)
- **Alternative**: Keyword matching
- **Features**:
  - Automatic theme discovery
  - Category mapping
  - Institutional area labeling
  - Insight generation

#### **3d. Audio Emotion Detection (`audio_emotion.py`)**
- **Features Extracted**:
  - MFCC (Mel-Frequency Cepstral Coefficients)
  - Mel Spectrogram
  - Chromagram
- **Emotions** (8 classes):
  - Neutral, Calm, Happy, Sad
  - Angry, Fearful, Disgusted, Surprised
- **Confidence Scores**: Per emotion

---

### 4. **Data Persistence Layer**

#### **Database Schema (`database.py`)**

**Interviews Table**:
```sql
CREATE TABLE interviews (
  id INTEGER PRIMARY KEY,
  room_id TEXT UNIQUE,
  interview_type TEXT,
  student_name TEXT,
  interviewer_name TEXT,
  program TEXT,
  cohort TEXT,
  status TEXT,  -- 'in-progress', 'completed'
  recording_path TEXT,
  duration_seconds INTEGER,
  created_at TIMESTAMP,
  started_at TIMESTAMP,
  ended_at TIMESTAMP
)
```

**Transcripts Table**:
```sql
CREATE TABLE transcripts (
  id INTEGER PRIMARY KEY,
  interview_id INTEGER,
  speaker TEXT,
  text TEXT,
  timestamp TIMESTAMP
)
```

**Analysis Results Table**:
```sql
CREATE TABLE analysis_results (
  id INTEGER PRIMARY KEY,
  interview_id INTEGER,
  analysis_type TEXT,  -- 'sentiment', 'topic', 'emotion'
  result_data JSON,
  confidence FLOAT,
  created_at TIMESTAMP
)
```

- **Storage**: SQLite (interviews.db)
- **Features**:
  - WAL mode for concurrent access
  - Foreign key constraints
  - JSON support for flexible results
  - Transaction support

---

## Data Flow Diagram

### **Interview Recording Flow**
```
1. User enters room
   ↓
2. WebRTC establishes peer connection
   ↓
3. Audio/Video streamed to browser
   ↓
4. Transcript collected (via STT/ASR)
   ↓
5. Store in DB (transcripts table)
   ↓
6. Trigger NLP pipeline:
   ├─→ Sentiment Analysis → Store result
   ├─→ Topic Classification → Store result
   ├─→ Emotion Detection → Store result
   └─→ Engagement Scoring → Store result
   ↓
7. Generate insights from results
   ↓
8. Update dashboard in real-time
   ↓
9. User completes interview
   ↓
10. Final summary generated and stored
```

### **Analysis Pipeline Flow**
```
Raw Transcript
    ↓
[nlp_utils.clean_transcript()]
    ↓
Cleaned Text
    ├─→ [Sentiment Analysis]
    │   • Input: Cleaned text
    │   • Model: mBERT
    │   • Output: {label, confidence}
    │
    ├─→ [Topic Classification]
    │   • Input: Cleaned text
    │   • Model: RoBERTa
    │   • Output: {topics: [{name, score}]}
    │
    ├─→ [Audio Emotion Detection]
    │   • Input: Audio features
    │   • Model: Deep Learning
    │   • Output: {emotions: [{name, score}]}
    │
    └─→ [Engagement Scoring]
        • Input: Cleaned text
        • Heuristics: Word count, keywords
        • Output: {score, level}
    ↓
Analysis Results
    ↓
[Store in DB]
    ↓
[Generate Insights]
    ↓
Dashboard Update
```

---

## Technology Stack

### **Backend**
| Component | Technology | Version |
|-----------|-----------|---------|
| Framework | Flask | 2.x |
| Real-time | Flask-SocketIO | Latest |
| Database | SQLite | 3.x |
| NLP - Sentiment | Transformers (mBERT) | Latest |
| NLP - Topic | Transformers (RoBERTa) | Latest |
| NLP - Emotion | DistilBERT | Latest |
| Audio Processing | librosa | Latest |
| ML Framework | PyTorch | Latest |
| Security | Flask-WTF, Werkzeug | Latest |

### **Frontend**
| Component | Technology |
|-----------|-----------|
| Video Conferencing | WebRTC |
| Real-time Communication | Socket.IO |
| Templating | Jinja2 |
| Styling | CSS3 |
| Interaction | JavaScript |

---

## Key Features

### **1. Interview Management**
- Create/Schedule interviews
- Room-based isolation
- Role-based access (Admin/User)
- Interview history tracking

### **2. Real-time Analysis**
- Live transcript processing
- Sentiment scoring
- Topic detection
- Emotion analysis

### **3. Analytics & Dashboards**
- Interview statistics
- Sentiment trends
- Topic distribution
- Performance metrics

### **4. Security**
- User authentication
- CSRF protection
- Rate limiting
- File validation
- SQL injection prevention

### **5. Scalability**
- Database WAL mode
- Async processing
- Session management
- Resource optimization

---

## Deployment Architecture

### **Development**
```
Local Machine
├── Flask Dev Server (localhost:5000)
├── SQLite Database
├── Model Files
└── Static Assets
```

### **Production**
```
Server/Cloud
├── WSGI Server (Gunicorn/uWSGI)
├── Reverse Proxy (Nginx)
├── SQLite/PostgreSQL Database
├── Model Files (GPU optional)
├── SSL Certificates
└── File Storage
```

---

## Model Files

| Model | Type | Size | Purpose |
|-------|------|------|---------|
| `taglish_sentiment_model_full.pth` | mBERT Fine-tuned | ~440MB | Sentiment Analysis |
| `model.safetensors` | RoBERTa Quantized | ~436MB | Topic Classification |
| `config.json` | Configuration | 1KB | Model config |

---

## Performance Considerations

### **Optimization**
- Model caching to avoid reloading
- Batch processing for multiple transcripts
- Async task processing
- Database indexing on frequently queried fields

### **Bottlenecks**
- Model inference time: ~200-500ms per transcript
- Database queries: Indexed for <100ms
- WebRTC streaming: Network dependent
- File uploads: Limited to 100MB

---

## Future Enhancements

1. **Multi-language Support** - Expand beyond Taglish
2. **Real-time Transcription** - Integrate speech-to-text
3. **Advanced Analytics** - ML-based insights
4. **Distributed Processing** - Scale with Celery/Redis
5. **Export Capabilities** - PDF/Excel reports
6. **API Integration** - Third-party service connections

---

## Dependencies

See `requirements.txt` for full list:
- Flask ecosystem (Flask, Flask-SocketIO, Flask-WTF)
- Transformers & PyTorch
- Data processing (pandas, numpy, scikit-learn)
- Audio processing (librosa)
- Database (sqlite3)
- Security (Werkzeug)

---

**Version**: 1.0  
**Last Updated**: March 11, 2026  
**System Status**: Production Ready ✅
