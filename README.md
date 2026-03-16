# AI-Driven Interview Analysis System 🎓

A comprehensive web application designed to conduct and analyze student admission and exit interviews. Built with **Flask**, **WebRTC**, and **Hugging Face** NLP models, this system provides real-time video conferencing, speech-to-text transcription, and deep AI-powered analytics including sentiment analysis, emotion detection, and topic classification.

## 🌟 Key Features

- **WebRTC Video Conferencing:** Real-time peer-to-peer audio and video communication using WebRTC and Socket.IO signaling.
- **Live Speech Recognition:** Automatic, continuous browser-based speech-to-text transcription using the Web Speech API.
- **AI Sentiment & Emotion Analysis:** 
  - Uses an **mBERT** model fine-tuned on Taglish (Tagalog-English) to classify sentences into Positive, Neutral, or Negative sentiment.
  - Transformer-based emotion detection (Joy, Anger, Sadness, Fear, Surprise, Love).
- **Topic Classification:** Automatically extracts key discussion themes such as Academics, Career, Faculty, Infrastructure, Mental Health, and Social.
- **Admin Analytics Dashboard:** A comprehensive, responsive dashboard visualizing sentiment trends, emotion breakdowns, and overall student engagement using Chart.js.
- **Role-Based Access Control:** Secure, isolated interfaces and permissions for Students, Interviewers (Users), and Administrators.
- **Responsive & Accessible Design:** Fully mobile-optimized UI with skeleton loaders, WCAG-compliant contrast ratios, and `aria` accessibility tags.

## 🛠️ Technology Stack

- **Backend:** Python 3, Flask, Flask-SocketIO (Eventlet), SQLite3
- **Frontend:** HTML5, CSS3 Component Architecture, Vanilla JavaScript, Chart.js
- **Real-Time Communication:** WebRTC, Web Speech API
- **Machine Learning / NLP:** PyTorch, Transformers (Hugging Face), sentence-transformers, KeyBERT

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**
- A modern web browser with camera/microphone access (Chrome or Edge recommended)

### Installation

1. **Clone the repository and navigate to the project directory:**
   ```bash
   git clone <repository-url>
   cd Admin-and-Exit-Interview
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   
   # Windows
   .\.venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements_minimal.txt flask-wtf flask-limiter
   ```
   *Note: If you plan to use `bertopic` locally, you may need a C compiler installed for `hdbscan`. The minimal requirements avoid this build step.*

4. **Initialize the Database:**
   The SQLite database (`interviews.db`) automatically provisions its tables on the first run. 

### Running the Application

Start the Flask server with Eventlet:
```bash
python app.py
```

The application will be accessible at:
```
http://localhost:5000
```
*(Or `https://localhost:5000` if SSL certificates are provided for local WebRTC testing).*

## 🔒 Security & Roles

- **Admin Account:** Gains access to the global Analytics Dashboard to review all past interviews and organizational trends.
- **Interviewer Account:** Can generate new meeting rooms, manage interview recordings, and view individual interview analyses.
- **Student Access:** Students join rooms via an 8-character string, participating only in the conference without the ability to create rooms or view deeper analytics.

## 📊 Analytics Processing

The NLP analysis is non-blocking. As a transcript is streamed from the browser via WebSockets during the interview, a background thread running a queue worker safely processes the text through the Taglish Sentiment Model and Emotion Classifier, continuously writing the results to the database without slowing down the video feed.

## 📄 License

This project was built for educational and organizational research purposes.
