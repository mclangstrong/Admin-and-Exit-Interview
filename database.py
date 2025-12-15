"""
Database Module for Interview Analysis System
==============================================
Handles SQLite database operations for storing:
- Interview sessions
- Transcripts
- Analysis results (sentiment, topics, emotions)
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Database path
DB_PATH = "interviews.db"


def get_connection():
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize database with required tables."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Interviews table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            room_id TEXT UNIQUE NOT NULL,
            interview_type TEXT DEFAULT 'admission',
            student_name TEXT,
            interviewer_name TEXT,
            program TEXT,
            cohort TEXT,
            status TEXT DEFAULT 'completed',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            ended_at TIMESTAMP,
            duration_seconds INTEGER,
            recording_path TEXT
        )
    ''')
    
    # Transcripts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interview_id INTEGER NOT NULL,
            speaker TEXT NOT NULL,
            text TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (interview_id) REFERENCES interviews(id)
        )
    ''')
    
    # Analysis results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interview_id INTEGER NOT NULL,
            transcript_id INTEGER,
            sentiment_label TEXT,
            sentiment_confidence REAL,
            sentiment_positive REAL,
            sentiment_neutral REAL,
            sentiment_negative REAL,
            emotions_json TEXT,
            topics_json TEXT,
            keyphrases_json TEXT,
            engagement_score REAL,
            engagement_level TEXT,
            audio_emotion TEXT,
            audio_emotion_confidence REAL,
            interview_state TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (interview_id) REFERENCES interviews(id),
            FOREIGN KEY (transcript_id) REFERENCES transcripts(id)
        )
    ''')
    
    # Aggregate statistics table (for dashboard)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interview_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interview_id INTEGER UNIQUE NOT NULL,
            total_words INTEGER,
            total_duration_seconds INTEGER,
            avg_sentiment_score REAL,
            dominant_sentiment TEXT,
            dominant_emotion TEXT,
            top_topics_json TEXT,
            avg_engagement_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (interview_id) REFERENCES interviews(id)
        )
    ''')
    
    # Users table for authentication
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            full_name TEXT,
            course TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    
    # Create default admin if not exists
    cursor.execute('SELECT COUNT(*) as count FROM users WHERE role = "admin"')
    if cursor.fetchone()['count'] == 0:
        import hashlib
        default_password = hashlib.sha256('admin123'.encode()).hexdigest()
        cursor.execute('''
            INSERT INTO users (username, password_hash, role, full_name)
            VALUES (?, ?, ?, ?)
        ''', ('admin', default_password, 'admin', 'System Administrator'))
        print("   Created default admin user (username: admin, password: admin123)")
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")


# ============================================================================
# USER OPERATIONS
# ============================================================================

def create_user(username: str, password: str, role: str = 'user', 
                full_name: str = '', email: str = '', course: str = '') -> Optional[int]:
    """Create a new user."""
    import hashlib
    
    conn = get_connection()
    cursor = conn.cursor()
    
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    try:
        cursor.execute('''
            INSERT INTO users (username, password_hash, role, full_name, email, course)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (username, password_hash, role, full_name, email, course))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        conn.close()
        return None


def verify_user(username: str, password: str) -> Optional[Dict]:
    """Verify user credentials and return user data."""
    import hashlib
    from datetime import datetime
    
    conn = get_connection()
    cursor = conn.cursor()
    
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    cursor.execute('''
        SELECT id, username, role, full_name, email 
        FROM users 
        WHERE username = ? AND password_hash = ?
    ''', (username, password_hash))
    
    row = cursor.fetchone()
    
    if row:
        # Update last login
        cursor.execute('''
            UPDATE users SET last_login = ? WHERE id = ?
        ''', (datetime.now().isoformat(), row['id']))
        conn.commit()
        conn.close()
        return dict(row)
    
    conn.close()
    return None


def get_user_by_id(user_id: int) -> Optional[Dict]:
    """Get user by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, username, role, full_name, email, course, created_at, last_login
        FROM users WHERE id = ?
    ''', (user_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


def get_all_users() -> List[Dict]:
    """Get all users (for admin)."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, username, role, full_name, email, created_at, last_login
        FROM users ORDER BY created_at DESC
    ''')
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def update_user_role(user_id: int, new_role: str) -> bool:
    """Update user role (admin only)."""
    if new_role not in ['admin', 'user']:
        return False
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('UPDATE users SET role = ? WHERE id = ?', (new_role, user_id))
    conn.commit()
    conn.close()
    return True


def delete_user(user_id: int) -> bool:
    """Delete user (admin only)."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()
    return True


# ============================================================================
# INTERVIEW OPERATIONS
# ============================================================================

def create_interview(room_id: str, metadata: Dict = None) -> int:
    """Create a new interview record."""
    conn = get_connection()
    cursor = conn.cursor()
    
    metadata = metadata or {}
    
    cursor.execute('''
        INSERT INTO interviews (room_id, interview_type, student_name, interviewer_name, program, cohort)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        room_id,
        metadata.get('interview_type', 'admission'),
        metadata.get('student_name', ''),
        metadata.get('interviewer_name', ''),
        metadata.get('program', ''),
        metadata.get('cohort', '')
    ))
    
    interview_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return interview_id


def update_interview(room_id: str, **kwargs) -> bool:
    """Update interview record."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Build update query dynamically
    updates = []
    values = []
    for key, value in kwargs.items():
        if key in ['student_name', 'interviewer_name', 'program', 'cohort', 
                   'interview_type', 'status', 'started_at', 'ended_at', 
                   'duration_seconds', 'recording_path']:
            updates.append(f"{key} = ?")
            values.append(value)
    
    if updates:
        values.append(room_id)
        cursor.execute(f'''
            UPDATE interviews SET {', '.join(updates)} WHERE room_id = ?
        ''', values)
        conn.commit()
    
    conn.close()
    return True


def get_interview(room_id: str) -> Optional[Dict]:
    """Get interview by room ID."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM interviews WHERE room_id = ?', (room_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


def get_interview_by_id(interview_id: int) -> Optional[Dict]:
    """Get interview by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM interviews WHERE id = ?', (interview_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


def get_all_interviews(limit: int = 50) -> List[Dict]:
    """Get all interviews."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT i.*, 
               s.dominant_sentiment,
               s.avg_engagement_score,
               s.dominant_emotion
        FROM interviews i
        LEFT JOIN interview_summary s ON i.id = s.interview_id
        ORDER BY i.created_at DESC
        LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


# ============================================================================
# TRANSCRIPT OPERATIONS
# ============================================================================

def add_transcript_line(interview_id: int, speaker: str, text: str) -> int:
    """Add a transcript line."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO transcripts (interview_id, speaker, text)
        VALUES (?, ?, ?)
    ''', (interview_id, speaker, text))
    
    transcript_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return transcript_id


def get_transcript(interview_id: int) -> List[Dict]:
    """Get full transcript for an interview."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM transcripts WHERE interview_id = ? ORDER BY timestamp
    ''', (interview_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


# ============================================================================
# ANALYSIS OPERATIONS
# ============================================================================

def save_analysis(interview_id: int, analysis: Dict, transcript_id: int = None) -> int:
    """Save analysis results."""
    conn = get_connection()
    cursor = conn.cursor()
    
    sentiment = analysis.get('sentiment', {})
    emotions = analysis.get('emotions', {})
    keyphrases = analysis.get('keyphrases', [])
    engagement = analysis.get('engagement', {})
    
    cursor.execute('''
        INSERT INTO analysis_results (
            interview_id, transcript_id,
            sentiment_label, sentiment_confidence,
            sentiment_positive, sentiment_neutral, sentiment_negative,
            emotions_json, keyphrases_json,
            engagement_score, engagement_level
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        interview_id,
        transcript_id,
        sentiment.get('label'),
        sentiment.get('confidence'),
        sentiment.get('probabilities', {}).get('Positive'),
        sentiment.get('probabilities', {}).get('Neutral'),
        sentiment.get('probabilities', {}).get('Negative'),
        json.dumps(emotions),
        json.dumps(keyphrases),
        engagement.get('score'),
        engagement.get('level')
    ))
    
    analysis_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return analysis_id


def save_audio_emotion(interview_id: int, emotion_result: Dict) -> bool:
    """Save audio emotion analysis."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE analysis_results 
        SET audio_emotion = ?, audio_emotion_confidence = ?, interview_state = ?
        WHERE interview_id = ? AND audio_emotion IS NULL
        ORDER BY id DESC LIMIT 1
    ''', (
        emotion_result.get('primary_emotion'),
        emotion_result.get('confidence'),
        emotion_result.get('interview_state', {}).get('primary'),
        interview_id
    ))
    
    conn.commit()
    conn.close()
    return True


def save_topics(interview_id: int, topics_result: Dict) -> bool:
    """Save topic analysis."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE analysis_results 
        SET topics_json = ?
        WHERE interview_id = ?
        ORDER BY id DESC LIMIT 1
    ''', (json.dumps(topics_result), interview_id))
    
    conn.commit()
    conn.close()
    return True


# ============================================================================
# SUMMARY/STATISTICS
# ============================================================================

def calculate_interview_summary(interview_id: int) -> Dict:
    """Calculate and save interview summary."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get all analysis for this interview
    cursor.execute('''
        SELECT * FROM analysis_results WHERE interview_id = ?
    ''', (interview_id,))
    analyses = cursor.fetchall()
    
    if not analyses:
        conn.close()
        return {}
    
    # Calculate averages
    sentiments = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    emotions_count = {}
    total_engagement = 0
    
    for a in analyses:
        if a['sentiment_label']:
            sentiments[a['sentiment_label']] = sentiments.get(a['sentiment_label'], 0) + 1
        
        if a['emotions_json']:
            emotions = json.loads(a['emotions_json'])
            for emo, score in emotions.items():
                emotions_count[emo] = emotions_count.get(emo, 0) + score
        
        if a['engagement_score']:
            total_engagement += a['engagement_score']
    
    # Determine dominant
    dominant_sentiment = max(sentiments, key=sentiments.get) if any(sentiments.values()) else 'Neutral'
    dominant_emotion = max(emotions_count, key=emotions_count.get) if emotions_count else 'neutral'
    avg_engagement = total_engagement / len(analyses) if analyses else 0
    
    # Get word count
    cursor.execute('SELECT COUNT(*) as count FROM transcripts WHERE interview_id = ?', (interview_id,))
    word_count = cursor.fetchone()['count']
    
    # Get topics
    cursor.execute('''
        SELECT topics_json FROM analysis_results 
        WHERE interview_id = ? AND topics_json IS NOT NULL
        ORDER BY id DESC LIMIT 1
    ''', (interview_id,))
    topics_row = cursor.fetchone()
    topics_json = topics_row['topics_json'] if topics_row else '{}'
    
    # Save summary
    cursor.execute('''
        INSERT OR REPLACE INTO interview_summary 
        (interview_id, total_words, avg_sentiment_score, dominant_sentiment, 
         dominant_emotion, top_topics_json, avg_engagement_score)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        interview_id,
        word_count,
        sentiments.get(dominant_sentiment, 0),
        dominant_sentiment,
        dominant_emotion,
        topics_json,
        round(avg_engagement, 2)
    ))
    
    conn.commit()
    conn.close()
    
    return {
        'dominant_sentiment': dominant_sentiment,
        'dominant_emotion': dominant_emotion,
        'avg_engagement': avg_engagement,
        'total_words': word_count
    }


def get_dashboard_stats() -> Dict:
    """Get statistics for dashboard."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Total interviews
    cursor.execute('SELECT COUNT(*) as count FROM interviews')
    total_interviews = cursor.fetchone()['count']
    
    # Sentiment distribution
    cursor.execute('''
        SELECT dominant_sentiment, COUNT(*) as count 
        FROM interview_summary 
        GROUP BY dominant_sentiment
    ''')
    sentiment_dist = {row['dominant_sentiment']: row['count'] for row in cursor.fetchall()}
    
    # Calculate percentages
    total_with_sentiment = sum(sentiment_dist.values()) or 1
    positive_rate = round(sentiment_dist.get('Positive', 0) / total_with_sentiment * 100)
    
    # Average engagement
    cursor.execute('SELECT AVG(avg_engagement_score) as avg FROM interview_summary')
    avg_engagement = cursor.fetchone()['avg'] or 0
    
    # Unique students
    cursor.execute('SELECT COUNT(DISTINCT student_name) as count FROM interviews WHERE student_name != ""')
    unique_students = cursor.fetchone()['count']
    
    # Interview types
    cursor.execute('''
        SELECT interview_type, COUNT(*) as count 
        FROM interviews 
        GROUP BY interview_type
    ''')
    type_dist = {row['interview_type']: row['count'] for row in cursor.fetchall()}
    
    # Emotion distribution (aggregate)
    cursor.execute('SELECT emotions_json FROM analysis_results WHERE emotions_json IS NOT NULL')
    all_emotions = {}
    for row in cursor.fetchall():
        emotions = json.loads(row['emotions_json'])
        for emo, score in emotions.items():
            all_emotions[emo] = all_emotions.get(emo, 0) + score
    
    # Normalize emotions
    total_emotion = sum(all_emotions.values()) or 1
    emotion_dist = {k: round(v / total_emotion * 100) for k, v in all_emotions.items()}
    
    # Recent trend (last 4 weeks approximated)
    cursor.execute('''
        SELECT date(created_at) as date, dominant_sentiment, COUNT(*) as count
        FROM interview_summary
        GROUP BY date(created_at), dominant_sentiment
        ORDER BY date DESC
        LIMIT 28
    ''')
    trend_data = cursor.fetchall()
    
    conn.close()
    
    return {
        'total_interviews': total_interviews,
        'positive_rate': positive_rate,
        'avg_engagement': round(avg_engagement, 1),
        'unique_students': unique_students,
        'sentiment_distribution': sentiment_dist,
        'type_distribution': type_dist,
        'emotion_distribution': emotion_dist,
        'trend_data': [dict(row) for row in trend_data]
    }


def get_recent_interviews(limit: int = 10) -> List[Dict]:
    """Get recent interviews with summary."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            i.id, i.room_id, i.student_name, i.program, 
            i.interview_type, i.created_at,
            s.dominant_sentiment, s.avg_engagement_score
        FROM interviews i
        LEFT JOIN interview_summary s ON i.id = s.interview_id
        ORDER BY i.created_at DESC
        LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


# Initialize on import
init_database()
