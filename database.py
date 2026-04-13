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
import secrets
from datetime import datetime
from typing import Dict, List, Any, Optional
from werkzeug.security import generate_password_hash, check_password_hash

# Database path
DB_PATH = "interviews.db"


# Allowed columns for dynamic UPDATE queries (Warning #5)
ALLOWED_UPDATE_COLUMNS = frozenset([
    'student_name', 'interviewer_name', 'program', 'cohort',
    'interview_type', 'status', 'started_at', 'ended_at',
    'duration_seconds', 'recording_path'
])


def get_connection():
    """Get database connection with WAL mode and busy timeout."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
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
    
    # Topic classifications table (for dashboard aggregation)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS topic_classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interview_id INTEGER UNIQUE NOT NULL,
            academics_percent REAL DEFAULT 0,
            career_percent REAL DEFAULT 0,
            faculty_percent REAL DEFAULT 0,
            infrastructure_percent REAL DEFAULT 0,
            mental_health_percent REAL DEFAULT 0,
            social_percent REAL DEFAULT 0,
            technology_percent REAL DEFAULT 0,
            total_sentences INTEGER DEFAULT 0,
            classified_sentences INTEGER DEFAULT 0,
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
    
    # Create default admin if not exists (Warning #3: random password)
    cursor.execute('SELECT COUNT(*) as count FROM users WHERE role = "admin"')
    if cursor.fetchone()['count'] == 0:
        random_password = secrets.token_urlsafe(16)
        default_password = generate_password_hash(random_password)
        cursor.execute('''
            INSERT INTO users (username, password_hash, role, full_name)
            VALUES (?, ?, ?, ?)
        ''', ('admin', default_password, 'admin', 'System Administrator'))
        print(f"\n{'=' * 60}")
        print(f"  ⚠️  DEFAULT ADMIN CREATED")
        print(f"  Username: admin")
        print(f"  Password: {random_password}")
        print(f"  CHANGE THIS PASSWORD IMMEDIATELY!")
        print(f"{'=' * 60}\n")
    
    conn.commit()
    conn.close()


# ============================================================================
# USER OPERATIONS
# ============================================================================

def create_user(username: str, password: str, role: str = 'user', 
                full_name: str = '', email: str = '', course: str = '') -> Optional[int]:
    """Create a new user."""
    conn = get_connection()
    cursor = conn.cursor()
    
    password_hash = generate_password_hash(password)
    
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
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, username, role, full_name, email, password_hash
        FROM users 
        WHERE username = ?
    ''', (username,))
    
    row = cursor.fetchone()
    
    if row and check_password_hash(row['password_hash'], password):
        # Update last login
        cursor.execute('''
            UPDATE users SET last_login = ? WHERE id = ?
        ''', (datetime.now().isoformat(), row['id']))
        conn.commit()
        conn.close()
        user_data = dict(row)
        del user_data['password_hash']  # Don't leak hash
        return user_data
    
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
        if key in ALLOWED_UPDATE_COLUMNS:
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


def get_interviews_for_export(days: int = None) -> List[Dict]:
    """Get interviews for report export with optional date filtering.
    
    Args:
        days: Number of days to look back (7, 30, 90). None = all time.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    if days:
        cursor.execute('''
            SELECT i.id, i.created_at, i.student_name, i.program, i.cohort,
                   i.interview_type, i.status, i.duration_seconds,
                   s.dominant_sentiment, s.dominant_emotion, s.avg_engagement_score,
                   s.total_words
            FROM interviews i
            LEFT JOIN interview_summary s ON i.id = s.interview_id
            WHERE i.created_at >= datetime('now', ? || ' days')
            ORDER BY i.created_at DESC
        ''', (f'-{days}',))
    else:
        cursor.execute('''
            SELECT i.id, i.created_at, i.student_name, i.program, i.cohort,
                   i.interview_type, i.status, i.duration_seconds,
                   s.dominant_sentiment, s.dominant_emotion, s.avg_engagement_score,
                   s.total_words
            FROM interviews i
            LEFT JOIN interview_summary s ON i.id = s.interview_id
            ORDER BY i.created_at DESC
        ''')
    
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
        WHERE id = (
            SELECT id FROM analysis_results
            WHERE interview_id = ? AND audio_emotion IS NULL
            ORDER BY id DESC LIMIT 1
        )
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
    sentiments = {'Positive': 0, 'Negative': 0}
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
    dominant_sentiment = max(sentiments, key=sentiments.get) if any(sentiments.values()) else 'Positive'
    dominant_emotion = max(emotions_count, key=emotions_count.get) if emotions_count else 'neutral'
    avg_engagement = total_engagement / len(analyses) if analyses else 0
    
    # Get actual word count by summing words in each transcript line
    cursor.execute('SELECT COALESCE(SUM(LENGTH(text) - LENGTH(REPLACE(text, " ", "")) + 1), 0) as count FROM transcripts WHERE interview_id = ?', (interview_id,))
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


def get_dashboard_stats(sentiment_filter: str = 'all', trend_days: int = 30) -> Dict:
    """Get statistics for dashboard with optional filters."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Calculate date filters
    from datetime import datetime, timedelta
    now = datetime.now()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    trend_start = (now - timedelta(days=trend_days)).strftime('%Y-%m-%d')
    
    # Total interviews
    cursor.execute('SELECT COUNT(*) as count FROM interviews')
    total_interviews = cursor.fetchone()['count']
    
    # Sentiment distribution - with optional month filter
    if sentiment_filter == 'month':
        # Filter to current month - from interview_summary
        cursor.execute('''
            SELECT dominant_sentiment, COUNT(*) as count 
            FROM interview_summary 
            WHERE dominant_sentiment IS NOT NULL
            AND created_at >= ?
            GROUP BY dominant_sentiment
        ''', (month_start.isoformat(),))
        sentiment_dist = {row['dominant_sentiment']: row['count'] for row in cursor.fetchall()}
        
        # Fallback to analysis_results
        if not sentiment_dist:
            cursor.execute('''
                SELECT sentiment_label, COUNT(*) as count 
                FROM analysis_results 
                WHERE sentiment_label IS NOT NULL
                AND created_at >= ?
                GROUP BY sentiment_label
            ''', (month_start.isoformat(),))
            sentiment_dist = {row['sentiment_label']: row['count'] for row in cursor.fetchall()}
    else:
        # All time - from interview_summary
        cursor.execute('''
            SELECT dominant_sentiment, COUNT(*) as count 
            FROM interview_summary 
            WHERE dominant_sentiment IS NOT NULL
            GROUP BY dominant_sentiment
        ''')
        sentiment_dist = {row['dominant_sentiment']: row['count'] for row in cursor.fetchall()}
        
        # Fallback to analysis_results
        if not sentiment_dist:
            cursor.execute('''
                SELECT sentiment_label, COUNT(*) as count 
                FROM analysis_results 
                WHERE sentiment_label IS NOT NULL
                GROUP BY sentiment_label
            ''')
            sentiment_dist = {row['sentiment_label']: row['count'] for row in cursor.fetchall()}
    
    # Calculate percentages
    total_with_sentiment = sum(sentiment_dist.values()) or 1
    positive_rate = round(sentiment_dist.get('Positive', 0) / total_with_sentiment * 100)
    
    # Average engagement - try interview_summary first, fallback to analysis_results
    cursor.execute('SELECT AVG(avg_engagement_score) as avg FROM interview_summary')
    avg_engagement = cursor.fetchone()['avg']
    
    # Fallback: if no summary data, aggregate from analysis_results
    if avg_engagement is None:
        cursor.execute('SELECT AVG(engagement_score) as avg FROM analysis_results WHERE engagement_score IS NOT NULL')
        avg_engagement = cursor.fetchone()['avg'] or 0
    
    # Unique students
    cursor.execute('SELECT COUNT(DISTINCT student_name) as count FROM interviews WHERE student_name != ""')
    unique_students = cursor.fetchone()['count']
    
    # Sentiment comparison by interview type (admission vs exit)
    cursor.execute('''
        SELECT 
            i.interview_type,
            s.dominant_sentiment,
            COUNT(*) as count
        FROM interviews i
        JOIN interview_summary s ON i.id = s.interview_id
        WHERE s.dominant_sentiment IS NOT NULL
        GROUP BY i.interview_type, s.dominant_sentiment
    ''')
    type_sentiment_rows = cursor.fetchall()
    
    # Build structured comparison: { admission: {Positive: N, Negative: N}, exit: {Positive: N, Negative: N} }
    type_sentiment = {
        'admission': {'Positive': 0, 'Negative': 0},
        'exit': {'Positive': 0, 'Negative': 0}
    }
    for row in type_sentiment_rows:
        itype = row['interview_type'] or 'admission'
        sentiment = row['dominant_sentiment'] or 'Positive'
        if itype in type_sentiment and sentiment in type_sentiment[itype]:
            type_sentiment[itype][sentiment] = row['count']
    
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
    
    # Trend data - filtered by days (from interview_summary)
    cursor.execute('''
        SELECT date(created_at) as date, dominant_sentiment, COUNT(*) as count
        FROM interview_summary
        WHERE date(created_at) >= ?
        GROUP BY date(created_at), dominant_sentiment
        ORDER BY date ASC
    ''', (trend_start,))
    trend_data = cursor.fetchall()
    
    # Fallback: if no summary data, use analysis_results
    if not trend_data:
        cursor.execute('''
            SELECT date(created_at) as date, sentiment_label as dominant_sentiment, COUNT(*) as count
            FROM analysis_results
            WHERE date(created_at) >= ?
            AND sentiment_label IS NOT NULL
            GROUP BY date(created_at), sentiment_label
            ORDER BY date ASC
        ''', (trend_start,))
        trend_data = cursor.fetchall()
    
    # Topic distribution from classified interviews
    cursor.execute('''
        SELECT 
            AVG(academics_percent) as academics,
            AVG(career_percent) as career,
            AVG(faculty_percent) as faculty,
            AVG(infrastructure_percent) as infrastructure,
            AVG(mental_health_percent) as mental_health,
            AVG(social_percent) as social,
            AVG(technology_percent) as technology
        FROM topic_classifications
    ''')
    topic_row = cursor.fetchone()
    
    topic_dist = {}
    if topic_row:
        topic_dist = {
            'Academics': round(topic_row['academics'] or 0, 1),
            'Career': round(topic_row['career'] or 0, 1),
            'Faculty': round(topic_row['faculty'] or 0, 1),
            'Infrastructure': round(topic_row['infrastructure'] or 0, 1),
            'Mental health': round(topic_row['mental_health'] or 0, 1),
            'Social': round(topic_row['social'] or 0, 1),
            'Technology': round(topic_row['technology'] or 0, 1)
        }
    
    conn.close()
    
    return {
        'total_interviews': total_interviews,
        'positive_rate': positive_rate,
        'avg_engagement': round(avg_engagement, 1),
        'unique_students': unique_students,
        'sentiment_distribution': sentiment_dist,
        'type_sentiment': type_sentiment,
        'emotion_distribution': emotion_dist,
        'topic_distribution': topic_dist,
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


# ============================================================================
# TOPIC CLASSIFICATION
# ============================================================================

def save_topic_classification(interview_id: int, topics: Dict, total_sentences: int, classified_sentences: int) -> bool:
    """Save topic classification results for an interview."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO topic_classifications (
            interview_id, academics_percent, career_percent, faculty_percent,
            infrastructure_percent, mental_health_percent, social_percent,
            technology_percent, total_sentences, classified_sentences
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        interview_id,
        topics.get('Academics', 0),
        topics.get('Career', 0),
        topics.get('Faculty', 0),
        topics.get('Infrastructure', 0),
        topics.get('Mental health', 0),
        topics.get('Social', 0),
        topics.get('Technology', 0),
        total_sentences,
        classified_sentences
    ))
    
    conn.commit()
    conn.close()
    return True


def get_topic_classification(interview_id: int) -> Optional[Dict]:
    """Get saved topic classification for a specific interview."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM topic_classifications WHERE interview_id = ?
    ''', (interview_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'success': True,
            'topics': {
                'Academics': row['academics_percent'],
                'Career': row['career_percent'],
                'Faculty': row['faculty_percent'],
                'Infrastructure': row['infrastructure_percent'],
                'Mental health': row['mental_health_percent'],
                'Social': row['social_percent'],
                'Technology': row['technology_percent']
            },
            'total_sentences': row['total_sentences'],
            'classified_sentences': row['classified_sentences']
        }
    
    return None


def get_aggregated_topic_distribution() -> Dict:
    """Get aggregated topic distribution across all classified interviews."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            AVG(academics_percent) as academics,
            AVG(career_percent) as career,
            AVG(faculty_percent) as faculty,
            AVG(infrastructure_percent) as infrastructure,
            AVG(mental_health_percent) as mental_health,
            AVG(social_percent) as social,
            AVG(technology_percent) as technology,
            COUNT(*) as total_classified
        FROM topic_classifications
    ''')
    
    row = cursor.fetchone()
    conn.close()
    
    if row and row['total_classified'] > 0:
        return {
            'Academics': round(row['academics'] or 0, 1),
            'Career': round(row['career'] or 0, 1),
            'Faculty': round(row['faculty'] or 0, 1),
            'Infrastructure': round(row['infrastructure'] or 0, 1),
            'Mental health': round(row['mental_health'] or 0, 1),
            'Social': round(row['social'] or 0, 1),
            'Technology': round(row['technology'] or 0, 1),
            'total_classified': row['total_classified']
        }
    
    return {
        'Academics': 0, 'Career': 0, 'Faculty': 0, 'Infrastructure': 0,
        'Mental health': 0, 'Social': 0, 'Technology': 0,
        'total_classified': 0
    }


# Initialize on import
init_database()
