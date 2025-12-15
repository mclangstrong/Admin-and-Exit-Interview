"""
AI-Driven Interview Analysis System
====================================
Main Flask application with:
- WebSocket signaling for WebRTC video conferencing
- Interview room management
- Recording and transcript handling
- Integration with NLP analysis pipeline
- SQLite database for persistent storage
- Trained mBERT model for sentiment analysis
- Role-based authentication (Admin/User)
"""

import os
import json
import uuid
from functools import wraps
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for, flash

from flask_socketio import SocketIO, emit, join_room, leave_room

# Import database module
import database as db

# ============================================================================
# APP CONFIGURATION
# ============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'interview-system-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'recordings'
app.config['TRANSCRIPT_FOLDER'] = 'transcripts'

# Initialize SocketIO for WebRTC signaling (using threading for Windows compatibility)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TRANSCRIPT_FOLDER'], exist_ok=True)

# In-memory storage for active rooms (maps room_id to interview_id)
active_rooms = {}

# Global analyzer instance (loaded once for performance)
_analyzer = None

def get_analyzer():
    """Get or create the NLP analyzer (uses trained mBERT model)."""
    global _analyzer
    if _analyzer is None:
        print("🤖 Loading trained mBERT model...")
        from nlp_utils import InterviewAnalyzer
        _analyzer = InterviewAnalyzer(
            load_sentiment=True,   # Load trained sentiment model
            load_emotion=True,     # Load emotion model
            load_keyphrase=True    # Load KeyBERT
        )
        print("✅ NLP analyzer ready with trained model!")
    return _analyzer


# ============================================================================
# AUTHENTICATION DECORATORS
# ============================================================================

def login_required(f):
    """Decorator to require login for a route."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """Decorator to require admin role for a route."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            return render_template('error.html', message='Access denied. Admin privileges required.'), 403
        return f(*args, **kwargs)
    return decorated_function


def user_required(f):
    """Decorator to require user (interviewer) role for a route."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# ============================================================================
# ROUTES - AUTHENTICATION
# ============================================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = db.verify_user(username, password)
        
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            session['full_name'] = user.get('full_name', '')
            
            # Redirect based on role
            if user['role'] == 'admin':
                return redirect(url_for('dashboard'))
            else:
                return redirect(url_for('user_home'))
        else:
            return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page for new users (interviewers only)."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        full_name = request.form.get('full_name')
        email = request.form.get('email', '')
        course = request.form.get('course', '')
        
        if password != confirm_password:
            return render_template('register.html', error='Passwords do not match')
        
        if len(password) < 6:
            return render_template('register.html', error='Password must be at least 6 characters')
        
        if not course:
            return render_template('register.html', error='Please select a course')
        
        user_id = db.create_user(username, password, role='user', full_name=full_name, email=email, course=course)
        
        if user_id:
            return redirect(url_for('login', success='Account created! Please sign in.'))
        else:
            return render_template('register.html', error='Username already exists')
    
    return render_template('register.html')


@app.route('/logout')
def logout():
    """Logout and clear session."""
    session.clear()
    return redirect(url_for('login'))


# ============================================================================
# ROUTES - PAGES (ROLE-BASED)
# ============================================================================

@app.route('/')
def index():
    """Landing page - redirect based on login status."""
    if 'user_id' in session:
        if session.get('role') == 'admin':
            return redirect(url_for('dashboard'))
        else:
            return redirect(url_for('user_home'))
    return redirect(url_for('login'))


@app.route('/home')
@user_required
def user_home():
    """Home page for interviewers (users)."""
    return render_template('index.html', user=session)


@app.route('/create-interview')
@user_required
def create_interview():
    """Create a new interview room (users/interviewers only)."""
    room_id = str(uuid.uuid4())[:8].upper()
    
    # Create in database
    interview_id = db.create_interview(room_id)
    
    # Store in active rooms
    active_rooms[room_id] = {
        'interview_id': interview_id,
        'created_at': datetime.now().isoformat(),
        'participants': [],
        'metadata': {},
        'status': 'waiting'
    }
    
    return render_template('interview_room.html', room_id=room_id, role='interviewer')


@app.route('/join/<room_id>')
def join_interview(room_id):
    """Join an existing interview room."""
    # Check active rooms first
    if room_id not in active_rooms:
        # Check database for existing interview
        interview = db.get_interview(room_id)
        if interview:
            active_rooms[room_id] = {
                'interview_id': interview['id'],
                'created_at': interview['created_at'],
                'participants': [],
                'metadata': {},
                'status': interview['status']
            }
        else:
            return render_template('error.html', message='Interview room not found.'), 404
    
    # Get student data from session for auto-fill
    student_data = None
    if 'user_id' in session:
        user_id = session['user_id']
        user = db.get_user_by_id(user_id)
        if user:
            student_data = {
                'name': user.get('full_name', ''),
                'course': user.get('course', ''),
                'cohort': datetime.now().strftime('%d/%m/%Y')  # Today's date in dd/mm/yyyy
            }
    
    return render_template('interview_room.html', room_id=room_id, role='student', student_data=student_data)


@app.route('/dashboard')
@admin_required
def dashboard():
    """Analytics dashboard with real data (admin only)."""
    return render_template('dashboard.html', user=session)


# ============================================================================
# ROUTES - API (Dashboard Data)
# ============================================================================

@app.route('/api/dashboard/stats', methods=['GET'])
def get_dashboard_stats():
    """Get dashboard statistics from database."""
    try:
        stats = db.get_dashboard_stats()
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dashboard/recent', methods=['GET'])
def get_recent_interviews():
    """Get recent interviews for dashboard table."""
    try:
        interviews = db.get_recent_interviews(limit=10)
        return jsonify({'success': True, 'interviews': interviews})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/interview/<int:interview_id>', methods=['GET'])
def get_interview_details(interview_id):
    """Get full interview details including transcript and analysis."""
    try:
        interview = db.get_interview_by_id(interview_id)
        if not interview:
            return jsonify({'error': 'Interview not found'}), 404
        
        transcript = db.get_transcript(interview_id)
        
        return jsonify({
            'success': True,
            'interview': interview,
            'transcript': transcript
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/interview/<int:interview_id>/details', methods=['GET'])
def get_full_interview_details(interview_id):
    """Get complete interview details with all analysis for View page."""
    try:
        interview = db.get_interview_by_id(interview_id)
        if not interview:
            return jsonify({'error': 'Interview not found'}), 404
        
        # Get transcript
        transcript_raw = db.get_transcript(interview_id)
        
        # Get all analysis results
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM analysis_results WHERE interview_id = ? ORDER BY id
        ''', (interview_id,))
        analysis = [dict(row) for row in cursor.fetchall()]
        
        # Get summary
        cursor.execute('''
            SELECT * FROM interview_summary WHERE interview_id = ?
        ''', (interview_id,))
        summary_row = cursor.fetchone()
        summary = dict(summary_row) if summary_row else None
        
        conn.close()
        
        # Merge transcript with analysis
        transcript = []
        for i, t in enumerate(transcript_raw):
            item = dict(t)
            # Find matching analysis
            if i < len(analysis):
                item['sentiment'] = analysis[i].get('sentiment_label')
                item['emotion'] = analysis[i].get('audio_emotion') or 'neutral'
                item['engagement'] = analysis[i].get('engagement_score')
            transcript.append(item)
        
        return jsonify({
            'success': True,
            'interview': interview,
            'transcript': transcript,
            'analysis': analysis,
            'summary': summary
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/interview/<int:interview_id>')
@admin_required
def view_interview_details(interview_id):
    """Page to view detailed interview analysis (admin only)."""
    return render_template('interview_details.html', interview_id=interview_id)


# ============================================================================
# ROUTES - API (Interview Operations)
# ============================================================================

@app.route('/api/rooms', methods=['GET'])
def get_rooms():
    """Get list of active rooms."""
    return jsonify({
        'rooms': [
            {'id': rid, 'status': data['status'], 'participants': len(data['participants'])}
            for rid, data in active_rooms.items()
        ]
    })


@app.route('/api/room/<room_id>/metadata', methods=['POST'])
def set_room_metadata(room_id):
    """Set interview metadata (program, cohort, date, etc.)."""
    if room_id not in active_rooms:
        return jsonify({'error': 'Room not found'}), 404
    
    data = request.json
    metadata = {
        'program': data.get('program', ''),
        'cohort': data.get('cohort', ''),
        'interview_type': data.get('interview_type', 'admission'),
        'student_name': data.get('student_name', ''),
        'interviewer_name': data.get('interviewer_name', ''),
    }
    
    active_rooms[room_id]['metadata'] = metadata
    
    # Update database
    db.update_interview(room_id, **metadata)
    
    return jsonify({'success': True, 'metadata': metadata})


@app.route('/api/upload-recording', methods=['POST'])
def upload_recording():
    """Upload recorded audio/video from interview."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    room_id = request.form.get('room_id', 'unknown')
    channel = request.form.get('channel', 'mixed')
    
    # Generate filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{room_id}_{channel}_{timestamp}.webm"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    audio_file.save(filepath)
    
    # Update database with recording path
    db.update_interview(room_id, recording_path=filepath)
    
    return jsonify({
        'success': True,
        'filename': filename,
        'path': filepath
    })


@app.route('/api/transcript', methods=['POST'])
def save_transcript_line():
    """Save a transcript line and analyze it."""
    data = request.json
    room_id = data.get('room_id')
    speaker = data.get('speaker', 'student')
    text = data.get('text', '')
    
    if not room_id or not text:
        return jsonify({'error': 'Missing room_id or text'}), 400
    
    if room_id not in active_rooms:
        return jsonify({'error': 'Room not found'}), 404
    
    interview_id = active_rooms[room_id]['interview_id']
    
    # Save transcript line
    transcript_id = db.add_transcript_line(interview_id, speaker, text)
    
    # Analyze with trained model
    try:
        print(f"\n{'='*60}")
        print(f"📝 ANALYZING TRANSCRIPT LINE")
        print(f"{'='*60}")
        print(f"Room ID: {room_id}")
        print(f"Speaker: {speaker}")
        print(f"Text: {text}")
        print(f"Interview ID: {interview_id}")
        
        analyzer = get_analyzer()
        print("🤖 Analyzer loaded, starting analysis...")
        
        analysis = analyzer.analyze(text)
        print(f"✅ Analysis complete!")
        print(f"   Sentiment: {analysis.get('sentiment', {}).get('label', 'N/A')}")
        print(f"   Emotions: {list(analysis.get('emotions', {}).keys())}")
        print(f"   Engagement: {analysis.get('engagement', {}).get('score', 'N/A')}")
        
        # Save analysis to database
        db.save_analysis(interview_id, analysis, transcript_id)
        print("💾 Analysis saved to database")
        
        response_data = {
            'success': True,
            'transcript_id': transcript_id,
            'analysis': analysis
        }
        print(f"📤 Sending response: success={response_data['success']}, has_analysis={analysis is not None}")
        print(f"{'='*60}\n")
        
        return jsonify(response_data)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ ANALYSIS ERROR")
        print(f"{'='*60}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'transcript_id': transcript_id,
            'analysis': None,
            'error': str(e)
        })


@app.route('/api/analyze', methods=['POST'])
def analyze_interview():
    """Analyze interview transcript with NLP pipeline (uses trained mBERT model)."""
    data = request.json
    transcript = data.get('transcript', '')
    room_id = data.get('room_id')
    
    if not transcript:
        return jsonify({'error': 'No transcript provided'}), 400
    
    try:
        # Use the trained model
        analyzer = get_analyzer()
        result = analyzer.analyze(transcript)
        
        # Save to database if room_id provided
        if room_id and room_id in active_rooms:
            interview_id = active_rooms[room_id]['interview_id']
            db.save_analysis(interview_id, result)
        
        return jsonify({'success': True, 'analysis': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze-topics', methods=['POST'])
def analyze_topics():
    """Extract topics from transcript."""
    data = request.json
    texts = data.get('texts', [])
    room_id = data.get('room_id')
    
    if not texts:
        return jsonify({'error': 'No texts provided'}), 400
    
    try:
        from topic_modeling import TopicModeler
        modeler = TopicModeler()
        topics = modeler.extract_topics(texts)
        
        # Save to database if room_id provided
        if room_id and room_id in active_rooms:
            interview_id = active_rooms[room_id]['interview_id']
            db.save_topics(interview_id, topics)
        
        return jsonify({'success': True, 'topics': topics})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze-emotion', methods=['POST'])
def analyze_emotion():
    """Analyze emotion from audio file."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    room_id = request.form.get('room_id')
    
    # Save temporarily
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_emotion.wav')
    audio_file.save(temp_path)
    
    try:
        from audio_emotion import AudioEmotionAnalyzer
        analyzer = AudioEmotionAnalyzer()
        result = analyzer.analyze(temp_path)
        os.remove(temp_path)
        
        # Save to database if room_id provided
        if room_id and room_id in active_rooms:
            interview_id = active_rooms[room_id]['interview_id']
            db.save_audio_emotion(interview_id, result)
        
        return jsonify({'success': True, 'emotion': result})
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500


@app.route('/api/interview/<room_id>/complete', methods=['POST'])
def complete_interview(room_id):
    """Mark interview as complete and calculate summary."""
    if room_id not in active_rooms:
        return jsonify({'error': 'Room not found'}), 404
    
    interview_id = active_rooms[room_id]['interview_id']
    
    # Update status
    db.update_interview(room_id, status='completed', ended_at=datetime.now().isoformat())
    
    # Calculate summary
    summary = db.calculate_interview_summary(interview_id)
    
    # Clean up active room
    del active_rooms[room_id]
    
    return jsonify({'success': True, 'summary': summary})


# ============================================================================
# WEBSOCKET EVENTS - WebRTC Signaling
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle new WebSocket connection."""
    print("="*60)
    print(f"🟢 CLIENT CONNECTED: {request.sid}")
    print("="*60)


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection."""
    print(f"Client disconnected: {request.sid}")
    for room_id, room_data in active_rooms.items():
        if request.sid in room_data['participants']:
            room_data['participants'].remove(request.sid)
            emit('participant-left', {'sid': request.sid}, room=room_id)


@socketio.on('join-room')
def handle_join_room(data):
    """Handle participant joining interview room."""
    room_id = data.get('room_id')
    role = data.get('role', 'student')
    
    print(f"🔌 join-room request: room={room_id}, role={role}, sid={request.sid}")
    
    # If room not in active_rooms, try to load from database
    if room_id not in active_rooms:
        interview = db.get_interview(room_id)
        if interview:
            active_rooms[room_id] = {
                'interview_id': interview['id'],
                'created_at': interview['created_at'],
                'participants': [],
                'metadata': {},
                'status': interview['status'] or 'waiting'
            }
            print(f"📦 Loaded room {room_id} from database")
        else:
            print(f"❌ Room {room_id} not found in database")
            emit('error', {'message': 'Room not found'})
            return
    
    # Join the SocketIO room
    join_room(room_id)
    
    # Add to participants list
    if request.sid not in active_rooms[room_id]['participants']:
        active_rooms[room_id]['participants'].append(request.sid)
    
    participant_count = len(active_rooms[room_id]['participants'])
    print(f"✅ User {request.sid} joined room {room_id} as {role} (total: {participant_count})")
    
    # Get user details for auto-fill if authenticated
    user_data = {}
    if 'user_id' in session:
        user = db.get_user_by_id(session['user_id'])
        if user:
            user_data = {
                'name': user.get('full_name', ''),
                'course': user.get('course', ''),
                'cohort': datetime.now().strftime('%d/%m/%Y')
            }
            print(f"   👤 With user data: {user_data['name']} / {user_data['course']}")

    # Notify all participants in the room
    emit('participant-joined', {
        'sid': request.sid,
        'role': role,
        'count': participant_count,
        'user_data': user_data
    }, room=room_id)


@socketio.on('leave-room')
def handle_leave_room(data):
    """Handle participant leaving interview room."""
    room_id = data.get('room_id')
    
    if room_id in active_rooms:
        leave_room(room_id)
        if request.sid in active_rooms[room_id]['participants']:
            active_rooms[room_id]['participants'].remove(request.sid)
        emit('participant-left', {'sid': request.sid}, room=room_id)


@socketio.on('offer')
def handle_offer(data):
    """Forward WebRTC offer to target peer."""
    room_id = data.get('room_id')
    print(f"📩 OFFER received from {request.sid} for room {room_id}")
    print(f"   Participants in room: {active_rooms.get(room_id, {}).get('participants', [])}")
    emit('offer', {
        'sdp': data.get('sdp'),
        'from': request.sid
    }, room=room_id, include_self=False)
    print(f"   OFFER broadcast to room {room_id}")


@socketio.on('answer')
def handle_answer(data):
    """Forward WebRTC answer to target peer."""
    room_id = data.get('room_id')
    print(f"📩 ANSWER received from {request.sid} for room {room_id}")
    emit('answer', {
        'sdp': data.get('sdp'),
        'from': request.sid
    }, room=room_id, include_self=False)
    print(f"   ANSWER broadcast to room {room_id}")


@socketio.on('ice-candidate')
def handle_ice_candidate(data):
    """Forward ICE candidate to target peer."""
    room_id = data.get('room_id')
    print(f"🧊 ICE candidate from {request.sid} for room {room_id}")
    emit('ice-candidate', {
        'candidate': data.get('candidate'),
        'from': request.sid
    }, room=room_id, include_self=False)


@socketio.on('interview-started')
def handle_interview_started(data):
    """Notify all participants that interview has started."""
    room_id = data.get('room_id')
    if room_id in active_rooms:
        active_rooms[room_id]['status'] = 'in-progress'
        started_at = datetime.now().isoformat()
        active_rooms[room_id]['started_at'] = started_at
        
        # Update database
        db.update_interview(room_id, status='in-progress', started_at=started_at)
        
    emit('interview-started', {'timestamp': datetime.now().isoformat()}, room=room_id)


@socketio.on('interview-ended')
def handle_interview_ended(data):
    """Notify all participants that interview has ended."""
    room_id = data.get('room_id')
    if room_id in active_rooms:
        active_rooms[room_id]['status'] = 'completed'
        ended_at = datetime.now().isoformat()
        active_rooms[room_id]['ended_at'] = ended_at
        
        # Update database and calculate summary
        interview_id = active_rooms[room_id]['interview_id']
        db.update_interview(room_id, status='completed', ended_at=ended_at)
        db.calculate_interview_summary(interview_id)
        
    emit('interview-ended', {'timestamp': datetime.now().isoformat()}, room=room_id)


@socketio.on('transcript-line')
def handle_transcript_line(data):
    """Broadcast transcript line to all participants in the room."""
    room_id = data.get('room_id')
    print(f"📝 Broadcasting transcript line to room {room_id}: {data.get('text')[:50]}...")
    
    # Broadcast to everyone in the room
    emit('transcript-line', data, room=room_id)


@socketio.on('analysis-update')
def handle_analysis_update(data):
    """Broadcast live analysis data to all participants in the room."""
    room_id = data.get('room_id')
    speaker = data.get('speaker', 'unknown')
    print(f"📊 Broadcasting analysis update from {speaker} to room {room_id}")
    
    # Broadcast to everyone in the room (interviewer will receive it)
    emit('analysis-update', data, room=room_id)


# ============================================================================
# MAIN
# ============================================================================

def generate_ssl_certificate():
    """Generate self-signed SSL certificate if it doesn't exist."""
    import socket
    from OpenSSL import crypto
    
    # Get local IP address
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print("🔐 Generating SSL certificate...")
    print(f"   Hostname: {hostname}")
    print(f"   Local IP: {local_ip}")
    
    # Create a key pair
    key = crypto.PKey()
    key.generate_key(crypto.TYPE_RSA, 2048)
    
    # Create a self-signed cert
    cert = crypto.X509()
    cert.get_subject().C = "PH"
    cert.get_subject().ST = "Metro Manila"
    cert.get_subject().L = "Manila"
    cert.get_subject().O = "InterviewAI"
    cert.get_subject().OU = "Development"
    cert.get_subject().CN = local_ip
    
    # Add Subject Alternative Names (SANs)
    san_list = [
        f"DNS:localhost",
        f"DNS:{hostname}",
        f"IP:127.0.0.1",
        f"IP:{local_ip}"
    ]
    san_extension = crypto.X509Extension(
        b"subjectAltName",
        False,
        ", ".join(san_list).encode()
    )
    cert.add_extensions([san_extension])
    
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(365*24*60*60)  # Valid for 1 year
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(key)
    cert.sign(key, 'sha256')
    
    # Save certificate and key
    with open("cert.pem", "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    
    with open("key.pem", "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))
    
    print("✅ SSL certificate generated successfully!")
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("   AI-DRIVEN INTERVIEW ANALYSIS SYSTEM")
    print("=" * 60)
    
    # Pre-load the analyzer
    get_analyzer()
    
    # Check if SSL certificates exist, generate if not
    import os
    cert_file = 'cert.pem'
    key_file = 'key.pem'
    
    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        print("📝 SSL certificates not found. Generating...")
        try:
            generate_ssl_certificate()
        except Exception as e:
            print(f"⚠️  Failed to generate certificate: {e}")
            print("   Falling back to HTTP mode")
    
    if os.path.exists(cert_file) and os.path.exists(key_file):
        print("🔒 HTTPS mode enabled")
        print("🌐 Server URLs:")
        print("   - https://localhost:5000")
        print("   - https://192.168.123.40:5000")
        print("📹 Video conferencing ready (camera/mic enabled)")
        print("🤖 NLP analysis pipeline with trained mBERT model")
        print("💾 SQLite database connected")
        print("=" * 60)
        print("⚠️  Note: You'll see a security warning in browser.")
        print("   Click 'Advanced' → 'Proceed to site' (safe for local network)")
        print("=" * 60)
        
        import ssl
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_file, key_file)
        
        socketio.run(app, host='0.0.0.0', port=5000, debug=True, ssl_context=context)
    else:
        print("⚠️  HTTP mode")
        print("🌐 Server URLs:")
        print("   - http://localhost:5000 (camera/mic works)")
        print("   - http://192.168.123.40:5000 (camera/mic blocked by browser)")
        print("📹 Video conferencing ready")
        print("🤖 NLP analysis pipeline with trained mBERT model")
        print("💾 SQLite database connected")
        print("=" * 60)
        
        socketio.run(app, host='0.0.0.0', port=5000, debug=True)

