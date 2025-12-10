/**
 * WebRTC Interview System
 * =======================
 * Handles:
 * - Peer-to-peer video/audio connection
 * - Dual-channel audio recording
 * - Real-time signaling via WebSocket
 * - Speech recognition for live transcription
 */

// ============================================================================
// Configuration
// ============================================================================

const ROOM_ID = document.getElementById('room-id')?.value;
const USER_ROLE = document.getElementById('user-role')?.value;

const ICE_SERVERS = {
    iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' },
    ]
};

// ============================================================================
// State
// ============================================================================

let socket = null;
let peerConnection = null;
let localStream = null;
let remoteStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;
let isMuted = false;
let isCameraOff = false;
let timerInterval = null;
let recordingStartTime = null;
let speechRecognition = null;
let transcript = [];

// ============================================================================
// DOM Elements
// ============================================================================

const localVideo = document.getElementById('local-video');
const remoteVideo = document.getElementById('remote-video');
const remotePlaceholder = document.getElementById('remote-placeholder');
const recordingStatus = document.getElementById('recording-status');
const timerDisplay = document.getElementById('timer');
const connectionStatus = document.getElementById('connection-status');
const transcriptContent = document.getElementById('transcript-content');
const liveSentiment = document.getElementById('live-sentiment');
const liveEmotion = document.getElementById('live-emotion');
const engagementBar = document.getElementById('engagement-bar');

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    console.log('🎬 Initializing Interview Room...');
    console.log(`📍 Room: ${ROOM_ID}, Role: ${USER_ROLE}`);

    // Check for secure context
    if (!window.isSecureContext) {
        console.warn('⚠️ Not a secure context! Camera/mic may not work.');
        console.warn('   Use localhost or HTTPS for full functionality.');
    }

    try {
        await initializeMedia();
    } catch (error) {
        console.error('Media initialization failed:', error);
        // Continue without local video - user can still see remote video
        alert('Camera/microphone access failed. You may need to use localhost or HTTPS.\n\nYou can still see the other participant.');
    }

    // Always initialize socket even if media fails
    initializeSocket();
    initializeSpeechRecognition();
    setupEventListeners();
});

async function initializeMedia() {
    console.log('📹 Requesting media access...');

    // Check if mediaDevices is available
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('getUserMedia not available. Use HTTPS or localhost.');
    }

    localStream = await navigator.mediaDevices.getUserMedia({
        video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: 'user'
        },
        audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true
        }
    });

    localVideo.srcObject = localStream;
    console.log('✅ Local media ready');
}

function initializeSocket() {
    console.log('🔌 Connecting to signaling server...');

    // Connect to server using IP address for network access
    const serverUrl = 'http://192.168.1.10:5000';
    console.log('📡 Server URL:', serverUrl);

    socket = io(serverUrl, {
        transports: ['websocket', 'polling'],
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000
    });

    socket.on('connect', () => {
        console.log('✅ Connected to server, socket ID:', socket.id);
        updateConnectionStatus('connected', 'Connected');

        // Join the room
        console.log('📤 Emitting join-room:', { room_id: ROOM_ID, role: USER_ROLE });
        socket.emit('join-room', { room_id: ROOM_ID, role: USER_ROLE });
    });

    socket.on('connect_error', (error) => {
        console.error('❌ Socket connection error:', error);
        updateConnectionStatus('disconnected', 'Connection Error');
    });

    socket.on('error', (data) => {
        console.error('❌ Server error:', data);
        alert('Error: ' + (data.message || 'Unknown error'));
    });

    socket.on('disconnect', (reason) => {
        console.log('❌ Disconnected from server:', reason);
        updateConnectionStatus('disconnected', 'Disconnected');
    });

    socket.on('participant-joined', async (data) => {
        console.log('👤 Participant joined:', data);
        console.log('   My socket ID:', socket.id);
        console.log('   Joining socket ID:', data.sid);
        console.log('   Total participants:', data.count);

        // If there are now 2 participants and I am NOT the one who just joined
        // then I (the first participant) should create the offer
        if (data.count === 2 && data.sid !== socket.id) {
            console.log('🔗 I am the first participant - creating offer for newcomer...');
            await createPeerConnection();
            await createOffer();
        }
    });

    socket.on('participant-left', (data) => {
        console.log('👋 Participant left:', data);
        if (remoteVideo) {
            remoteVideo.srcObject = null;
            remotePlaceholder.style.display = 'flex';
        }
    });

    socket.on('offer', async (data) => {
        console.log('📨 Received offer');
        await createPeerConnection();
        await peerConnection.setRemoteDescription(new RTCSessionDescription(data.sdp));
        await createAnswer();
    });

    socket.on('answer', async (data) => {
        console.log('📨 Received answer');
        await peerConnection.setRemoteDescription(new RTCSessionDescription(data.sdp));
    });

    socket.on('ice-candidate', async (data) => {
        console.log('🧊 Received ICE candidate');
        if (peerConnection) {
            await peerConnection.addIceCandidate(new RTCIceCandidate(data.candidate));
        }
    });

    socket.on('interview-started', (data) => {
        console.log('🎬 Interview started:', data);
    });

    socket.on('interview-ended', (data) => {
        console.log('🛑 Interview ended by other participant:', data);

        // Stop recording if active
        if (isRecording) {
            stopRecording();
        }

        // Stop speech recognition
        if (speechRecognition) {
            speechRecognition.stop();
        }

        // Close peer connection
        if (peerConnection) {
            peerConnection.close();
            peerConnection = null;
        }

        // Stop all local tracks
        if (localStream) {
            localStream.getTracks().forEach(track => track.stop());
        }

        // Show message and redirect
        alert('The interview has ended. Redirecting to home page...');
        window.location.href = '/home';
    });
}

// ============================================================================
// WebRTC Peer Connection
// ============================================================================

async function createPeerConnection() {
    console.log('🔗 Creating peer connection...');

    peerConnection = new RTCPeerConnection(ICE_SERVERS);

    // Add local tracks
    localStream.getTracks().forEach(track => {
        peerConnection.addTrack(track, localStream);
    });

    // Handle remote stream
    peerConnection.ontrack = (event) => {
        console.log('📹 Received remote track');
        remoteStream = event.streams[0];
        remoteVideo.srcObject = remoteStream;
        remotePlaceholder.style.display = 'none';
        document.getElementById('remote-name').textContent =
            USER_ROLE === 'interviewer' ? 'Student' : 'Interviewer';
    };

    // Handle ICE candidates
    peerConnection.onicecandidate = (event) => {
        if (event.candidate) {
            socket.emit('ice-candidate', {
                room_id: ROOM_ID,
                candidate: event.candidate
            });
        }
    };

    // Connection state changes
    peerConnection.onconnectionstatechange = () => {
        console.log('Connection state:', peerConnection.connectionState);
        switch (peerConnection.connectionState) {
            case 'connected':
                updateConnectionStatus('connected', 'Connected');
                break;
            case 'disconnected':
            case 'failed':
                updateConnectionStatus('disconnected', 'Connection Lost');
                break;
        }
    };
}

async function createOffer() {
    console.log('📤 Creating offer...');
    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);

    socket.emit('offer', {
        room_id: ROOM_ID,
        sdp: offer
    });
}

async function createAnswer() {
    console.log('📤 Creating answer...');
    const answer = await peerConnection.createAnswer();
    await peerConnection.setLocalDescription(answer);

    socket.emit('answer', {
        room_id: ROOM_ID,
        sdp: answer
    });
}

// ============================================================================
// Recording
// ============================================================================

function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

async function startRecording() {
    console.log('⏺️ Starting recording...');

    recordedChunks = [];

    // Create a combined stream for recording
    const audioContext = new AudioContext();
    const destination = audioContext.createMediaStreamDestination();

    // Add local audio
    if (localStream) {
        const localSource = audioContext.createMediaStreamSource(
            new MediaStream(localStream.getAudioTracks())
        );
        localSource.connect(destination);
    }

    // Add remote audio
    if (remoteStream) {
        const remoteSource = audioContext.createMediaStreamSource(
            new MediaStream(remoteStream.getAudioTracks())
        );
        remoteSource.connect(destination);
    }

    // Create combined stream with video
    const videoTrack = localStream?.getVideoTracks()[0];
    const combinedStream = new MediaStream([
        ...destination.stream.getAudioTracks(),
        ...(videoTrack ? [videoTrack] : [])
    ]);

    mediaRecorder = new MediaRecorder(combinedStream, {
        mimeType: 'video/webm;codecs=vp9,opus'
    });

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            recordedChunks.push(event.data);
        }
    };

    mediaRecorder.onstop = () => {
        console.log('⏹️ Recording stopped');
        saveRecording();
    };

    mediaRecorder.start(1000); // Capture in 1-second chunks
    isRecording = true;
    recordingStartTime = Date.now();

    // Update UI
    updateRecordingUI(true);
    startTimer();

    // Start speech recognition
    if (speechRecognition) {
        speechRecognition.start();
    }

    // Notify server
    socket.emit('interview-started', { room_id: ROOM_ID });
}

function stopRecording() {
    console.log('⏹️ Stopping recording...');

    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }

    isRecording = false;

    // Update UI
    updateRecordingUI(false);
    stopTimer();

    // Stop speech recognition
    if (speechRecognition) {
        speechRecognition.stop();
    }
}

async function saveRecording() {
    const blob = new Blob(recordedChunks, { type: 'video/webm' });

    // Upload to server
    const formData = new FormData();
    formData.append('audio', blob, 'interview.webm');
    formData.append('room_id', ROOM_ID);
    formData.append('channel', 'mixed');

    try {
        const response = await fetch('/api/upload-recording', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        console.log('✅ Recording saved:', result);
        alert('Recording saved successfully!');
    } catch (error) {
        console.error('Failed to save recording:', error);

        // Fallback: download locally
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `interview_${ROOM_ID}_${Date.now()}.webm`;
        a.click();
    }
}

function downloadRecording() {
    if (recordedChunks.length === 0) {
        alert('No recording available. Start and stop a recording first.');
        return;
    }

    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `interview_${ROOM_ID}_${Date.now()}.webm`;
    a.click();
}

// ============================================================================
// Speech Recognition (Student Only)
// ============================================================================

function initializeSpeechRecognition() {
    // Only students can have their speech transcribed
    if (USER_ROLE !== 'student') {
        console.log('🎤 Speech recognition disabled for interviewer (only student speech is transcribed)');
        return;
    }

    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        console.warn('Speech recognition not supported');
        return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    speechRecognition = new SpeechRecognition();

    speechRecognition.continuous = true;
    speechRecognition.interimResults = true;
    speechRecognition.lang = 'en-US'; // Will also pick up Tagalog reasonably

    speechRecognition.onresult = (event) => {
        const results = event.results;
        const lastResult = results[results.length - 1];

        if (lastResult.isFinal) {
            const text = lastResult[0].transcript;
            addTranscriptLine('student', text);
            analyzeTranscriptLine(text);
        }
    };

    speechRecognition.onerror = (event) => {
        console.warn('Speech recognition error:', event.error);
        if (event.error !== 'no-speech') {
            // Restart on recoverable errors
            setTimeout(() => {
                if (isRecording && speechRecognition) {
                    speechRecognition.start();
                }
            }, 1000);
        }
    };

    speechRecognition.onend = () => {
        // Auto-restart if still recording
        if (isRecording) {
            speechRecognition.start();
        }
    };

    console.log('🎤 Speech recognition initialized (Student only)');
}

function addTranscriptLine(speaker, text) {
    const line = { speaker, text, timestamp: new Date().toISOString() };
    transcript.push(line);

    // Update UI
    const placeholder = transcriptContent.querySelector('.transcript-placeholder');
    if (placeholder) {
        placeholder.remove();
    }

    const lineEl = document.createElement('div');
    lineEl.className = 'transcript-line';
    lineEl.innerHTML = `
        <div class="transcript-speaker">${speaker === 'interviewer' ? '👔 Interviewer' : '🎓 Student'}</div>
        <div class="transcript-text">${text}</div>
    `;
    transcriptContent.appendChild(lineEl);
    transcriptContent.scrollTop = transcriptContent.scrollHeight;
}

async function analyzeTranscriptLine(text) {
    try {
        // Save transcript line to database AND analyze with trained model
        const response = await fetch('/api/transcript', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                room_id: ROOM_ID,
                speaker: USER_ROLE,
                text: text
            })
        });

        const result = await response.json();

        if (result.success && result.analysis) {
            const analysis = result.analysis;

            // Update sentiment display
            liveSentiment.textContent = analysis.sentiment?.label || '--';
            liveSentiment.className = 'analysis-value ' + (analysis.sentiment?.label?.toLowerCase() || '');

            // Update emotion display
            if (analysis.emotions) {
                const topEmotion = Object.entries(analysis.emotions)
                    .sort((a, b) => b[1] - a[1])[0];
                liveEmotion.textContent = topEmotion ? topEmotion[0] : '--';
            }

            // Update engagement bar
            if (analysis.engagement) {
                engagementBar.style.width = `${analysis.engagement.score * 10}%`;
            }
        }
    } catch (error) {
        console.warn('Analysis failed:', error);
    }
}

// ============================================================================
// Controls
// ============================================================================

function toggleMic() {
    const audioTracks = localStream?.getAudioTracks();
    if (audioTracks && audioTracks.length > 0) {
        isMuted = !isMuted;
        audioTracks.forEach(track => {
            track.enabled = !isMuted;
        });

        const micBtn = document.getElementById('mic-btn');
        micBtn.classList.toggle('active', isMuted);
        micBtn.querySelector('.btn-label').textContent = isMuted ? 'Unmute' : 'Mute';
    }
}

function toggleCamera() {
    const videoTracks = localStream?.getVideoTracks();
    if (videoTracks && videoTracks.length > 0) {
        isCameraOff = !isCameraOff;
        videoTracks.forEach(track => {
            track.enabled = !isCameraOff;
        });

        const cameraBtn = document.getElementById('camera-btn');
        cameraBtn.classList.toggle('active', isCameraOff);
        cameraBtn.querySelector('.btn-label').textContent = isCameraOff ? 'Show' : 'Camera';
    }
}

function toggleTranscript() {
    const sidebar = document.querySelector('.interview-sidebar');
    sidebar.classList.toggle('hidden');
}

function endInterview() {
    if (confirm('Are you sure you want to end this interview?')) {
        // Stop recording if active
        if (isRecording) {
            stopRecording();
        }

        // Notify server
        socket.emit('interview-ended', { room_id: ROOM_ID });
        socket.emit('leave-room', { room_id: ROOM_ID });

        // Close peer connection
        if (peerConnection) {
            peerConnection.close();
        }

        // Stop all tracks
        if (localStream) {
            localStream.getTracks().forEach(track => track.stop());
        }

        // Redirect to dashboard
        window.location.href = '/dashboard';
    }
}

// ============================================================================
// UI Helpers
// ============================================================================

function updateRecordingUI(recording) {
    const recordBtn = document.getElementById('record-btn');

    if (recording) {
        recordingStatus.classList.add('recording');
        recordingStatus.querySelector('.status-text').textContent = 'Recording';
        recordBtn.classList.add('recording');
        recordBtn.querySelector('.btn-label').textContent = 'Stop Recording';
        recordBtn.querySelector('.btn-icon').textContent = '⏹️';
    } else {
        recordingStatus.classList.remove('recording');
        recordingStatus.querySelector('.status-text').textContent = 'Not Recording';
        recordBtn.classList.remove('recording');
        recordBtn.querySelector('.btn-label').textContent = 'Start Recording';
        recordBtn.querySelector('.btn-icon').textContent = '⏺️';
    }
}

function updateConnectionStatus(status, text) {
    const indicator = connectionStatus.querySelector('.status-indicator');
    indicator.className = 'status-indicator ' + status;
    connectionStatus.childNodes[1].textContent = ' ' + text;
}

function startTimer() {
    timerInterval = setInterval(() => {
        const elapsed = Date.now() - recordingStartTime;
        const hours = Math.floor(elapsed / 3600000);
        const minutes = Math.floor((elapsed % 3600000) / 60000);
        const seconds = Math.floor((elapsed % 60000) / 1000);

        timerDisplay.textContent =
            String(hours).padStart(2, '0') + ':' +
            String(minutes).padStart(2, '0') + ':' +
            String(seconds).padStart(2, '0');
    }, 1000);
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

function copyRoomCode() {
    navigator.clipboard.writeText(ROOM_ID).then(() => {
        alert('Room code copied: ' + ROOM_ID);
    });
}

// ============================================================================
// Event Listeners
// ============================================================================

function setupEventListeners() {
    // Metadata form
    const metadataForm = document.getElementById('metadata-form');
    if (metadataForm) {
        metadataForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const metadata = {
                interview_type: document.getElementById('interview-type').value,
                program: document.getElementById('program').value,
                cohort: document.getElementById('cohort').value,
                student_name: document.getElementById('student-name').value
            };

            try {
                const response = await fetch(`/api/room/${ROOM_ID}/metadata`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(metadata)
                });

                const result = await response.json();
                if (result.success) {
                    alert('Interview details saved!');
                }
            } catch (error) {
                console.error('Failed to save metadata:', error);
            }
        });
    }

    // Handle page unload
    window.addEventListener('beforeunload', (e) => {
        if (isRecording) {
            e.preventDefault();
            e.returnValue = 'Recording is in progress. Are you sure you want to leave?';
        }
    });
}
