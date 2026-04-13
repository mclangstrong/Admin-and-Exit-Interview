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
        // TURN servers (relay) — needed when both peers are behind symmetric NATs
        {
            urls: 'turn:a.relay.metered.ca:80',
            username: 'e8dd65b92f6aee43ccbacaad',
            credential: '5V3sFnChHHELD/ew'
        },
        {
            urls: 'turn:a.relay.metered.ca:80?transport=tcp',
            username: 'e8dd65b92f6aee43ccbacaad',
            credential: '5V3sFnChHHELD/ew'
        },
        {
            urls: 'turns:a.relay.metered.ca:443',
            username: 'e8dd65b92f6aee43ccbacaad',
            credential: '5V3sFnChHHELD/ew'
        },
    ]
};

// Debug mode - set to true for verbose console logging
const DEBUG = true;

// CSRF token for secure API requests
const CSRF_TOKEN = document.querySelector('meta[name="csrf-token"]')?.content || '';

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
let _speechBuffer = '';       // Debounce buffer for rapid speech
let _speechDebounceTimer = null;  // Debounce timer ID
const SPEECH_DEBOUNCE_MS = 3000;  // Collect speech for 3 seconds before sending
let recordingAudioContext = null; // AudioContext used during recording (for cleanup)
let remoteAudioPollTimer = null;  // Polls for remote audio if it arrives late

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

    // Verify analysis UI elements exist (interviewer only)
    console.log('🔍 Checking for analysis UI elements...');
    console.log('   liveSentiment:', liveSentiment ? '✅ Found' : '❌ Not found');
    console.log('   liveEmotion:', liveEmotion ? '✅ Found' : '❌ Not found');
    console.log('   engagementBar:', engagementBar ? '✅ Found' : '❌ Not found');
    console.log('   transcriptContent:', transcriptContent ? '✅ Found' : '❌ Not found');

    if (!liveSentiment && USER_ROLE === 'interviewer') {
        console.error('❌ CRITICAL: Analysis elements not found but user is interviewer!');
        console.log('   This means the HTML template may have an issue.');
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
    makeDraggable(localVideo); // Make local video draggable
    autoFillStudentData(); // Auto-fill interview details if student data available
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

    // Auto-detect server URL from current page (works with any IP/hostname)
    const serverUrl = window.location.origin;
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

        // Auto-fill form if user data is provided AND it's a student joining
        if (data.user_data && data.role === 'student') {
            if (DEBUG) console.log('📋 Received user data for auto-fill:', data.user_data);
            fillFormWithStudentData(data.user_data);
        }

        // Create peer connection immediately when second participant joins
        if (data.count === 2) {
            if (!peerConnection) {
                console.log('🔗 Creating peer connection...');
                await createPeerConnection();
            }

            // Warm up the NLP pipeline so first transcript is fast
            try {
                fetch('/api/warmup', { method: 'POST' });
                if (DEBUG) console.log('🔥 NLP warm-up request sent');
            } catch (e) {
                // Non-critical, ignore errors
            }

            // If I am NOT the one who just joined, I should create the offer
            if (data.sid !== socket.id) {
                if (DEBUG) console.log('📤 I am the first participant - creating offer for newcomer...');
                // Small delay to ensure both sides have added tracks
                await new Promise(resolve => setTimeout(resolve, 500));
                await createOffer();
            } else {
                if (DEBUG) console.log('📥 I am the second participant - waiting for offer...');
            }
        }

        // Late-join recovery: if student joins after interview already started,
        // start speech recognition immediately
        if (USER_ROLE === 'student' && data.interview_in_progress && speechRecognition && !isRecording) {
            console.log('🎤 Late-join: Interview already in progress, starting speech recognition...');
            isRecording = true;
            try {
                speechRecognition.start();
                console.log('✅ Late-join speech recognition started');
            } catch (error) {
                console.error('❌ Failed to start late-join speech recognition:', error);
            }
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

        // Create peer connection if it doesn't exist yet
        if (!peerConnection) {
            console.log('🔗 Creating peer connection for offer...');
            await createPeerConnection();
        }

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

        // Start speech recognition if we're the student
        if (USER_ROLE === 'student' && speechRecognition && !isRecording) {
            console.log('🎤 Starting speech recognition (triggered by interview start)...');
            isRecording = true; // Mark as recording so speech recognition knows to run
            try {
                speechRecognition.start();
                console.log('✅ Speech recognition started successfully');
            } catch (error) {
                console.error('❌ Failed to start speech recognition:', error);
            }
        }
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

        // Redirect to appropriate page with ended flag for modal
        if (USER_ROLE === 'student') {
            window.location.href = '/home?ended=1';
        } else {
            window.location.href = '/dashboard?ended=1';
        }
    });

    socket.on('transcript-line', (data) => {
        console.log('📝 Received transcript line:', data);

        // Only add if it's from another participant (avoid duplicates)
        if (data.speaker !== USER_ROLE) {
            const placeholder = transcriptContent?.querySelector('.transcript-placeholder');
            if (placeholder) {
                placeholder.remove();
            }

            const lineEl = document.createElement('div');
            lineEl.className = 'transcript-line';
            lineEl.innerHTML = `
                <div class="transcript-speaker">${data.speaker === 'interviewer' ? '👔 Interviewer' : '🎓 Student'}</div>
                <div class="transcript-text">${data.text}</div>
            `;
            transcriptContent?.appendChild(lineEl);
            if (transcriptContent) {
                transcriptContent.scrollTop = transcriptContent.scrollHeight;
            }
        }
    });

    socket.on('analysis-update', (data) => {
        if (DEBUG) console.log('📊 Received analysis update:', data);

        // Only update if it's from another participant (student → interviewer)
        if (data.speaker !== USER_ROLE) {
            if (liveSentiment && data.analysis.sentiment) {
                liveSentiment.textContent = data.analysis.sentiment;
                liveSentiment.className = 'analysis-value ' + (data.analysis.sentiment.toLowerCase());
            }

            if (liveEmotion && data.analysis.emotion) {
                liveEmotion.textContent = data.analysis.emotion;
            }

            if (engagementBar && data.analysis.engagement !== undefined) {
                engagementBar.style.width = `${data.analysis.engagement * 10}%`;
            }
        }
    });

    // Listen for async analysis results pushed from server
    socket.on('analysis-result', (data) => {
        if (DEBUG) console.log('📊 Received analysis result via WebSocket:', data);

        if (data.analysis) {
            const analysis = data.analysis;

            // Update local UI (if interviewer)
            if (liveSentiment) {
                const sentimentLabel = analysis.sentiment?.label || '--';
                liveSentiment.textContent = sentimentLabel;
                liveSentiment.className = 'analysis-value ' + (sentimentLabel.toLowerCase());
            }

            if (liveEmotion && analysis.emotions) {
                const topEmotion = Object.entries(analysis.emotions)
                    .sort((a, b) => b[1] - a[1])[0];
                liveEmotion.textContent = topEmotion ? topEmotion[0] : '--';
            }

            if (engagementBar && analysis.engagement) {
                engagementBar.style.width = `${analysis.engagement.score * 10}%`;
            }
        }
    });
}

// ============================================================================
// WebRTC Peer Connection
// ============================================================================

async function createPeerConnection() {
    console.log('🔗 Creating peer connection...');

    peerConnection = new RTCPeerConnection(ICE_SERVERS);

    // Add local tracks (only if we have local stream)
    if (localStream) {
        localStream.getTracks().forEach(track => {
            const sender = peerConnection.addTrack(track, localStream);
            console.log(`➕ Added ${track.kind} track:`, track.id);
        });
        console.log('✅ Added local tracks to peer connection');
    } else {
        console.warn('⚠️ No local stream - receive-only mode (can see remote video but not send)');
    }

    // Handle remote stream
    peerConnection.ontrack = (event) => {
        console.log('📹 Received remote track:', event.track.kind);
        if (!remoteStream) {
            remoteStream = new MediaStream();
            remoteVideo.srcObject = remoteStream;
            remotePlaceholder.style.display = 'none';
            document.getElementById('remote-name').textContent =
                USER_ROLE === 'interviewer' ? 'Student' : 'Interviewer';
        }
        remoteStream.addTrack(event.track);
        console.log('✅ Remote track added to stream');
    };

    // Handle ICE candidates
    peerConnection.onicecandidate = (event) => {
        if (event.candidate) {
            console.log('🧊 Sending ICE candidate');
            socket.emit('ice-candidate', {
                room_id: ROOM_ID,
                candidate: event.candidate
            });
        }
    };

    // Connection state changes
    peerConnection.onconnectionstatechange = () => {
        console.log('🔌 Connection state:', peerConnection.connectionState);
        switch (peerConnection.connectionState) {
            case 'connected':
                updateConnectionStatus('connected', 'Connected');
                console.log('✅ Peer connection established!');
                break;
            case 'disconnected':
            case 'failed':
                updateConnectionStatus('disconnected', 'Connection Lost');
                console.log('❌ Peer connection failed');
                break;
        }
    };

    // ICE connection state
    peerConnection.oniceconnectionstatechange = () => {
        console.log('🧊 ICE connection state:', peerConnection.iceConnectionState);
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
    let recordingSetupOk = false;

    // ── Audio recording setup (wrapped in try/catch so speech recognition
    //    starts even if recording hardware/codec fails) ──
    try {
        // Create a combined audio stream for recording (mixing both participants)
        const audioContext = new AudioContext();
        recordingAudioContext = audioContext; // Store for cleanup
        const destination = audioContext.createMediaStreamDestination();
        let remoteAudioConnected = false;

        // Add local audio (interviewer's mic)
        if (localStream && localStream.getAudioTracks().length > 0) {
            const localSource = audioContext.createMediaStreamSource(
                new MediaStream(localStream.getAudioTracks())
            );
            localSource.connect(destination);
            console.log('🎙️ Local audio connected to recording mix');
        } else {
            console.warn('⚠️ No local audio tracks available for recording');
        }

        // Helper: connect remote audio to the mixer
        function connectRemoteAudio() {
            if (remoteAudioConnected) return true;
            if (remoteStream && remoteStream.getAudioTracks().length > 0 && audioContext.state !== 'closed') {
                try {
                    const remoteSource = audioContext.createMediaStreamSource(
                        new MediaStream(remoteStream.getAudioTracks())
                    );
                    remoteSource.connect(destination);
                    remoteAudioConnected = true;
                    console.log('🎙️ Remote audio connected to recording mix');
                    return true;
                } catch (e) {
                    console.error('Failed to connect remote audio:', e);
                    return false;
                }
            }
            return false;
        }

        // Try connecting remote audio immediately
        connectRemoteAudio();

        // Poll for remote audio if it hasn't connected yet (covers timing gaps)
        if (!remoteAudioConnected) {
            console.log('⏳ Remote audio not ready yet — polling every 500ms...');
            let pollAttempts = 0;
            const MAX_POLL_ATTEMPTS = 60; // 30 seconds max
            remoteAudioPollTimer = setInterval(() => {
                pollAttempts++;
                if (connectRemoteAudio()) {
                    console.log(`✅ Remote audio connected after ${pollAttempts * 500}ms`);
                    clearInterval(remoteAudioPollTimer);
                    remoteAudioPollTimer = null;
                } else if (pollAttempts >= MAX_POLL_ATTEMPTS) {
                    console.warn('⚠️ Remote audio never became available during recording');
                    clearInterval(remoteAudioPollTimer);
                    remoteAudioPollTimer = null;
                }
            }, 500);
        }

        // Also listen for new remote tracks added after recording starts (renegotiation)
        if (peerConnection) {
            const origOnTrack = peerConnection.ontrack;
            peerConnection.ontrack = (event) => {
                // Call the original handler first (adds track to remoteVideo)
                if (origOnTrack) origOnTrack(event);

                // Connect new audio tracks to the recording mix
                if (event.track.kind === 'audio' && !remoteAudioConnected && audioContext.state !== 'closed') {
                    try {
                        const newRemoteSource = audioContext.createMediaStreamSource(
                            new MediaStream([event.track])
                        );
                        newRemoteSource.connect(destination);
                        remoteAudioConnected = true;
                        console.log('🎙️ Late remote audio track connected to recording mix via ontrack');
                        // Stop polling if still running
                        if (remoteAudioPollTimer) {
                            clearInterval(remoteAudioPollTimer);
                            remoteAudioPollTimer = null;
                        }
                    } catch (e) {
                        console.error('Failed to connect late remote audio:', e);
                    }
                }
            };
        }

        // Create audio-only combined stream (no video — smaller files, audio playback only)
        const combinedStream = new MediaStream([
            ...destination.stream.getAudioTracks()
        ]);

        // Check if we have any tracks to record
        if (combinedStream.getTracks().length === 0) {
            console.warn('⚠️ No audio tracks for recording, but speech recognition will still start');
        } else {
            mediaRecorder = new MediaRecorder(combinedStream, {
                mimeType: 'audio/webm;codecs=opus'
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
            recordingSetupOk = true;
            console.log('✅ MediaRecorder started');
        }
    } catch (recordingError) {
        console.error('⚠️ Recording setup failed (speech recognition will still start):', recordingError);
    }

    // ── These MUST run regardless of whether recording setup succeeded ──
    isRecording = true;
    recordingStartTime = Date.now();

    // Update UI
    updateRecordingUI(true);
    startTimer();

    // Start speech recognition (independent of recording)
    if (speechRecognition) {
        console.log('🎤 Starting speech recognition...');
        try {
            speechRecognition.start();
            console.log('✅ Speech recognition started successfully');
        } catch (error) {
            console.error('❌ Failed to start speech recognition:', error);
        }
    } else {
        console.warn('⚠️ Speech recognition not available');
    }

    // Notify server (so student can start transcription)
    socket.emit('interview-started', { room_id: ROOM_ID });

    if (!recordingSetupOk) {
        console.warn('⚠️ Audio recording is NOT active, but live transcription IS running');
    }
}

function stopRecording() {
    console.log('⏹️ Stopping recording...');

    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }

    isRecording = false;

    // Cleanup audio context and polling
    if (remoteAudioPollTimer) {
        clearInterval(remoteAudioPollTimer);
        remoteAudioPollTimer = null;
    }
    if (recordingAudioContext && recordingAudioContext.state !== 'closed') {
        recordingAudioContext.close();
        recordingAudioContext = null;
    }

    // Update UI
    updateRecordingUI(false);
    stopTimer();

    // Stop speech recognition
    if (speechRecognition) {
        speechRecognition.stop();
    }
}

async function saveRecording() {
    const blob = new Blob(recordedChunks, { type: 'audio/webm' });

    // Upload to server
    const formData = new FormData();
    formData.append('audio', blob, 'interview.webm');
    formData.append('room_id', ROOM_ID);
    formData.append('channel', 'combined');

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

    const blob = new Blob(recordedChunks, { type: 'audio/webm' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `interview_${ROOM_ID}_${Date.now()}.webm`;
    a.click();
}

// ============================================================================
// Speech Recognition (Both Roles)
// ============================================================================

function initializeSpeechRecognition() {
    console.log(`🎤 Initializing speech recognition for role: ${USER_ROLE}...`);

    console.log('🎤 Checking for Web Speech API support...');

    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        console.error('❌ Speech recognition not supported in this browser');
        console.log('   Please use Chrome or Edge for speech recognition');
        return;
    }

    try {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        console.log('✅ SpeechRecognition API found:', SpeechRecognition);

        speechRecognition = new SpeechRecognition();
        console.log('✅ SpeechRecognition instance created:', speechRecognition);

        speechRecognition.continuous = true;
        speechRecognition.interimResults = true;
        speechRecognition.lang = 'en-US'; // Will also pick up Tagalog reasonably

        speechRecognition.onresult = (event) => {
            const results = event.results;
            const lastResult = results[results.length - 1];

            // ALWAYS log speech detection (diagnostic)
            console.log('🎤 SPEECH DETECTED:', lastResult[0].transcript, '| Final:', lastResult.isFinal, '| Confidence:', lastResult[0].confidence);
            _lastSpeechTime = Date.now(); // Reset backoff — user is speaking

            if (lastResult.isFinal) {
                const text = lastResult[0].transcript.trim();

                // Only process non-empty text
                if (text.length > 0) {
                    console.log(`✅ FINAL TRANSCRIPT (${USER_ROLE}):`, text);

                    // Add to transcript UI with correct speaker label
                    addTranscriptLine(USER_ROLE, text);

                    // Only analyze student speech — interviewer speech is
                    // saved to transcript but NOT sent through NLP
                    if (USER_ROLE === 'student') {
                        // Debounce: buffer rapid speech into one combined request
                        _speechBuffer += ((_speechBuffer ? ' ' : '') + text);
                        clearTimeout(_speechDebounceTimer);
                        _speechDebounceTimer = setTimeout(() => {
                            const buffered = _speechBuffer.trim();
                            _speechBuffer = '';
                            if (buffered.length > 0) {
                                analyzeTranscriptLine(buffered);
                            }
                        }, SPEECH_DEBOUNCE_MS);
                    } else {
                        // Interviewer: just save transcript, no analysis
                        saveTranscriptOnly(text);
                    }
                }
            }
        };

        // ── Restart backoff state ──
        // Prevents rapid start/end loops that cause Chrome to throttle
        let _restartDelay = 300;           // Current delay (ms)
        const _RESTART_MIN_DELAY = 300;    // Minimum delay between restarts
        const _RESTART_MAX_DELAY = 5000;   // Maximum backoff delay (5s)
        let _lastSpeechTime = 0;           // Timestamp of last successful speech detection

        speechRecognition.onerror = (event) => {
            console.warn('⚠️ Speech recognition error:', event.error);
            if (event.error === 'no-speech') {
                // no-speech is normal — onend will fire and handle the restart
                return;
            }
            if (event.error === 'aborted') {
                // aborted usually means another tab or the system took over the mic
                console.warn('⚠️ Speech recognition aborted — will retry with delay');
            }
            // Restart on recoverable errors with delay
            setTimeout(() => {
                if (isRecording && speechRecognition) {
                    console.log('🔄 Restarting speech recognition after error...');
                    try {
                        speechRecognition.start();
                    } catch (e) {
                        // Already running — that's fine
                    }
                }
            }, 1500);
        };

        speechRecognition.onend = () => {
            console.log(`🛑 Speech recognition ended. isRecording: ${isRecording}, isMuted: ${isMuted}, restartDelay: ${_restartDelay}ms`);
            if (!isRecording || isMuted) return; // Don't restart if muted

            // Schedule restart with current backoff delay
            setTimeout(() => {
                if (isRecording && speechRecognition && !isMuted) {
                    try {
                        speechRecognition.start();
                    } catch (e) {
                        // Already running — ignore
                    }
                }
            }, _restartDelay);

            // Increase backoff if no speech was detected recently (prevents tight loop)
            const timeSinceLastSpeech = Date.now() - _lastSpeechTime;
            if (timeSinceLastSpeech > 10000) {
                // No speech in 10+ seconds — increase delay (Chrome may be throttling)
                _restartDelay = Math.min(_restartDelay * 1.5, _RESTART_MAX_DELAY);
            } else {
                // Speech was detected recently — keep delay low
                _restartDelay = _RESTART_MIN_DELAY;
            }
        };

        speechRecognition.onstart = () => {
            console.log('▶️ Speech recognition STARTED. Role:', USER_ROLE);
            // Reset backoff on successful start
            _restartDelay = _RESTART_MIN_DELAY;
        };

        console.log(`🎤 Speech recognition initialized successfully (${USER_ROLE})`);

        // === DIAGNOSTIC: Standalone speech test ===
        // Call testSpeechRecognition() from the browser console to test
        window.testSpeechRecognition = function() {
            console.log('🧪 === SPEECH RECOGNITION TEST ===');
            const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
            const test = new SR();
            test.continuous = false;
            test.interimResults = false;
            test.lang = 'en-US';
            test.onresult = (e) => {
                console.log('🧪 ✅ TEST HEARD:', e.results[0][0].transcript);
                console.log('🧪 Speech recognition IS WORKING!');
            };
            test.onerror = (e) => {
                console.log('🧪 ❌ TEST ERROR:', e.error);
            };
            test.onend = () => {
                console.log('🧪 TEST ENDED');
            };
            test.onstart = () => {
                console.log('🧪 TEST STARTED - speak now!');
            };
            test.start();
            console.log('🧪 Speak something within 10 seconds...');
        };

        // === DIAGNOSTIC: Mic level test ===
        window.testMicLevel = async function() {
            console.log('🧪 === MIC LEVEL TEST ===');
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                const ctx = new AudioContext();
                const source = ctx.createMediaStreamSource(stream);
                const analyser = ctx.createAnalyser();
                source.connect(analyser);
                const data = new Uint8Array(analyser.fftSize);
                let maxLevel = 0;
                const check = setInterval(() => {
                    analyser.getByteTimeDomainData(data);
                    let level = 0;
                    for (let i = 0; i < data.length; i++) {
                        level = Math.max(level, Math.abs(data[i] - 128));
                    }
                    maxLevel = Math.max(maxLevel, level);
                    const bar = '█'.repeat(Math.floor(level / 4));
                    console.log(`🎤 Mic level: ${level} ${bar}`);
                }, 500);
                setTimeout(() => {
                    clearInterval(check);
                    stream.getTracks().forEach(t => t.stop());
                    ctx.close();
                    console.log(`🧪 Max mic level: ${maxLevel} ${maxLevel > 5 ? '✅ Mic is working!' : '❌ No audio detected - check mic!'}`);
                }, 5000);
            } catch(e) {
                console.error('🧪 ❌ Mic test failed:', e);
            }
        };

        console.log('🧪 TIP: Run testSpeechRecognition() or testMicLevel() in console to diagnose');
    } catch (error) {
        console.error('❌ Failed to initialize speech recognition:', error);
        speechRecognition = null;
    }
}

function addTranscriptLine(speaker, text) {
    const line = { speaker, text, timestamp: new Date().toISOString() };
    transcript.push(line);

    // Update UI (only if transcript panel exists - interviewer has it, student doesn't)
    if (transcriptContent) {
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

    // Broadcast to other participants in the room (always send, even if no local UI)
    console.log('📤 Broadcasting transcript to room:', text);
    socket.emit('transcript-line', {
        room_id: ROOM_ID,
        speaker: speaker,
        text: text,
        timestamp: line.timestamp
    });
}

function analyzeTranscriptLine(text) {
    if (DEBUG) console.log('🔍 analyzeTranscriptLine called with text:', text);

    // Fire-and-forget POST — analysis results arrive via WebSocket 'analysis-result' event
    fetch('/api/transcript', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': CSRF_TOKEN
        },
        body: JSON.stringify({
            room_id: ROOM_ID,
            speaker: USER_ROLE,
            text: text
        })
    }).then(response => {
        if (!response.ok) {
            console.error(`❌ Transcript API error: ${response.status} ${response.statusText}`);
        }
    }).catch(error => {
        console.error('❌ Transcript save failed:', error);
    });
}

function saveTranscriptOnly(text) {
    // Save interviewer transcript to DB without triggering NLP analysis
    fetch('/api/transcript', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': CSRF_TOKEN
        },
        body: JSON.stringify({
            room_id: ROOM_ID,
            speaker: USER_ROLE,
            text: text,
            analyze: false
        })
    }).then(response => {
        if (!response.ok) {
            console.error(`❌ Interviewer transcript API error: ${response.status} ${response.statusText}`);
        }
    }).catch(error => {
        console.error('❌ Interviewer transcript save failed:', error);
    });
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

        // Also pause/resume speech recognition — the Web Speech API has its
        // own mic access independent of the MediaStream tracks. Without this,
        // the user's speech is still transcribed even when "muted."
        if (speechRecognition && isRecording) {
            if (isMuted) {
                try {
                    speechRecognition.abort(); // abort instead of stop to prevent onend restart
                    console.log('🔇 Speech recognition paused (mic muted)');
                } catch (e) { /* ignore */ }
            } else {
                try {
                    speechRecognition.start();
                    console.log('🔊 Speech recognition resumed (mic unmuted)');
                } catch (e) { /* already running */ }
            }
        }

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
        // Notify server immediately so the student gets redirected
        socket.emit('interview-ended', { room_id: ROOM_ID });
        socket.emit('leave-room', { room_id: ROOM_ID });

        // Stop recording and wait for upload before redirecting
        if (isRecording && mediaRecorder && mediaRecorder.state !== 'inactive') {
            console.log('⏹️ Stopping recording before ending interview...');

            // Override onstop to save recording, then cleanup and redirect
            mediaRecorder.onstop = async () => {
                console.log('⏹️ Recording stopped, uploading...');
                await saveRecording();
                console.log('✅ Recording saved, now redirecting...');
                cleanupAndRedirect();
            };

            // Stop speech recognition
            if (speechRecognition) {
                speechRecognition.stop();
            }
            isRecording = false;
            updateRecordingUI(false);
            stopTimer();

            mediaRecorder.stop();
        } else {
            // No recording active, just cleanup and redirect
            cleanupAndRedirect();
        }
    }
}

function cleanupAndRedirect() {
    // Close peer connection
    if (peerConnection) {
        peerConnection.close();
    }

    // Stop all tracks
    if (localStream) {
        localStream.getTracks().forEach(track => track.stop());
    }

    // Redirect based on role
    if (USER_ROLE === 'interviewer') {
        window.location.href = '/dashboard';
    } else {
        window.location.href = '/home';
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
    if (!connectionStatus) return;

    // Simply rebuild the entire content to avoid text duplication
    connectionStatus.innerHTML = `
        <span class="status-indicator ${status}"></span>
        ${text}
    `;
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
                student_name: document.getElementById('student-name-field').value
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

// ============================================================================
// Draggable Functionality
// ============================================================================

function makeDraggable(element) {
    if (!element) return;

    let isDragging = false;
    let currentX;
    let currentY;
    let initialX;
    let initialY;
    let xOffset = 0;
    let yOffset = 0;

    // Get the parent element (video-wrapper)
    const container = element.parentElement;
    if (!container) return;

    // Add cursor style
    container.style.cursor = 'move';

    container.addEventListener('mousedown', dragStart);
    document.addEventListener('mousemove', drag);
    document.addEventListener('mouseup', dragEnd);

    function dragStart(e) {
        // Only drag if clicking on the video container, not controls
        if (e.target === container || e.target === element) {
            initialX = e.clientX - xOffset;
            initialY = e.clientY - yOffset;
            isDragging = true;
            container.style.cursor = 'grabbing';
        }
    }

    function drag(e) {
        if (isDragging) {
            e.preventDefault();

            currentX = e.clientX - initialX;
            currentY = e.clientY - initialY;

            xOffset = currentX;
            yOffset = currentY;

            // Get viewport dimensions
            const viewportWidth = window.innerWidth;
            const viewportHeight = window.innerHeight;
            const rect = container.getBoundingClientRect();

            // Constrain to viewport boundaries
            let newLeft = currentX;
            let newTop = currentY;

            // Keep within horizontal bounds
            if (newLeft < 0) newLeft = 0;
            if (newLeft + rect.width > viewportWidth) {
                newLeft = viewportWidth - rect.width;
            }

            // Keep within vertical bounds
            if (newTop < 0) newTop = 0;
            if (newTop + rect.height > viewportHeight) {
                newTop = viewportHeight - rect.height;
            }

            setTranslate(newLeft, newTop, container);
        }
    }

    function dragEnd(e) {
        if (isDragging) {
            initialX = currentX;
            initialY = currentY;
            isDragging = false;
            container.style.cursor = 'move';
        }
    }

    function setTranslate(xPos, yPos, el) {
        el.style.left = xPos + 'px';
        el.style.top = yPos + 'px';
    }
}

// ============================================================================
// Auto-Fill Student Data
// ============================================================================

function autoFillStudentData() {
    // Only auto-fill if student data is available (hidden inputs exist)
    const studentName = document.getElementById('student-name');
    const studentCourse = document.getElementById('student-course');
    const studentCohort = document.getElementById('student-cohort');

    if (!studentName || !studentCourse || !studentCohort) {
        console.log('ℹ️  No student data available for auto-fill');
        return;
    }

    // Get the form fields
    const programField = document.getElementById('program');
    const cohortField = document.getElementById('cohort');
    const studentNameField = document.getElementById('student-name-field');

    if (programField && studentCourse.value) {
        programField.value = studentCourse.value;
        console.log('✅ Auto-filled Program:', studentCourse.value);
    }

    if (cohortField && studentCohort.value) {
        cohortField.value = studentCohort.value;
        console.log('✅ Auto-filled Cohort:', studentCohort.value);
    }

    if (studentNameField && studentName.value) {
        studentNameField.value = studentName.value;
        console.log('✅ Auto-filled Student Name:', studentName.value);
    }

    console.log('✅ Interview details auto-filled from student data');
}

function fillFormWithStudentData(data) {
    if (!data) return;

    // Get the form fields
    const programField = document.getElementById('program');
    const cohortField = document.getElementById('cohort');
    const studentNameField = document.getElementById('student-name-field');

    // Only fill if the field exists and is empty (or force update? Requirement implies auto-fill)
    // I'll overwrite to ensure data is correct as per 'automatically filled by getting the data of the student who joined'

    if (programField && data.course) {
        programField.value = data.course;
        console.log('✅ Socket Auto-filled Program:', data.course);
    }

    if (cohortField && data.cohort) {
        cohortField.value = data.cohort;
        console.log('✅ Socket Auto-filled Cohort:', data.cohort);
    }

    if (studentNameField && data.name) {
        studentNameField.value = data.name;
        console.log('✅ Socket Auto-filled Student Name:', data.name);
    }

    // Auto-save to backend immediately
    autoSaveStudentData(data);
}

async function autoSaveStudentData(data) {
    if (!data) return;

    const metadata = {
        interview_type: document.getElementById('interview-type')?.value || 'admission',
        program: data.course || '',
        cohort: data.cohort || '',
        student_name: data.name || ''
    };

    console.log('💾 Auto-saving student data to backend:', metadata);

    try {
        const response = await fetch(`/api/room/${ROOM_ID}/metadata`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(metadata)
        });

        const result = await response.json();
        if (result.success) {
            console.log('✅ Student data auto-saved successfully');
        } else {
            console.error('❌ Failed to auto-save student data:', result);
        }
    } catch (error) {
        console.error('❌ Error auto-saving student data:', error);
    }
}
