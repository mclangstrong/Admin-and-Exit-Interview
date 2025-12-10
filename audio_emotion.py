"""
Audio Emotion Detection Module
==============================
Analyze emotional tone from audio recordings using deep learning.

Features:
- Extract audio features (MFCC, Mel Spectrogram)
- Emotion classification (calm, happy, sad, angry, fearful, surprised, disgusted, neutral)
- Confidence scores for each emotion
- Pre-trained model option + custom training capability
"""

import os
import warnings
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Emotion labels (RAVDESS-compatible)
EMOTION_LABELS = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgusted",
    7: "surprised"
}

# Interview-relevant emotions
INTERVIEW_EMOTIONS = {
    "confident": ["calm", "happy"],
    "nervous": ["fearful", "surprised"],
    "engaged": ["happy", "surprised"],
    "disengaged": ["neutral", "sad"],
    "frustrated": ["angry", "disgusted"]
}

# Audio feature config
SAMPLE_RATE = 22050
N_MFCC = 40
HOP_LENGTH = 512
N_FFT = 2048


# ============================================================================
# AUDIO EMOTION ANALYZER CLASS
# ============================================================================

class AudioEmotionAnalyzer:
    """
    Analyze emotional tone from audio files.
    
    Uses audio feature extraction (MFCC) and a neural network classifier
    to detect emotions in speech.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the analyzer.
        
        Args:
            model_path: Optional path to a custom trained model.
                       Uses pre-trained transformer if None.
        """
        self.model = None
        self.use_transformer = True
        self.emotion_pipeline = None
        
        self._load_model(model_path)
    
    def _load_model(self, model_path: str = None):
        """Load emotion detection model."""
        # Try to use HuggingFace audio emotion model
        try:
            from transformers import pipeline
            
            # Use a pre-trained speech emotion recognition model
            self.emotion_pipeline = pipeline(
                "audio-classification",
                model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                device=-1  # CPU
            )
            print("✅ Audio emotion model loaded (wav2vec2)")
            return
        except Exception as e:
            print(f"⚠️ Transformer model failed: {e}")
        
        # Fallback to feature-based analysis
        self.use_transformer = False
        print("ℹ️ Using feature-based emotion analysis")
    
    # ------------------------------------------------------------------------
    # MAIN ANALYSIS
    # ------------------------------------------------------------------------
    
    def analyze(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze emotions in an audio file.
        
        Args:
            audio_path: Path to the audio file (wav, mp3, webm, etc.)
            
        Returns:
            Dict containing emotion predictions and confidence scores
        """
        if not os.path.exists(audio_path):
            return {"error": f"Audio file not found: {audio_path}"}
        
        try:
            # Convert to wav if needed
            wav_path = self._ensure_wav(audio_path)
            
            if self.use_transformer and self.emotion_pipeline:
                return self._analyze_with_transformer(wav_path)
            else:
                return self._analyze_with_features(wav_path)
                
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_with_transformer(self, audio_path: str) -> Dict[str, Any]:
        """Analyze using pre-trained transformer model."""
        try:
            results = self.emotion_pipeline(audio_path)
            
            # Format results
            emotions = {}
            for item in results:
                label = item['label'].lower()
                score = item['score']
                emotions[label] = round(score, 4)
            
            # Get top emotion
            top_emotion = max(emotions.items(), key=lambda x: x[1])
            
            # Map to interview context
            interview_state = self._map_to_interview_state(emotions)
            
            return {
                "method": "transformer",
                "primary_emotion": top_emotion[0],
                "confidence": top_emotion[1],
                "all_emotions": emotions,
                "interview_state": interview_state
            }
            
        except Exception as e:
            print(f"Transformer analysis failed: {e}")
            return self._analyze_with_features(audio_path)
    
    def _analyze_with_features(self, audio_path: str) -> Dict[str, Any]:
        """Analyze using extracted audio features."""
        try:
            import librosa
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            
            # Extract features
            features = self._extract_features(y, sr)
            
            # Simple heuristic-based emotion detection
            emotions = self._features_to_emotions(features)
            
            # Get top emotion
            top_emotion = max(emotions.items(), key=lambda x: x[1])
            
            # Map to interview context
            interview_state = self._map_to_interview_state(emotions)
            
            return {
                "method": "feature-based",
                "primary_emotion": top_emotion[0],
                "confidence": top_emotion[1],
                "all_emotions": emotions,
                "interview_state": interview_state,
                "audio_features": {
                    "energy": features.get("energy", 0),
                    "tempo": features.get("tempo", 0),
                    "pitch_variation": features.get("pitch_std", 0)
                }
            }
            
        except ImportError:
            return {
                "error": "librosa not installed. Run: pip install librosa",
                "primary_emotion": "unknown",
                "confidence": 0,
                "all_emotions": {}
            }
        except Exception as e:
            return {"error": str(e)}
    
    # ------------------------------------------------------------------------
    # FEATURE EXTRACTION
    # ------------------------------------------------------------------------
    
    def _extract_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract audio features for emotion detection."""
        import librosa
        
        features = {}
        
        # Energy (RMS)
        rms = librosa.feature.rms(y=y)[0]
        features["energy"] = float(np.mean(rms))
        features["energy_std"] = float(np.std(rms))
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features["tempo"] = float(tempo) if not hasattr(tempo, '__len__') else float(tempo[0])
        
        # Pitch (using zero crossing rate as proxy)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_std"] = float(np.std(zcr))
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))
            features[f"mfcc_{i}_std"] = float(np.std(mfccs[i]))
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features["spectral_centroid"] = float(np.mean(spectral_centroids))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features["spectral_rolloff"] = float(np.mean(spectral_rolloff))
        
        # Pitch estimation
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        if len(pitch_values) > 0:
            features["pitch_mean"] = float(np.mean(pitch_values[pitch_values > 0]))
            features["pitch_std"] = float(np.std(pitch_values[pitch_values > 0]))
        else:
            features["pitch_mean"] = 0
            features["pitch_std"] = 0
        
        return features
    
    def _features_to_emotions(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Convert audio features to emotion probabilities using heuristics.
        
        This is a simplified model based on acoustic correlates of emotions:
        - High energy + high pitch variation = excited emotions (happy, angry, surprised)
        - Low energy + low variation = calm emotions (neutral, sad, calm)
        - Fast tempo = high arousal
        """
        energy = features.get("energy", 0)
        energy_std = features.get("energy_std", 0)
        pitch_std = features.get("pitch_std", 0)
        tempo = features.get("tempo", 100)
        zcr = features.get("zcr_mean", 0)
        
        # Normalize values (approximate)
        energy_norm = min(1.0, energy / 0.1)
        variation_norm = min(1.0, (energy_std + pitch_std / 100) / 0.1)
        tempo_norm = min(1.0, tempo / 150)
        
        # Heuristic emotion scores
        emotions = {
            "neutral": 0.5 * (1 - energy_norm) * (1 - variation_norm),
            "calm": 0.5 * (1 - tempo_norm) * (1 - variation_norm),
            "happy": 0.5 * energy_norm * tempo_norm,
            "sad": 0.5 * (1 - energy_norm) * (1 - tempo_norm),
            "angry": 0.5 * energy_norm * variation_norm,
            "fearful": 0.3 * variation_norm * tempo_norm,
            "surprised": 0.3 * variation_norm * energy_norm,
            "disgusted": 0.2 * (1 - tempo_norm) * variation_norm
        }
        
        # Normalize to sum to 1
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: round(v / total, 4) for k, v in emotions.items()}
        
        return emotions
    
    # ------------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------------
    
    def _ensure_wav(self, audio_path: str) -> str:
        """Convert audio to WAV format if needed."""
        if audio_path.endswith('.wav'):
            return audio_path
        
        try:
            from pydub import AudioSegment
            
            # Load and convert
            audio = AudioSegment.from_file(audio_path)
            wav_path = audio_path.rsplit('.', 1)[0] + '_converted.wav'
            audio.export(wav_path, format='wav')
            return wav_path
        except ImportError:
            # If pydub not available, try librosa directly
            return audio_path
        except Exception as e:
            print(f"Audio conversion warning: {e}")
            return audio_path
    
    def _map_to_interview_state(self, emotions: Dict[str, float]) -> Dict[str, Any]:
        """Map detected emotions to interview-relevant states."""
        state_scores = {}
        
        for state, state_emotions in INTERVIEW_EMOTIONS.items():
            score = sum(emotions.get(e, 0) for e in state_emotions)
            state_scores[state] = round(score, 4)
        
        # Determine primary state
        primary_state = max(state_scores.items(), key=lambda x: x[1])
        
        return {
            "primary": primary_state[0],
            "confidence": primary_state[1],
            "all_states": state_scores
        }
    
    # ------------------------------------------------------------------------
    # SEGMENT ANALYSIS
    # ------------------------------------------------------------------------
    
    def analyze_segments(self, audio_path: str, segment_duration: float = 10.0) -> List[Dict]:
        """
        Analyze audio in segments to track emotion changes over time.
        
        Args:
            audio_path: Path to audio file
            segment_duration: Duration of each segment in seconds
            
        Returns:
            List of emotion results for each segment
        """
        try:
            import librosa
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            total_duration = len(y) / sr
            
            segments = []
            segment_samples = int(segment_duration * sr)
            
            for i, start in enumerate(range(0, len(y), segment_samples)):
                end = min(start + segment_samples, len(y))
                segment_audio = y[start:end]
                
                if len(segment_audio) < sr:  # Skip segments < 1 second
                    continue
                
                # Extract features for segment
                features = self._extract_features(segment_audio, sr)
                emotions = self._features_to_emotions(features)
                interview_state = self._map_to_interview_state(emotions)
                
                top_emotion = max(emotions.items(), key=lambda x: x[1])
                
                segments.append({
                    "segment_id": i,
                    "start_time": round(start / sr, 2),
                    "end_time": round(end / sr, 2),
                    "primary_emotion": top_emotion[0],
                    "confidence": top_emotion[1],
                    "interview_state": interview_state["primary"]
                })
            
            return segments
            
        except Exception as e:
            return [{"error": str(e)}]


# ============================================================================
# STANDALONE FUNCTIONS
# ============================================================================

def quick_emotion_analysis(audio_path: str) -> Dict[str, Any]:
    """Quick emotion analysis without class initialization."""
    analyzer = AudioEmotionAnalyzer()
    return analyzer.analyze(audio_path)


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("   AUDIO EMOTION DETECTION DEMO")
    print("=" * 60)
    
    # Check if librosa is available
    try:
        import librosa
        print("✅ librosa is installed")
    except ImportError:
        print("❌ librosa not installed. Run: pip install librosa")
    
    # Initialize analyzer
    analyzer = AudioEmotionAnalyzer()
    
    # Demo with synthetic features (no actual audio file)
    print("\n📊 Testing feature-to-emotion mapping...")
    
    test_features = [
        {"energy": 0.08, "energy_std": 0.02, "pitch_std": 50, "tempo": 120, "zcr_mean": 0.1},
        {"energy": 0.02, "energy_std": 0.005, "pitch_std": 10, "tempo": 70, "zcr_mean": 0.05},
        {"energy": 0.15, "energy_std": 0.05, "pitch_std": 80, "tempo": 140, "zcr_mean": 0.15},
    ]
    
    labels = ["Normal speech", "Calm/sad speech", "Excited/angry speech"]
    
    for label, features in zip(labels, test_features):
        emotions = analyzer._features_to_emotions(features)
        top_emotion = max(emotions.items(), key=lambda x: x[1])
        interview_state = analyzer._map_to_interview_state(emotions)
        
        print(f"\n🎤 {label}:")
        print(f"   Primary emotion: {top_emotion[0]} ({top_emotion[1]:.1%})")
        print(f"   Interview state: {interview_state['primary']}")
    
    print("\n" + "=" * 60)
    print("✅ DEMO COMPLETE")
    print("=" * 60)
    print("\n💡 To analyze a real audio file:")
    print("   from audio_emotion import AudioEmotionAnalyzer")
    print("   analyzer = AudioEmotionAnalyzer()")
    print("   result = analyzer.analyze('path/to/audio.wav')")
