"""
NLP Utilities for Interview Analysis
=====================================
This module provides comprehensive NLP analysis for student interviews:
- Key Phrase Extraction (using KeyBERT)
- Sentiment Analysis (using trained mBERT model)
- Emotion Analysis (using pre-trained transformer)
- Engagement Measurement (custom heuristics)
"""

import os
import warnings
import numpy as np
import torch
from typing import Dict, List, Any, Optional

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths to models
SENTIMENT_MODEL_PATH = "./taglish_mbert_model_final"
SENTIMENT_SINGLE_FILE = "./taglish_sentiment_model_full.pth"

# Emotion model from Hugging Face
EMOTION_MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"

# Label mappings
SENTIMENT_LABELS = {0: "Positive", 1: "Neutral", 2: "Negative"}
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Engagement thresholds
ENGAGEMENT_THRESHOLDS = {
    "word_count_low": 10,
    "word_count_high": 50,
    "keyword_threshold": 3,
}

# Filler words to remove (English and Tagalog)
FILLER_WORDS = [
    # English fillers
    "um", "uh", "er", "ah", "like", "you know", "i mean", "basically",
    "actually", "literally", "honestly", "obviously", "right", "okay", "so",
    # Tagalog fillers
    "ano", "eh", "kasi", "parang", "ganun", "ganon", "alam mo", "di ba",
    "tapos", "sige", "oo", "naman", "lang", "diba", "yung", "yun"
]

# Common ASR errors and corrections
ASR_CORRECTIONS = {
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "kinda": "kind of",
    "sorta": "sort of",
}


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def clean_transcript(text: str, remove_fillers: bool = True, 
                    fix_asr_errors: bool = True) -> str:
    """
    Clean raw transcript text from ASR output.
    
    Args:
        text: Raw transcript text
        remove_fillers: Whether to remove filler words
        fix_asr_errors: Whether to fix common ASR transcription errors
        
    Returns:
        Cleaned transcript text
    """
    import re
    
    if not text:
        return ""
    
    # Lowercase for processing
    cleaned = text.strip()
    
    # Fix ASR errors
    if fix_asr_errors:
        for error, correction in ASR_CORRECTIONS.items():
            cleaned = re.sub(rf'\b{error}\b', correction, cleaned, flags=re.IGNORECASE)
    
    # Remove filler words
    if remove_fillers:
        for filler in FILLER_WORDS:
            # Remove as standalone word
            cleaned = re.sub(rf'\b{filler}\b[,.]?\s*', '', cleaned, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\s+([.,!?])', r'\1', cleaned)
    
    # Fix multiple punctuation
    cleaned = re.sub(r'[.]{2,}', '.', cleaned)
    cleaned = re.sub(r'[,]{2,}', ',', cleaned)
    
    return cleaned.strip()


def tokenize_text(text: str, for_taglish: bool = True) -> List[str]:
    """
    Tokenize text with special handling for Taglish (Tagalog-English mix).
    
    Args:
        text: Input text to tokenize
        for_taglish: Apply Taglish-specific handling
        
    Returns:
        List of tokens
    """
    import re
    
    if not text:
        return []
    
    # Basic tokenization
    tokens = text.split()
    
    # Handle contractions and special Taglish patterns
    if for_taglish:
        expanded_tokens = []
        for token in tokens:
            # Handle Tagalog affixes attached to English words (code-switching)
            # e.g., "mag-programming" -> ["mag", "programming"]
            if '-' in token and any(c.isalpha() for c in token):
                parts = token.split('-')
                expanded_tokens.extend(parts)
            else:
                expanded_tokens.append(token)
        tokens = expanded_tokens
    
    # Remove punctuation from tokens but keep the words
    cleaned_tokens = []
    for token in tokens:
        # Remove leading/trailing punctuation
        cleaned = re.sub(r'^[^\w]+|[^\w]+$', '', token)
        if cleaned:
            cleaned_tokens.append(cleaned)
    
    return cleaned_tokens


def normalize_text(text: str, lowercase: bool = True, 
                  remove_numbers: bool = False) -> str:
    """
    Normalize text for NLP processing.
    
    Args:
        text: Input text
        lowercase: Convert to lowercase
        remove_numbers: Remove numeric characters
        
    Returns:
        Normalized text
    """
    import re
    
    if not text:
        return ""
    
    normalized = text.strip()
    
    # Lowercase
    if lowercase:
        normalized = normalized.lower()
    
    # Remove numbers
    if remove_numbers:
        normalized = re.sub(r'\d+', '', normalized)
    
    # Normalize unicode characters
    import unicodedata
    normalized = unicodedata.normalize('NFKC', normalized)
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized.strip()


def preprocess_transcript(text: str) -> Dict[str, Any]:
    """
    Full preprocessing pipeline for interview transcripts.
    
    Args:
        text: Raw transcript text
        
    Returns:
        Dict with cleaned text, tokens, and metadata
    """
    original_length = len(text.split())
    
    # Clean
    cleaned = clean_transcript(text)
    
    # Normalize
    normalized = normalize_text(cleaned)
    
    # Tokenize
    tokens = tokenize_text(normalized)
    
    return {
        "original": text,
        "cleaned": cleaned,
        "normalized": normalized,
        "tokens": tokens,
        "original_word_count": original_length,
        "processed_word_count": len(tokens),
        "words_removed": original_length - len(tokens)
    }



# ============================================================================
# INTERVIEW ANALYZER CLASS
# ============================================================================

class InterviewAnalyzer:
    """
    Comprehensive NLP analyzer for student interview responses.
    
    Features:
    - Key Phrase Extraction
    - Sentiment Analysis (Positive/Neutral/Negative)
    - Emotion Analysis (joy, sadness, anger, fear, love, surprise)
    - Engagement Scoring (0-10 scale)
    """
    
    def __init__(self, load_sentiment: bool = True, load_emotion: bool = True, 
                 load_keyphrase: bool = True):
        """
        Initialize the analyzer with optional component loading.
        
        Args:
            load_sentiment: Whether to load the sentiment model
            load_emotion: Whether to load the emotion model
            load_keyphrase: Whether to load KeyBERT for key phrases
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚙️  Device: {self.device}")
        
        self.sentiment_model = None
        self.sentiment_tokenizer = None
        self.emotion_pipeline = None
        self.keyphrase_model = None
        
        if load_sentiment:
            self._load_sentiment_model()
        
        if load_emotion:
            self._load_emotion_model()
        
        if load_keyphrase:
            self._load_keyphrase_model()
        
        print("✅ InterviewAnalyzer ready!")
    
    # ------------------------------------------------------------------------
    # MODEL LOADING
    # ------------------------------------------------------------------------
    
    def _load_sentiment_model(self):
        """Load the trained Taglish sentiment model."""
        print("📥 Loading Sentiment Model...")
        try:
            from transformers import BertTokenizer, BertForSequenceClassification
            
            if os.path.exists(SENTIMENT_MODEL_PATH):
                self.sentiment_tokenizer = BertTokenizer.from_pretrained(SENTIMENT_MODEL_PATH)
                self.sentiment_model = BertForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH)
                self.sentiment_model.to(self.device)
                self.sentiment_model.eval()
                print("   ✓ Sentiment model loaded from folder")
            elif os.path.exists(SENTIMENT_SINGLE_FILE):
                # Load from single .pth file
                self.sentiment_model = torch.load(SENTIMENT_SINGLE_FILE, map_location=self.device)
                self.sentiment_model.eval()
                # Still need tokenizer from HuggingFace
                self.sentiment_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
                print("   ✓ Sentiment model loaded from .pth file")
            else:
                print("   ⚠️ Sentiment model not found. Train the model first.")
        except Exception as e:
            print(f"   ❌ Error loading sentiment model: {e}")
    
    def _load_emotion_model(self):
        """Load pre-trained emotion detection model."""
        print("📥 Loading Emotion Model...")
        try:
            from transformers import pipeline
            self.emotion_pipeline = pipeline(
                "text-classification", 
                model=EMOTION_MODEL_NAME, 
                top_k=None,
                device=0 if self.device == "cuda" else -1
            )
            print("   ✓ Emotion model loaded")
        except Exception as e:
            print(f"   ❌ Error loading emotion model: {e}")
    
    def _load_keyphrase_model(self):
        """Load KeyBERT for key phrase extraction."""
        print("📥 Loading KeyBERT...")
        try:
            from keybert import KeyBERT
            self.keyphrase_model = KeyBERT("paraphrase-multilingual-MiniLM-L12-v2")
            print("   ✓ KeyBERT loaded")
        except Exception as e:
            print(f"   ❌ Error loading KeyBERT: {e}")
    
    # ------------------------------------------------------------------------
    # ANALYSIS METHODS
    # ------------------------------------------------------------------------
    
    def extract_keyphrases(self, text: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Extract key phrases from text using KeyBERT.
        
        Args:
            text: Input text to analyze
            top_n: Number of key phrases to return
            
        Returns:
            List of dicts with 'phrase' and 'score'
        """
        if self.keyphrase_model is None:
            return []
        
        try:
            keywords = self.keyphrase_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_n=top_n,
                use_mmr=True,
                diversity=0.5
            )
            return [{"phrase": kw[0], "score": round(kw[1], 4)} for kw in keywords]
        except Exception as e:
            print(f"⚠️ Keyphrase extraction error: {e}")
            return []
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using the trained Taglish model.
        
        Returns:
            Dict with 'label', 'confidence', and 'probabilities'
        """
        if self.sentiment_model is None or self.sentiment_tokenizer is None:
            return {"label": "Unknown", "confidence": 0.0, "probabilities": {}}
        
        try:
            inputs = self.sentiment_tokenizer(
                text, 
                padding='max_length', 
                truncation=True,
                max_length=128, 
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
            
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            
            return {
                "label": SENTIMENT_LABELS[pred_idx],
                "confidence": round(float(probs[pred_idx]), 4),
                "probabilities": {
                    "Positive": round(float(probs[0]), 4),
                    "Neutral": round(float(probs[1]), 4),
                    "Negative": round(float(probs[2]), 4)
                }
            }
        except Exception as e:
            print(f"⚠️ Sentiment analysis error: {e}")
            return {"label": "Error", "confidence": 0.0, "probabilities": {}}
    
    def analyze_emotion(self, text: str) -> Dict[str, float]:
        """
        Analyze emotions in text using pre-trained model.
        
        Returns:
            Dict mapping emotion names to scores (0-1)
        """
        if self.emotion_pipeline is None:
            return {}
        
        try:
            results = self.emotion_pipeline(text)[0]
            return {item['label']: round(item['score'], 4) for item in results}
        except Exception as e:
            print(f"⚠️ Emotion analysis error: {e}")
            return {}
    
    def calculate_engagement(self, text: str, sentiment_result: Dict = None, 
                              keyphrases: List[Dict] = None) -> Dict[str, Any]:
        """
        Calculate engagement score based on multiple factors.
        
        Factors:
        - Response length (word count)
        - Sentiment intensity (how strong the sentiment is)
        - Specificity (number of meaningful key phrases)
        - Vocabulary diversity
        
        Returns:
            Dict with 'score' (0-10), 'level', and 'factors'
        """
        words = text.split()
        word_count = len(words)
        unique_words = len(set(words))
        
        # Factor 1: Length score (0-3 points)
        if word_count < ENGAGEMENT_THRESHOLDS["word_count_low"]:
            length_score = 1.0
        elif word_count > ENGAGEMENT_THRESHOLDS["word_count_high"]:
            length_score = 3.0
        else:
            # Linear interpolation
            length_score = 1.0 + 2.0 * (word_count - 10) / 40
        
        # Factor 2: Sentiment intensity (0-2.5 points)
        intensity_score = 0.0
        if sentiment_result and "confidence" in sentiment_result:
            # Higher confidence = stronger opinion = more engaged
            intensity_score = sentiment_result["confidence"] * 2.5
        
        # Factor 3: Specificity / Key phrases (0-2.5 points)
        specificity_score = 0.0
        if keyphrases:
            kp_count = len(keyphrases)
            avg_score = np.mean([kp["score"] for kp in keyphrases]) if keyphrases else 0
            specificity_score = min(2.5, (kp_count / 3) * avg_score * 2.5)
        
        # Factor 4: Vocabulary diversity (0-2 points)
        diversity_ratio = unique_words / max(word_count, 1)
        diversity_score = diversity_ratio * 2.0
        
        # Total score (0-10)
        total_score = length_score + intensity_score + specificity_score + diversity_score
        total_score = round(min(10.0, max(0.0, total_score)), 2)
        
        # Engagement level
        if total_score >= 7:
            level = "High"
        elif total_score >= 4:
            level = "Medium"
        else:
            level = "Low"
        
        return {
            "score": total_score,
            "level": level,
            "factors": {
                "length": round(length_score, 2),
                "sentiment_intensity": round(intensity_score, 2),
                "specificity": round(specificity_score, 2),
                "vocabulary_diversity": round(diversity_score, 2)
            }
        }
    
    # ------------------------------------------------------------------------
    # MAIN ANALYSIS METHOD
    # ------------------------------------------------------------------------
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on text.
        
        Args:
            text: Input text (interview response)
            
        Returns:
            Dict containing all analysis results
        """
        print(f"\n🔍 Analyzing: \"{text[:50]}...\"" if len(text) > 50 else f"\n🔍 Analyzing: \"{text}\"")
        
        # Run all analyses
        keyphrases = self.extract_keyphrases(text)
        sentiment = self.analyze_sentiment(text)
        emotions = self.analyze_emotion(text)
        engagement = self.calculate_engagement(text, sentiment, keyphrases)
        
        result = {
            "text": text,
            "word_count": len(text.split()),
            "keyphrases": keyphrases,
            "sentiment": sentiment,
            "emotions": emotions,
            "engagement": engagement
        }
        
        return result
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple texts."""
        return [self.analyze(t) for t in texts]


# ============================================================================
# STANDALONE FUNCTIONS
# ============================================================================

def quick_analyze(text: str) -> Dict[str, Any]:
    """
    Quick analysis without keeping models in memory.
    Use InterviewAnalyzer for batch processing.
    """
    analyzer = InterviewAnalyzer()
    return analyzer.analyze(text)


# ============================================================================
# DEMO / TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("   NLP UTILITIES - INTERVIEW ANALYZER DEMO")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = InterviewAnalyzer()
    
    # Test samples
    test_texts = [
        "Sobrang saya ko po dahil natuto ako ng maraming bagong skills sa programming. Feeling ko ready na ako for my next steps!",
        "Okay lang naman, may mga natutunan pero hindi ko masyadong feel yung ibang topics.",
        "Nahihirapan po ako, hindi ko maintindihan yung mga lessons at parang hindi para sa akin ang course na ito.",
    ]
    
    print("\n" + "=" * 70)
    print("   ANALYSIS RESULTS")
    print("=" * 70)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'─' * 70}")
        print(f"📝 SAMPLE {i}")
        print(f"{'─' * 70}")
        
        result = analyzer.analyze(text)
        
        print(f"\n💬 Text: \"{text[:60]}...\"")
        print(f"📊 Word Count: {result['word_count']}")
        
        print(f"\n🎯 SENTIMENT: {result['sentiment']['label']} ({result['sentiment']['confidence']:.1%})")
        
        print(f"\n😊 EMOTIONS:")
        if result['emotions']:
            sorted_emotions = sorted(result['emotions'].items(), key=lambda x: x[1], reverse=True)
            for emotion, score in sorted_emotions[:3]:
                bar = "█" * int(score * 20)
                print(f"   {emotion:<10}: {bar} {score:.1%}")
        
        print(f"\n🔑 KEY PHRASES:")
        for kp in result['keyphrases'][:3]:
            print(f"   • {kp['phrase']} (relevance: {kp['score']:.2f})")
        
        print(f"\n📈 ENGAGEMENT: {result['engagement']['score']}/10 ({result['engagement']['level']})")
        print(f"   Factors: {result['engagement']['factors']}")
    
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE")
    print("=" * 70)
