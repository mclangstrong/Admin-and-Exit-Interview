"""
Topic Classifier Module
========================
Uses RoBERTa-Tagalog model to classify interview transcripts into topic categories:
- Academics
- Career
- Faculty
- Infrastructure
- Mental health
- Social
- Technology
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import os
from typing import List, Dict, Tuple, Optional

# Model path
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

# Topic labels (sorted alphabetically as per model training)
TOPIC_LABELS = ['Academics', 'Career', 'Faculty', 'Infrastructure', 'Mental health', 'Social', 'Technology']

# Classification thresholds per topic
THRESHOLDS = {
    'Academics': 0.60,
    'Career': 0.50,
    'Faculty': 0.50,
    'Infrastructure': 0.10,
    'Mental health': 0.50,
    'Social': 0.50,
    'Technology': 0.30
}

# Global model cache
_topic_tokenizer = None
_topic_model = None
_model_loaded = False


def load_topic_model() -> Tuple[Optional[object], Optional[object]]:
    """Load the topic classification model (lazy loading with caching)."""
    global _topic_tokenizer, _topic_model, _model_loaded
    
    if _model_loaded:
        return _topic_tokenizer, _topic_model
    
    print("🏷️ Loading Topic Classification Model...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Topic model folder not found: {MODEL_PATH}")
        return None, None
    
    try:
        # Load tokenizer from HuggingFace
        _topic_tokenizer = AutoTokenizer.from_pretrained("jcblaise/roberta-tagalog-base")
        
        # Load model from local folder
        _topic_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        _topic_model.eval()  # Set to evaluation mode
        
        _model_loaded = True
        print("✅ Topic Classification Model loaded successfully!")
        return _topic_tokenizer, _topic_model
        
    except Exception as e:
        print(f"❌ Failed to load topic model: {e}")
        return None, None


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def predict_sentence(sentence: str, tokenizer, model) -> List[Tuple[str, float]]:
    """Predict topics for a single sentence."""
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    probs = torch.sigmoid(logits).squeeze().tolist()
    if isinstance(probs, float):
        probs = [probs]
    
    results = []
    for i, label in enumerate(TOPIC_LABELS):
        score = probs[i]
        threshold = THRESHOLDS.get(label, 0.5)
        if score >= threshold:
            results.append((label, score))
    
    return results


def classify_transcript(transcript_texts: List[str]) -> Dict:
    """
    Classify a list of transcript texts and return topic distribution.
    
    Args:
        transcript_texts: List of text strings from the transcript
        
    Returns:
        Dictionary with topic percentages and stats
    """
    tokenizer, model = load_topic_model()
    
    if tokenizer is None or model is None:
        return {
            'success': False,
            'error': 'Topic classification model not loaded',
            'topics': {}
        }
    
    # Count classifications per topic
    topic_counts = {label: 0 for label in TOPIC_LABELS}
    total_sentences = 0
    classified_sentences = 0
    
    for text in transcript_texts:
        # Split each transcript entry into sentences
        sentences = split_into_sentences(text)
        
        for sentence in sentences:
            if len(sentence.strip()) < 5:  # Skip very short sentences
                continue
                
            total_sentences += 1
            predictions = predict_sentence(sentence, tokenizer, model)
            
            if predictions:
                classified_sentences += 1
                for label, score in predictions:
                    topic_counts[label] += 1
    
    # Calculate percentages
    total_classifications = sum(topic_counts.values())
    
    if total_classifications > 0:
        topic_percentages = {
            label: round((count / total_classifications) * 100, 1)
            for label, count in topic_counts.items()
        }
    else:
        topic_percentages = {label: 0.0 for label in TOPIC_LABELS}
    
    return {
        'success': True,
        'topics': topic_percentages,
        'topic_counts': topic_counts,
        'total_sentences': total_sentences,
        'classified_sentences': classified_sentences
    }


# Pre-load model at import time (optional - can be commented out for lazy loading)
# load_topic_model()
