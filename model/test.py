import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import os
import sys

# ================= CONFIGURATION =================
# This automatically finds the 'model' folder sitting next to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model")

# THRESHOLDS (From your training)
THRESHOLDS = {
    'Academics': 0.60,       # Tinaasan (from 0.45) para mabawasan ang false positives
    'Career': 0.50,
    'Faculty': 0.50,         # Binabaan
    'Infrastructure': 0.10,  # DRASTIC DROP (from 0.95) - Dahil rare topic ito
    'Mental health': 0.50,
    'Social': 0.50,
    'Technology': 0.30       # Binabaan
}

LABELS = sorted(list(THRESHOLDS.keys()))
# =================================================

def load_model():
    print("\n" + "="*50)
    print(" 🛠️  DIAGNOSTICS")
    print("="*50)
    print(f"📂 Script Location: {SCRIPT_DIR}")
    print(f"📂 Looking for model in: {MODEL_PATH}")
    
    # 1. Check if model folder exists
    if not os.path.exists(MODEL_PATH):
        print("\n❌ ERROR: The folder 'model' was not found.")
        print(f"   -> I expected it here: {MODEL_PATH}")
        print("   -> Please make sure the folder is named 'model' and is next to test.py")
        return None, None
    
    print("✅ Folder found.")
    
    try:
        # 2. Load Tokenizer from INTERNET (Crucial Step)
        # We MUST download the dictionary from the web because it's not in your local folder
        print("⬇️  Downloading/Loading Tokenizer (Dictionary)...")
        tokenizer = AutoTokenizer.from_pretrained("jcblaise/roberta-tagalog-base")
        
        # 3. Load Model from LOCAL FOLDER
        print(f"🧠 Loading Model Weights (Brain) from local folder...")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        
        print("✅ Model loaded successfully!")
        return tokenizer, model
    except OSError as e:
        print(f"\n❌ ERROR: {e}")
        print("   -> Make sure you are connected to the internet (for the tokenizer).")
        print("   -> Make sure 'pytorch_model.bin' and 'config.json' are inside the 'model' folder.")
        return None, None
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        return None, None

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def predict(sentence, tokenizer, model):
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
    if isinstance(probs, float): probs = [probs]
    
    results = []
    for i, label in enumerate(LABELS):
        score = probs[i]
        threshold = THRESHOLDS.get(label, 0.5)
        if score >= threshold:
            results.append((label, score))
            
    return results

def main():
    tokenizer, model = load_model()
    if not model: return

    print("\n" + "="*60)
    print(" 🤖 TAGLISH TOPIC CLASSIFIER IS READY")
    print("="*60)

    while True:
        try:
            user_input = input("\n📝 Enter paragraph (or 'exit'): ")
        except EOFError:
            break
            
        if user_input.lower() in ['exit', 'quit']: break
        if not user_input.strip(): continue

        sentences = split_into_sentences(user_input)
        print(f"\n🔍 Analyzed {len(sentences)} sentences:\n")

        for i, sent in enumerate(sentences):
            predictions = predict(sent, tokenizer, model)
            print(f"  Sentence {i+1}: \"{sent}\"")
            
            if predictions:
                predictions.sort(key=lambda x: x[1], reverse=True)
                for label, score in predictions:
                    print(f"      • {label} ({score*100:.1f}%)")
            else:
                print("      ❌ No specific topic detected.")
            print("-" * 40)

if __name__ == "__main__":
    main()