

import os
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "bert-base-multilingual-cased"
MAX_LENGTH = 128
NUM_LABELS = 3

# Training config
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# Use combined dataset with real interviews
DATASET_PATH = "Taglish_Training_Dataset_Combined.xlsx"
OUTPUT_DIR = "./taglish_mbert_model_final"

LABEL_MAP = {"Positive": 0, "Neutral": 1, "Negative": 2}
INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(file_path):
    """Load the expanded dataset."""
    print("=" * 60)
    print("📊 LOADING EXPANDED DATASET")
    print("=" * 60)
    
    df = pd.read_excel(file_path)
    print(f"✓ Total samples: {len(df)}")
    print(f"✓ Unique texts: {df['text'].nunique()}")
    
    # Map original readiness labels to sentiment
    # High -> Positive (0), Medium -> Neutral (1), Low -> Negative (2)
    sentiment_map = {"High": 0, "Medium": 1, "Low": 2}
    df['label'] = df['label_readiness'].map(sentiment_map)
    
    print("\n📈 Class Distribution:")
    for label_name, label_id in LABEL_MAP.items():
        count = (df['label'] == label_id).sum()
        print(f"   {label_name}: {count}")
    
    return df


def split_data(df, test_size=0.3):
    """Split into train/validation."""
    print("\n" + "=" * 60)
    print("📂 SPLITTING DATA (70/30)")
    print("=" * 60)
    
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=SEED, stratify=df['label']
    )
    
    print(f"✓ Training: {len(train_df)}")
    print(f"✓ Validation: {len(val_df)}")
    
    return train_df, val_df


# ============================================================================
# TOKENIZATION
# ============================================================================

def tokenize_data(train_df, val_df, tokenizer):
    """Tokenize the data."""
    print("\n" + "=" * 60)
    print("🔤 TOKENIZING DATA")
    print("=" * 60)
    
    def tokenize_fn(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH
        )
    
    train_ds = Dataset.from_pandas(train_df[['text', 'label']])
    val_ds = Dataset.from_pandas(val_df[['text', 'label']])
    
    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)
    
    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")
    
    train_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    print(f"✓ Tokenized: train={len(train_ds)}, val={len(val_ds)}")
    
    return train_ds, val_ds


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(eval_pred):
    """Compute metrics during training."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def final_evaluation(trainer, val_ds):
    """Final detailed evaluation."""
    print("\n" + "=" * 60)
    print("📊 FINAL EVALUATION")
    print("=" * 60)
    
    preds = trainer.predict(val_ds)
    logits = preds.predictions
    labels = preds.label_ids
    
    pred_classes = np.argmax(logits, axis=-1)
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    confidence = np.max(probs, axis=-1)
    
    acc = accuracy_score(labels, pred_classes)
    print(f"\n🎯 Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    print("\n📋 Classification Report:")
    print("-" * 60)
    print(classification_report(labels, pred_classes, 
                                target_names=['Positive', 'Neutral', 'Negative'], digits=4))
    
    print("📊 Confusion Matrix:")
    cm = confusion_matrix(labels, pred_classes)
    print(f"\n{'':>10} | {'Positive':>8} | {'Neutral':>8} | {'Negative':>8}")
    print("-" * 45)
    for i, row in enumerate(cm):
        label = INVERSE_LABEL_MAP[i]
        print(f"{label:>10} | {row[0]:>8} | {row[1]:>8} | {row[2]:>8}")
    
    print("\n📈 Confidence Statistics:")
    print(f"   Mean: {np.mean(confidence):.4f}")
    print(f"   Min:  {np.min(confidence):.4f}")
    print(f"   Max:  {np.max(confidence):.4f}")
    
    return acc, pred_classes, confidence


# ============================================================================
# TRAINING
# ============================================================================

def train_model():
    """Main training function."""
    print("\n" + "=" * 70)
    print("   MBERT FINE-TUNING ON EXPANDED TAGLISH DATASET")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n⚙️  Device: {device}")
    
    # Load data
    df = load_data(DATASET_PATH)
    train_df, val_df = split_data(df)
    
    # Tokenizer and model
    print("\n📥 Loading mBERT...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS,
        id2label=INVERSE_LABEL_MAP, label2id=LABEL_MAP
    )
    model.to(device)
    
    train_ds, val_ds = tokenize_data(train_df, val_df, tokenizer)
    
    # Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        seed=SEED,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    
    print("\n⚙️  Config: epochs=3, batch=8, lr=2e-5")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    print("\n🏋️  Training...")
    trainer.train()
    
    # Evaluate
    acc, _, _ = final_evaluation(trainer, val_ds)
    
    # Save
    print("\n💾 Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✓ Saved to: {OUTPUT_DIR}")

    # Save as 1 file (Complete Model Object)
    full_model_path = "taglish_sentiment_model_full.pth"
    # Ensure compatibility by saving the underlying model (unwrapped)
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save, full_model_path)
    print(f"✓ Saved as single file to: {full_model_path}")
    
    return trainer, tokenizer, acc


# ============================================================================
# INFERENCE
# ============================================================================

def predict_readiness(text, model_path=OUTPUT_DIR):
    """Predict readiness for new text."""
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    inputs = tokenizer(text, padding='max_length', truncation=True,
                       max_length=MAX_LENGTH, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    
    return {
        'text': text,
        'predicted_label': INVERSE_LABEL_MAP[pred],
        'confidence': float(probs[pred]),
        'all_probabilities': {
            'Positive': float(probs[0]),
            'Neutral': float(probs[1]),
            'Negative': float(probs[2])
        }
    }


def test_examples():
    """Test with examples."""
    print("\n" + "=" * 60)
    print("🧪 TESTING")
    print("=" * 60)
    
    tests = [
        "Confident na ako sa Python ko, marami na akong projects na nagawa.",
        "May basic knowledge ako sa programming, pero need ko pa mag-practice.",
        "Wala pa akong background sa coding, pero willing to learn.",
        "Na-master ko na ang web development, ready na ako mag-work.",
        "Struggling ako sa algorithms, ang hirap intindihin.",
    ]
    
    for i, text in enumerate(tests, 1):
        r = predict_readiness(text)
        print(f"\n📝 Test {i}:")
        print(f"   \"{text[:50]}...\"")
        print(f"   → {r['predicted_label']} ({r['confidence']:.1%})")


if __name__ == "__main__":
    trainer, tokenizer, acc = train_model()
    test_examples()
    
    print("\n" + "=" * 60)
    print("✅ COMPLETE!")
    print("=" * 60)
    print(f"📊 Accuracy: {acc:.2%}")
    print(f"📂 Model: {OUTPUT_DIR}")
    print("\n💡 Usage:")
    print("   from train_mbert_expanded import predict_readiness")
    print("   result = predict_readiness('Your text')")
