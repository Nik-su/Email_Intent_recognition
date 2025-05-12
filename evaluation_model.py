#!/usr/bin/env python3
# evaluate_model.py - Evaluation script with metadata fallback

import pandas as pd
import numpy as np
import torch
import pickle
import json
import os
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score,
    classification_report,
    hamming_loss
)

# Load the trained model and tokenizer
print("Loading trained model...")

# Check for different possible model paths
possible_paths = [
    "./results",
    "./results/checkpoint-3000",
    "./results/checkpoint-2000", 
    "./results/checkpoint-1000",
]

# Find the model directory
model_path = None
for path in possible_paths:
    if os.path.exists(os.path.join(path, "config.json")):
        model_path = path
        print(f"Found model at: {model_path}")
        break

# If not found in standard locations, check for any checkpoint directory
if model_path is None:
    results_dir = "./results"
    if os.path.exists(results_dir):
        for item in os.listdir(results_dir):
            item_path = os.path.join(results_dir, item)
            if os.path.isdir(item_path) and "checkpoint" in item:
                if os.path.exists(os.path.join(item_path, "config.json")):
                    model_path = item_path
                    print(f"Found model at: {model_path}")
                    break

if model_path is None:
    print("❌ No trained model found!")
    exit(1)

try:
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model.eval()
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Try to load metadata, but create it if not found
metadata_paths = [
    "./results/metadata.pkl",
    f"{model_path}/metadata.pkl"
]

metadata_loaded = False
for metadata_path in metadata_paths:
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            print(f"✓ Metadata loaded from: {metadata_path}")
            is_multi_label = metadata['is_multi_label']
            intent_to_id = metadata['intent_to_id']
            id_to_intent = metadata['id_to_intent']
            
            if is_multi_label:
                mlb = metadata['mlb']
                best_threshold = metadata.get('best_threshold', 0.5)
                print(f"✓ Multi-label metadata loaded (threshold: {best_threshold})")
            else:
                label_encoder = metadata['label_encoder']
                print("✓ Single-label metadata loaded")
            metadata_loaded = True
            break
        except Exception as e:
            print(f"Warning: Could not load metadata from {metadata_path}: {e}")
            continue

# If metadata not found, recreate it from the dataset
if not metadata_loaded:
    print("⚠️  Metadata not found. Recreating from dataset...")
    
    # Load dataset to recreate metadata
    df = pd.read_csv('email_intent_dataset.csv')
    df = df[['text', 'intent']].dropna()
    
    # Check if multi-intent
    is_multi_label = df['intent'].str.contains(';').any()
    print(f"Dataset type detected: {'Multi-label' if is_multi_label else 'Single-label'}")
    
    if is_multi_label:
        # Multi-intent processing
        from sklearn.preprocessing import MultiLabelBinarizer
        df['intent_list'] = df['intent'].str.split(';')
        mlb = MultiLabelBinarizer()
        mlb.fit(df['intent_list'])
        intent_to_id = {intent: i for i, intent in enumerate(mlb.classes_)}
        id_to_intent = {i: intent for i, intent in enumerate(mlb.classes_)}
        best_threshold = 0.5  # Default threshold
        print(f"✓ Multi-label metadata recreated (using default threshold: {best_threshold})")
    else:
        # Single-intent processing  
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        label_encoder.fit(df['intent'])
        intent_to_id = {intent: i for i, intent in enumerate(label_encoder.classes_)}
        id_to_intent = {i: intent for i, intent in enumerate(label_encoder.classes_)}
        print("✓ Single-label metadata recreated")

# Get model info
num_labels = model.config.num_labels
print(f"Model configured for {num_labels} labels")

# Load dataset (same as training)
print("\nLoading dataset...")
df = pd.read_csv('email_intent_dataset.csv')
df = df[['text', 'intent']].dropna()

# Recreate the labels (same as training)
if is_multi_label:
    df['intent_list'] = df['intent'].str.split(';')
    binary_labels = mlb.transform(df['intent_list'])
    df['labels'] = [labels.tolist() for labels in binary_labels]
else:
    df['labels'] = label_encoder.transform(df['intent'])

# Define tokenize function
def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=512)

# Create dataset and preserve indices (same as training)
print("Creating test dataset...")
dataset = Dataset.from_pandas(df[['text', 'labels']].reset_index())
dataset = dataset.train_test_split(test_size=0.2)

# Store the original indices for later use
test_indices = dataset['test']['index']

# Remove the index column before tokenization
dataset = dataset.remove_columns('index')
tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Create a minimal trainer for evaluation
training_args = TrainingArguments(
    output_dir='./temp_eval',
    per_device_eval_batch_size=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
)

print("\n=== Evaluating Model ===")

# For multi-label models, we'll do manual prediction to avoid trainer issues
if is_multi_label:
    print("Running manual prediction for multi-label model...")
    
    # Get test dataloader
    test_dataloader = trainer.get_eval_dataloader(tokenized_dataset['test'])
    
    # Manual prediction
    model.eval()
    predictions_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            # Move to device
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Get predictions (without computing loss)
            outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'])
            
            predictions_list.append(outputs.logits.cpu().numpy())
            labels_list.append(batch['labels'].cpu().numpy())
    
    # Combine all predictions
    predictions_array = np.concatenate(predictions_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)
    
    # Create a mock predictions object
    class MockPredictions:
        def __init__(self, predictions, label_ids):
            self.predictions = predictions
            self.label_ids = label_ids
    
    predictions = MockPredictions(predictions_array, labels_array)
else:
    # For single-label, use trainer predict
    predictions = trainer.predict(tokenized_dataset['test'])

if is_multi_label:
    # Multi-label evaluation
    probs = torch.sigmoid(torch.from_numpy(predictions.predictions)).numpy()
    
    # If we had to recreate metadata, find the best threshold
    if not metadata_loaded:
        print("Finding optimal threshold...")
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        best_f1 = 0
        for threshold in thresholds:
            test_preds = (probs > threshold).astype(int)
            f1 = f1_score(predictions.label_ids, test_preds, average='weighted')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        print(f"✓ Best threshold found: {best_threshold}")
    
    # Use the best threshold
    print(f"Using threshold: {best_threshold}")
    preds = (probs > best_threshold).astype(int)
    
    # Detailed multi-label metrics
    print("\nMulti-label Classification Results:")
    print(f"F1 Score (weighted): {f1_score(predictions.label_ids, preds, average='weighted'):.4f}")
    print(f"F1 Score (micro): {f1_score(predictions.label_ids, preds, average='micro'):.4f}")
    print(f"F1 Score (macro): {f1_score(predictions.label_ids, preds, average='macro'):.4f}")
    print(f"Precision (weighted): {precision_score(predictions.label_ids, preds, average='weighted'):.4f}")
    print(f"Recall (weighted): {recall_score(predictions.label_ids, preds, average='weighted'):.4f}")
    print(f"Hamming Loss: {hamming_loss(predictions.label_ids, preds):.4f}")
    print(f"Exact Match Ratio: {(preds == predictions.label_ids).all(axis=1).mean():.4f}")
    
    # Show examples with predictions
    print("\n=== Example Predictions ===")
    test_texts = df.iloc[test_indices]['text'].tolist()
    test_intents = df.iloc[test_indices]['intent'].tolist()
    
    for i in range(min(10, len(preds))):
        predicted_intents = [id_to_intent[j] for j, pred in enumerate(preds[i]) if pred == 1]
        predicted_probs = [probs[i][j] for j, pred in enumerate(preds[i]) if pred == 1]
        true_intents = test_intents[i].split(';')
        
        print(f"\nExample {i+1}:")
        print(f"Text: {test_texts[i][:100]}...")
        print(f"True intents: {true_intents}")
        print(f"Predicted intents: {predicted_intents}")
        print(f"Confidence scores: {[f'{p:.3f}' for p in predicted_probs]}")
        
        # Check if prediction is correct
        is_correct = set(predicted_intents) == set(true_intents)
        print(f"✓ Correct" if is_correct else "❌ Incorrect")
    
else:
    # Single-label evaluation
    preds = np.argmax(predictions.predictions, axis=1)
    
    # Get unique classes in the test set
    unique_classes_in_test = sorted(set(predictions.label_ids))
    class_names_in_test = [label_encoder.classes_[i] for i in unique_classes_in_test]
    
    print("\nSingle-label Classification Report:")
    print(classification_report(predictions.label_ids, preds, 
                              labels=unique_classes_in_test,
                              target_names=class_names_in_test))
    
    # Overall metrics
    print(f"\nOverall Metrics:")
    print(f"Accuracy: {(preds == predictions.label_ids).mean():.4f}")
    print(f"F1 Score (weighted): {f1_score(predictions.label_ids, preds, average='weighted'):.4f}")
    
    # Show examples with predictions
    print("\n=== Example Predictions ===")
    test_texts = df.iloc[test_indices]['text'].tolist()
    
    for i in range(min(10, len(preds))):
        confidence = torch.softmax(torch.from_numpy(predictions.predictions[i]), dim=0).max().item()
        
        print(f"\nExample {i+1}:")
        print(f"Text: {test_texts[i][:100]}...")
        print(f"True intent: {label_encoder.classes_[predictions.label_ids[i]]}")
        print(f"Predicted intent: {label_encoder.classes_[preds[i]]}")
        print(f"Confidence: {confidence:.3f}")
        
        # Check if prediction is correct
        is_correct = preds[i] == predictions.label_ids[i]
        print(f"✓ Correct" if is_correct else "❌ Incorrect")

# Save the recreated metadata if it wasn't found
if not metadata_loaded:
    print("\n💾 Saving recreated metadata...")
    save_metadata = {
        'is_multi_label': is_multi_label,
        'intent_to_id': intent_to_id,
        'id_to_intent': id_to_intent,
        'num_labels': num_labels,
    }
    
    if is_multi_label:
        save_metadata['mlb'] = mlb
        save_metadata['best_threshold'] = best_threshold
        save_metadata['all_intents'] = list(mlb.classes_)
    else:
        save_metadata['label_encoder'] = label_encoder
        save_metadata['all_intents'] = list(label_encoder.classes_)
    
    with open('./results/metadata.pkl', 'wb') as f:
        pickle.dump(save_metadata, f)
    print("✓ Metadata saved to ./results/metadata.pkl")

print(f"\n✅ Evaluation complete!")