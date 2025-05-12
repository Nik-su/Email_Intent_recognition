# import pandas as pd
# from sklearn.preprocessing import LabelEncoder

# # Load your dataset again to get the labels
# df = pd.read_csv('/content/email_intent_dataset.csv')  # adjust path if needed

# # Fit the label encoder
# label_encoder = LabelEncoder()
# label_encoder.fit(df['intent'])

# # Create the mapping from ID (number) to label (string)
# id2label = {i: label for i, label in enumerate(label_encoder.classes_)}

# from transformers import BertForSequenceClassification, BertTokenizerFast

# model_path = "./results/checkpoint-45"  # or your saved path

# model = BertForSequenceClassification.from_pretrained(model_path)
# tokenizer = BertTokenizerFast.from_pretrained(model_path)
# email_text ="Check the Madison Avenue lease for missing protections - need to see if we have adequate coverage." 
# #"Extract all key dates from the office building purchase agreement - closing, contingencies, everything."
# #"Please abstract the lease for the Johnson project (PDF attached). We need to know the base rent, commencement and expiry dates, renewal options, and escalation schedule."
# #"Could you do a background check on Wexford Corp before we proceed? I'm particularly interested in any public disputes or bankruptcies in the past 5 years."
# #"Hi team, Can you pull together a schedule of important dates for the escrow process on the 125 King St deal? We're especially concerned with closing and due diligence periods. Thanks!"
# #"Please check if there are any subletting clauses missing in the lease for the Brooklyn site."
# import torch

# # Tokenize input
# inputs = tokenizer(email_text, return_tensors="pt", truncation=True, padding=True)

# # Get model output
# with torch.no_grad():
#     outputs = model(**inputs)
#     probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
#     pred_label_id = torch.argmax(probs, dim=1).item()
# # Use this only if you saved it
# predicted_intent = id2label[pred_label_id]
# print(f"Predicted Intent: {predicted_intent}")

# Inference_test.py

# import sys
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# import torch
# from transformers import BertForSequenceClassification, BertTokenizerFast

# # Accept input from command-line
# email_text = sys.argv[1]

# # Load label mapping
# df = pd.read_csv("email_intent_dataset.csv")
# label_encoder = LabelEncoder()
# label_encoder.fit(df['intent'])
# id2label = {i: label for i, label in enumerate(label_encoder.classes_)}

# # Load model
# model_path = "./results/checkpoint-45"
# model = BertForSequenceClassification.from_pretrained(model_path)
# tokenizer = BertTokenizerFast.from_pretrained(model_path)
# model.eval()

# # Predict
# inputs = tokenizer(email_text, return_tensors="pt", truncation=True, padding=True)
# with torch.no_grad():
#     outputs = model(**inputs)
#     probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
#     pred_label_id = torch.argmax(probs, dim=1).item()
#     predicted_intent = id2label[pred_label_id]
#     confidence = round(probs[0][pred_label_id].item(), 4)

# # Print the result as a string
# print(f"{predicted_intent}|{confidence}")


'multi-intent'
#!/usr/bin/env python3
# Inference_test.py - Updated for multi-intent compatibility

import sys
import os
import pandas as pd
import torch
import pickle
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizerFast

# Check command line arguments
if len(sys.argv) < 2:
    print("Usage: python3 Inference_test.py \"email text here\"")
    sys.exit(1)

email_text = sys.argv[1]

# Find the model path
model_paths = [
    # "./results",
    # "./results/checkpoint-3000",
    "/content/drive/MyDrive/model_log/results/checkpoint-2500"
    # "./results/checkpoint-1000",
]

model_path = None
for path in model_paths:
    if os.path.exists(os.path.join(path, "config.json")):
        model_path = path
        break

# If not found, check for any checkpoint directory
if model_path is None:
    results_dir = "email_model/results"
    if os.path.exists(results_dir):
        for item in os.listdir(results_dir):
            item_path = os.path.join(results_dir, item)
            if os.path.isdir(item_path) and "checkpoint" in item:
                if os.path.exists(os.path.join(item_path, "config.json")):
                    model_path = item_path
                    break

if model_path is None:
    print("ERROR: No trained model found")
    sys.exit(1)

# Load model and tokenizer
try:
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model.eval()
except Exception as e:
    print(f"ERROR: Could not load model - {e}")
    sys.exit(1)

# Try to load metadata, create fallback if not found
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
            
            is_multi_label = metadata['is_multi_label']
            intent_to_id = metadata['intent_to_id']
            id_to_intent = metadata['id_to_intent']
            
            if is_multi_label:
                mlb = metadata['mlb']
                best_threshold = metadata.get('best_threshold', 0.5)
            else:
                label_encoder = metadata['label_encoder']
            
            metadata_loaded = True
            break
        except:
            continue

# If metadata not found, recreate from dataset
if not metadata_loaded:
    # Load dataset to recreate mappings
    try:
        df = pd.read_csv("email_intent_dataset.csv")
        
        # Detect if multi-intent
        is_multi_label = df['intent'].str.contains(';').any()
        
        if is_multi_label:
            from sklearn.preprocessing import MultiLabelBinarizer
            df['intent_list'] = df['intent'].str.split(';')
            mlb = MultiLabelBinarizer()
            mlb.fit(df['intent_list'])
            intent_to_id = {intent: i for i, intent in enumerate(mlb.classes_)}
            id_to_intent = {i: intent for i, intent in enumerate(mlb.classes_)}
            best_threshold = 0.5  # Default threshold
        else:
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            label_encoder.fit(df['intent'])
            intent_to_id = {intent: i for i, intent in enumerate(label_encoder.classes_)}
            id_to_intent = {i: intent for i, intent in enumerate(label_encoder.classes_)}
        
    except Exception as e:
        print(f"ERROR: Could not load dataset or recreate metadata - {e}")
        sys.exit(1)

# Predict intent
try:
    # Tokenize input
    inputs = tokenizer(email_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        if is_multi_label:
            # Multi-label prediction
            probs = torch.sigmoid(logits).squeeze()
            
            # Use best threshold or default
            threshold = best_threshold if 'best_threshold' in locals() else 0.5
            predicted_indices = (probs > threshold).nonzero(as_tuple=True)[0]
            
            # If no intents exceed threshold, take the top one
            if len(predicted_indices) == 0:
                predicted_indices = [torch.argmax(probs).item()]
            
            # Get predicted intents and their probabilities
            predicted_intents = []
            intent_probabilities = []
            
            for idx in predicted_indices:
                intent_name = id_to_intent[idx.item()]
                probability = probs[idx].item()
                predicted_intents.append(intent_name)
                intent_probabilities.append(probability)
            
            # Sort by probability (highest first)
            paired_results = list(zip(predicted_intents, intent_probabilities))
            paired_results.sort(key=lambda x: x[1], reverse=True)
            
            # Format output - join multiple intents with semicolon
            intent_strings = [intent for intent, _ in paired_results]
            result_intent = ";".join(intent_strings)
            
            # Calculate average confidence for multi-intent
            avg_confidence = np.mean([prob for _, prob in paired_results])
            
            # Return result in expected format
            print(f"{result_intent}|{avg_confidence:.4f}")
            
        else:
            # Single-label prediction
            probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()
            pred_label_id = torch.argmax(probs, dim=0).item()
            confidence = probs[pred_label_id].item()
            
            # Get predicted intent
            predicted_intent = id_to_intent[pred_label_id]
            
            # Return result in expected format
            print(f"{predicted_intent}|{confidence:.4f}")
            
except Exception as e:
    print(f"ERROR: Prediction failed - {e}")
    sys.exit(1)
