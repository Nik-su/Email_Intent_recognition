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

import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast

# Accept input from command-line
email_text = sys.argv[1]

# Load label mapping
df = pd.read_csv("email_intent_dataset.csv")
label_encoder = LabelEncoder()
label_encoder.fit(df['intent'])
id2label = {i: label for i, label in enumerate(label_encoder.classes_)}

# Load model
model_path = "./results/checkpoint-45"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model.eval()

# Predict
inputs = tokenizer(email_text, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_label_id = torch.argmax(probs, dim=1).item()
    predicted_intent = id2label[pred_label_id]
    confidence = round(probs[0][pred_label_id].item(), 4)

# Print the result as a string
print(f"{predicted_intent}|{confidence}")
