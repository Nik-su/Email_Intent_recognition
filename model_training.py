


# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import classification_report
# from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
# from datasets import Dataset
# import torch


# df = pd.read_csv('/content/email_intent_dataset.csv') 
# df = df[['text', 'intent']]  
# df = df.dropna()


# label_encoder = LabelEncoder()
# df['label'] = label_encoder.fit_transform(df['intent'])
# label2id = {label: i for i, label in enumerate(label_encoder.classes_)}
# id2label = {v: k for k, v in label2id.items()}


# tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# def tokenize(batch):
#     return tokenizer(batch['text'], padding='max_length', truncation=True)

# dataset = Dataset.from_pandas(df[['text', 'label']])
# dataset = dataset.train_test_split(test_size=0.2)
# tokenized_dataset = dataset.map(tokenize, batched=True)
# tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
# tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])


# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label2id), id2label=id2label, label2id=label2id)


# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     logging_dir='./logs',
  
# )


# from sklearn.metrics import accuracy_score, f1_score

# def compute_metrics(p):
#     preds = np.argmax(p.predictions, axis=1)
#     return {
#         'accuracy': accuracy_score(p.label_ids, preds),
#         'f1': f1_score(p.label_ids, preds, average='weighted')
#     }


# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset['train'],
#     eval_dataset=tokenized_dataset['test'],
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )

# trainer.train()


# predictions = trainer.predict(tokenized_dataset['test'])
# y_true = predictions.label_ids
# y_pred = np.argmax(predictions.predictions, axis=1)

# print("Classification Report:\n")
# print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

'Updated-training'
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
from torch import nn
import pickle
import json

# Custom model that handles both single and multi-label
class FlexibleBertModel(BertForSequenceClassification):
    def __init__(self, config, is_multi_label=False):
        super().__init__(config)
        self.is_multi_label = is_multi_label
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Filter out kwargs that BERT doesn't expect
        bert_kwargs = {k: v for k, v in kwargs.items() if k not in ['num_items_in_batch']}
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, **bert_kwargs)
        pooled_output = self.dropout(outputs[1])
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.is_multi_label:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float())
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {'loss': loss, 'logits': logits}

def analyze_dataset(df):
    """Analyze the dataset to understand single vs multi-intent distribution"""
    print("=== Dataset Analysis ===")
    
    # Check for multi-intent examples
    has_multi_intent = df['intent'].str.contains(';').any()
    
    if has_multi_intent:
        # Split intents and count occurrences
        df['intent_list'] = df['intent'].str.split(';')
        df['num_intents'] = df['intent_list'].apply(len)
        
        print("Multi-intent dataset detected!")
        print(f"Total examples: {len(df)}")
        print("\nNumber of intents per example:")
        print(df['num_intents'].value_counts().sort_index())
        
        # Show examples of different multi-intent cases
        print("\nExample of 2-intent case:")
        two_intent_example = df[df['num_intents'] == 2].iloc[0]
        print(f"Text: {two_intent_example['text'][:100]}...")
        print(f"Intents: {two_intent_example['intent']}")
        
        if (df['num_intents'] >= 3).any():
            print("\nExample of 3+ intent case:")
            multi_intent_example = df[df['num_intents'] >= 3].iloc[0]
            print(f"Text: {multi_intent_example['text'][:100]}...")
            print(f"Intents: {multi_intent_example['intent']}")
    else:
        print("Single-intent dataset detected!")
        print(f"Total examples: {len(df)}")
        print("\nIntent distribution:")
        print(df['intent'].value_counts())
    
    return has_multi_intent

# Load dataset
print("Loading dataset...")
df = pd.read_csv('email_intent_dataset.csv')
df = df[['text', 'intent']].dropna()

# Analyze dataset
is_multi_intent = analyze_dataset(df)

# Process labels based on type
if is_multi_intent:
    print("\n=== Processing Multi-Intent Dataset ===")
    # Multi-intent processing
    df['intent_list'] = df['intent'].str.split(';')
    mlb = MultiLabelBinarizer()
    binary_labels = mlb.fit_transform(df['intent_list'])
    df['labels'] = [labels.tolist() for labels in binary_labels]
    intent_to_id = {intent: i for i, intent in enumerate(mlb.classes_)}
    id_to_intent = {i: intent for i, intent in enumerate(mlb.classes_)}
    num_labels = len(mlb.classes_)
    
    print(f"Number of unique intents: {num_labels}")
    print("All intents:", list(mlb.classes_))
else:
    print("\n=== Processing Single-Intent Dataset ===")
    # Single-intent processing
    label_encoder = LabelEncoder()
    df['labels'] = label_encoder.fit_transform(df['intent'])
    intent_to_id = {intent: i for i, intent in enumerate(label_encoder.classes_)}
    id_to_intent = {i: intent for i, intent in enumerate(label_encoder.classes_)}
    num_labels = len(label_encoder.classes_)
    
    print(f"Number of unique intents: {num_labels}")
    print("All intents:", list(label_encoder.classes_))

# Initialize tokenizer
print("\nInitializing tokenizer...")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=512)

# Create dataset
print("Creating dataset...")
dataset = Dataset.from_pandas(df[['text', 'labels']])
dataset = dataset.train_test_split(test_size=0.2)
tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Initialize model
print("Initializing model...")
model = FlexibleBertModel.from_pretrained(
    'bert-base-uncased',
    num_labels=num_labels,
    is_multi_label=is_multi_intent
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
    #evaluation_strategy="epoch",
    #save_strategy="epoch",
    #load_best_model_at_end=True,
    metric_for_best_model="eval_f1" if is_multi_intent else "eval_accuracy",
    greater_is_better=True,
)

# Enhanced compute metrics function
def compute_metrics(p):
    if is_multi_intent:
        # Multi-label metrics with threshold optimization
        probs = torch.sigmoid(torch.from_numpy(p.predictions)).numpy()
        
        # Test different thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            preds = (probs > threshold).astype(int)
            from sklearn.metrics import f1_score
            f1 = f1_score(p.label_ids, preds, average='weighted')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # Use best threshold for final metrics
        preds = (probs > best_threshold).astype(int)
        
        from sklearn.metrics import (
            f1_score, 
            precision_score, 
            recall_score, 
            accuracy_score,
            hamming_loss
        )
        
        return {
            'f1': f1_score(p.label_ids, preds, average='weighted'),
            'f1_micro': f1_score(p.label_ids, preds, average='micro'),
            'precision': precision_score(p.label_ids, preds, average='weighted'),
            'recall': recall_score(p.label_ids, preds, average='weighted'),
            'hamming_loss': hamming_loss(p.label_ids, preds),
            'exact_match': (preds == p.label_ids).all(axis=1).mean(),
            'best_threshold': best_threshold
        }
    else:
        # Single-label metrics
        preds = np.argmax(p.predictions, axis=1)
        from sklearn.metrics import accuracy_score, f1_score
        return {
            'accuracy': accuracy_score(p.label_ids, preds),
            'f1': f1_score(p.label_ids, preds, average='weighted')
        }

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train model
print("\n=== Training Model ===")
trainer.train()

# Detailed evaluation
print("\n=== Evaluating Model ===")
predictions = trainer.predict(tokenized_dataset['test'])

if is_multi_intent:
    # Multi-label evaluation
    probs = torch.sigmoid(torch.from_numpy(predictions.predictions)).numpy()
    
    # Find best threshold from validation
    eval_results = trainer.evaluate()
    best_threshold = eval_results.get('eval_best_threshold', 0.5)
    print(f"Using threshold: {best_threshold}")
    
    preds = (probs > best_threshold).astype(int)
    
    # Detailed multi-label metrics
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    print("\nMulti-label Classification Results:")
    print(f"F1 Score (weighted): {f1_score(predictions.label_ids, preds, average='weighted'):.4f}")
    print(f"F1 Score (micro): {f1_score(predictions.label_ids, preds, average='micro'):.4f}")
    print(f"Precision (weighted): {precision_score(predictions.label_ids, preds, average='weighted'):.4f}")
    print(f"Recall (weighted): {recall_score(predictions.label_ids, preds, average='weighted'):.4f}")
    print(f"Exact Match Ratio: {(preds == predictions.label_ids).all(axis=1).mean():.4f}")
    
    # Show examples with predictions
    print("\n=== Example Predictions ===")
    test_texts = df.iloc[tokenized_dataset['test'].indices]['text'].tolist()
    test_intents = df.iloc[tokenized_dataset['test'].indices]['intent'].tolist()
    
    for i in range(min(5, len(preds))):
        predicted_intents = [id_to_intent[j] for j, pred in enumerate(preds[i]) if pred == 1]
        predicted_probs = [probs[i][j] for j, pred in enumerate(preds[i]) if pred == 1]
        true_intents = test_intents[i].split(';')
        
        print(f"\nExample {i+1}:")
        print(f"Text: {test_texts[i][:100]}...")
        print(f"True intents: {true_intents}")
        print(f"Predicted intents: {predicted_intents}")
        print(f"Confidence scores: {[f'{p:.3f}' for p in predicted_probs]}")
        
    # Save threshold for inference
    metadata_extra = {'best_threshold': best_threshold}
    
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
    
    # Show examples with predictions
    print("\n=== Example Predictions ===")
    test_texts = df.iloc[tokenized_dataset['test'].indices]['text'].tolist()
    
    for i in range(min(5, len(preds))):
        confidence = torch.softmax(torch.from_numpy(predictions.predictions[i]), dim=0).max().item()
        
        print(f"\nExample {i+1}:")
        print(f"Text: {test_texts[i][:100]}...")
        print(f"True intent: {label_encoder.classes_[predictions.label_ids[i]]}")
        print(f"Predicted intent: {label_encoder.classes_[preds[i]]}")
        print(f"Confidence: {confidence:.3f}")
        
    metadata_extra = {}

# Save model and metadata
print("\n=== Saving Model ===")
model.save_pretrained('./results')
tokenizer.save_pretrained('./results')

# Save comprehensive metadata for inference
metadata = {
    'is_multi_label': is_multi_intent,
    'intent_to_id': intent_to_id,
    'id_to_intent': id_to_intent,
    'num_labels': num_labels,
    **metadata_extra
}

if is_multi_intent:
    metadata['mlb'] = mlb
    metadata['all_intents'] = list(mlb.classes_)
else:
    metadata['label_encoder'] = label_encoder
    metadata['all_intents'] = list(label_encoder.classes_)

with open('./results/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

# Save summary report
summary = {
    'dataset_type': 'multi-intent' if is_multi_intent else 'single-intent',
    'total_examples': len(df),
    'num_intents': num_labels,
    'train_examples': len(tokenized_dataset['train']),
    'test_examples': len(tokenized_dataset['test']),
}

if is_multi_intent:
    summary['best_threshold'] = best_threshold
    summary['max_intents_per_example'] = int(df['num_intents'].max())

with open('./results/training_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n=== Training Completed Successfully! ===")
print(f"Model type: {'Multi-label' if is_multi_intent else 'Single-label'}")
print(f"Model saved to: './results'")
print(f"Total examples trained on: {len(df)}")
if is_multi_intent:
    print(f"Supports up to {int(df['num_intents'].max())} intents per example")
    print(f"Best threshold for inference: {best_threshold}")
print("Ready for inference!")