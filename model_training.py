


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch


df = pd.read_csv('/content/email_intent_dataset.csv') 
df = df[['text', 'intent']]  
df = df.dropna()


label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['intent'])
label2id = {label: i for i, label in enumerate(label_encoder.classes_)}
id2label = {v: k for k, v in label2id.items()}


tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True)

dataset = Dataset.from_pandas(df[['text', 'label']])
dataset = dataset.train_test_split(test_size=0.2)
tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])


model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label2id), id2label=id2label, label2id=label2id)


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
  
)


from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        'accuracy': accuracy_score(p.label_ids, preds),
        'f1': f1_score(p.label_ids, preds, average='weighted')
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()


predictions = trainer.predict(tokenized_dataset['test'])
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)

print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
