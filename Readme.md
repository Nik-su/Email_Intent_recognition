# 📧 Email Intent Classification System

A BERT-based deep learning model for classifying emails into real estate business intents. Supports both single and multi-intent classification with high accuracy.

## 🚀 Features

- ✅ **8 Intent Categories**: Lease abstraction, clause protection, company research, etc.
- ✅ **Multi-Intent Support**: Handle emails with multiple tasks (e.g., "Extract lease terms AND check clauses")
- ✅ **BERT-based Model**: Fine-tuned on 10,000+ business emails
- ✅ **FastAPI Integration**: REST API for real-time predictions
- ✅ **High Accuracy**: 100% accuracy on test dataset
- ✅ **Auto-Detection**: Automatically handles single vs. multi-intent classification

## 📋 Intent Categories

1. **Intent_Transaction_Date_navigator** - Extract transaction dates and timelines
2. **Intent_Clause_Protect** - Identify risky or problematic clauses
3. **Intent_Lease_Abstraction** - Extract key lease terms and provisions
4. **Intent_Comparison_LOI_Lease** - Compare LOI with final lease terms
5. **Intent_Company_research** - Research company background and credibility
6. **Intent_Amendment_Abstraction** - Analyze lease amendments
7. **Intent_Sales_Listings_Comparison** - Compare property sales listings
8. **Intent_Lease_Listings_Comparison** - Compare lease offerings

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- GPU recommended for training (but not required)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/email-intent-classification.git
cd email-intent-classification

# Install dependencies
pip install -r requirements.txt

# Generate dataset
python dataset.py

# Train the model
python model_training.py

# Test the model
python Inference_test.py "Please extract lease terms and check for risky clauses"
```

## 🎯 Quick Start

### 1. Generate Dataset
```bash
python dataset.py
```
*Creates 10,000+ examples with single and multi-intent labels*

### 2. Train Model
```bash
python model_training.py
```
*Training takes ~1 hour on GPU, ~3 hours on CPU*

### 3. Single Prediction
```bash
python Inference_test.py "Extract the lease terms from this document"
# Output: Intent_Lease_Abstraction|0.9542
```

### 4. Start API Server
```bash
python app.py
```
*API available at: http://localhost:8000*

## 🌐 API Usage

### REST API Examples

**Single Intent:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"email_text": "Please abstract the lease terms"}'
```

**Multi Intent:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"email_text": "Extract lease terms and check for risky clauses"}'
```

**Response Format:**
```json
{
  "email_text": "Extract lease terms and check clauses",
  "predicted_intent": "Intent_Lease_Abstraction;Intent_Clause_Protect",
  "confidence": 0.8721
}


## 🔧 Technical Details

### Model Architecture
- **Base Model**: BERT-base-uncased (110M parameters)
- **Custom Layer**: FlexibleBertModel
- **Loss Functions**: 
  - Single-intent: CrossEntropyLoss
  - Multi-intent: BCEWithLogitsLoss

### Training Configuration
- **Epochs**: 3
- **Batch Size**: 8
- **Learning Rate**: 2e-5
- **Max Sequence Length**: 512 tokens
- **Data Split**: 80% train, 20% test

### Performance Metrics
- **F1 Score**: 1.000
- **Precision**: 1.000
- **Recall**: 1.000
- **Exact Match Ratio**: 1.000

## 📊 Dataset Details

- **Total Examples**: 10,000+
- **Single-Intent**: 5,600 examples (700 per intent)
- **Multi-Intent**: 4,400 examples
- **Augmentation**: NLP libraries (nlpaug, synonym replacement)
- **Language**: English
- **Domain**: Real Estate Business

## 🚀 Examples

### Single Intent Examples
```python
# Extract lease terms
"Please abstract the lease for the Johnson project"
→ Intent_Lease_Abstraction

# Check clauses
"Review this lease for any risky provisions"
→ Intent_Clause_Protect
```

### Multi Intent Examples
```python
# Two intents
"Extract lease terms and check for risky clauses"
→ Intent_Lease_Abstraction;Intent_Clause_Protect

# Three intents
"Abstract the lease, check clauses, and research the company"
→ Intent_Lease_Abstraction;Intent_Clause_Protect;Intent_Company_research
```