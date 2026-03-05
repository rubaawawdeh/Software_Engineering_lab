# ==============================================
# Requirement Classification Pipeline
# Classes: Clear | Unclear | Incomplete | Conflict
# ==============================================

import spacy
import re
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# ------------------------
# 1. Load SpaCy Model
# ------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Error: SpaCy model 'en_core_web_sm' not found. Please run:")
    print("python -m spacy download en_core_web_sm")
    exit()

# ------------------------
# 2. Raw Requirement Input
# ------------------------
raw_requirement = "The system MUST allow users to login securely, and it should not store passwords in plain text!!!"

# ------------------------
# 3. Cleaning Function
# ------------------------
def clean_requirement(text):
    text = text.lower()  # lowercase
    text = text.strip('“"').rstrip('!').strip()  # remove quotes/exclamations
    text = text.replace(", and it should", " and")  # custom fix
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

clean_output = clean_requirement(raw_requirement)

# ------------------------
# 4. Segmentation Function
# ------------------------
# Split on "and not" and rebuild second segment
parts = clean_output.split(" and not ", 1)
segment_1 = parts[0].strip()
segment_2 = "the system must not " + parts[1].strip() if len(parts) > 1 else ""
segments = [segment_1, segment_2] if segment_2 else [segment_1]

# ------------------------
# 5. Tokenization
# ------------------------
all_tokens = []
for i, segment in enumerate(segments):
    doc = nlp(segment)
    tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
    all_tokens.append(tokens)

# ------------------------
# 6. Rule Lists
# ------------------------
incomplete_terms = [
    "secure", "securely", "fast", "efficient", "easy", 
    "user friendly", "reliable", "robust", "quick", "intuitive", 
    "scalable", "maintainable", "flexible", "high performance"
]

unclear_terms = [
    "should", "may", "could", "might", 
    "as needed", "appropriate", "optimal", 
    "sufficient", "reasonable", "acceptable"
]

conflict_patterns = [
    r"but", r"however", r"although", r"yet", 
    r"not .* and .*", r"either .* or .*"
]

# ------------------------
# 7. Rule Functions
# ------------------------
def is_incomplete(text):
    text_lower = text.lower()
    return any(term in text_lower for term in incomplete_terms)

def is_unclear(text):
    text_lower = text.lower()
    return any(term in text_lower for term in unclear_terms)

def is_conflict(text):
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in conflict_patterns)

# ------------------------
# 8. Load BERT Model (optional fallback)
# ------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=4
)

label_map = {
    0: "Clear",
    1: "Unclear",
    2: "Incomplete",
    3: "Conflict"
}

# ------------------------
# 9. Classification Function
# ------------------------
def classify_requirement(text):
    # Rule-based first
    if is_conflict(text):
        return "Conflict"
    if is_incomplete(text):
        return "Incomplete"
    if is_unclear(text):
        return "Unclear"
    
    # Fallback to BERT (optional, currently random predictions if not fine-tuned)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    
    return label_map[pred]

# ------------------------
# 10. Apply Classification
# ------------------------
for i, segment in enumerate(segments):
    label = classify_requirement(segment)
    print(f"Segment {i+1}: {segment}")
    print(f"Tokens: {all_tokens[i]}")
    print(f"Classification: {label}\n")
