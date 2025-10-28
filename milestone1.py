import pandas as pd
import ast
import spacy
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
# ======================
# Load & Prepare Dataset
# ======================
data = pd.read_csv("bank_chatbot_dataset_large.csv")

# Convert entity strings to dictionary
def parse_entities(x):
    if pd.isnull(x):
        return {}
    try:
        # Handle format like PERSON:Teja|MONEY:500
        entities = {}
        parts = x.split("|")
        for p in parts:
            if ":" in p:
                k, v = p.split(":", 1)
                entities[k.strip()] = v.strip()
        return entities
    except Exception as e:
        return {}

data["entities"] = data["entities"].apply(parse_entities)


# Features and labels
X = data["text"]
y = data["intent"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======================
# Build Intent Classifier
# ======================
intent_model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2))),  # capture unigrams & bigrams
    ("clf", LogisticRegression(max_iter=1000))
])

# Train model
intent_model.fit(X_train, y_train)

# Evaluate
print("=== Intent Classification Report ===")
y_pred = intent_model.predict(X_test)
print(classification_report(y_test, y_pred,zero_division=0))
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)*100
print(f"\naccuracy:{accuracy:.2f}%")

# ======================
# Load SpaCy Model
# ======================
nlp = spacy.load("en_core_web_sm")

# ======================
# Entity Extraction & Slot Filling
# ======================
def extract_entities(query_text):
    """
    Extract entities from a query using spaCy + custom rules.
    Returns a dictionary with slots filled.
    """
    doc = nlp(query_text)
    entities = {ent.label_.lower(): ent.text for ent in doc.ents}

    # Custom rule: account number (assume numeric, 5+ digits)
    for token in doc:
        if token.like_num and len(token.text) >= 5:
            entities["account_number"] = token.text

    # Custom rule: account type
    account_keywords = ["savings", "current", "checking"]
    for word in account_keywords:
        if word in query_text.lower():
            entities["account_type"] = word

    # Custom rule: amount (if number with currency or standalone number)
    for i, token in enumerate(doc):
      if token.like_num:
        # Check previous token for keywords
        if i > 0 and doc[i-1].lemma_.lower() in ["transfer", "send", "pay", "deposit", "credit"]:
            entities["amount"] = token.text
    if "date" in entities:
        entities.pop("date")
    # You can add more rules here for loan type, IFSC, date, etc.

    return entities

# ======================
# Test Milestone 1 Engine
# ======================
if __name__ == "__main__":
    test_queries = [
        "Show me balance of my savings account 12345",
        "Transfer 5000 to account 98765",
        "What is the interest rate for my fixed deposit?",
        "I want to open a current account"
    ]

    for query in test_queries:
        intent = intent_model.predict([query])[0]
        entities = extract_entities(query)
        print("\nQuery:", query)
        print("Predicted Intent:", intent)
        print("Extracted Entities / Slots:", entities)