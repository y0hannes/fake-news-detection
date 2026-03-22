import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def vectorize_text(train_text, test_text):

    # Apply cleaning
    train_text = train_text.apply(clean_text)
    test_text = test_text.apply(clean_text)

    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=10000,
        ngram_range=(1, 2)
    )

    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)

    return X_train, X_test, vectorizer
