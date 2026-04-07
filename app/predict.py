from app.preprocess import clean_text

def predict_news(model, vectorizer, text):

    cleaned_text = clean_text(text)
    text_vector = vectorizer.transform([cleaned_text])

    prediction = model.predict(text_vector, verbose=0)[0][0]

    if prediction > 0.5:
        return "Fake News"
    else:
        return "Real News"
