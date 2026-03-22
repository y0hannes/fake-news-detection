import joblib
from tensorflow.keras.models import load_model

def save_artifacts(model, vectorizer, model_path='fake_news_model.keras', vec_path='vectorizer.joblib'):
    model.save(model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"Artifacts saved to {model_path} and {vec_path}")

def load_artifacts(model_path='fake_news_model.keras', vec_path='vectorizer.joblib'):
    model = load_model(model_path)
    vectorizer = joblib.load(vec_path)
    print("Artifacts loaded successfully")
    return model, vectorizer
