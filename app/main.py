import argparse
import os
from sklearn.model_selection import train_test_split
from app.data_loader import load_dataset, show_distribution
from app.preprocess import vectorize_text
from app.model import build_model
from app.train import train_model
from app.evaluate import evaluate_model, plot_history
from app.utils import save_artifacts

def main(dataset_path):
    # 1. Load data
    print(f"--- Loading data from {dataset_path} ---")
    df = load_dataset(dataset_path)
    df = df.dropna(subset=['news', 'label'])
    show_distribution(df)

    # 2. Split data
    X = df['news']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Vectorize
    print("\n--- Vectorizing text ---")
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)

    # 4. Build and train
    print("\n--- Building and training model ---")
    model = build_model(X_train_vec.shape[1])
    history = train_model(model, X_train_vec, y_train)

    # 5. Evaluate
    print("\n--- Evaluating model ---")
    evaluate_model(model, X_test_vec, y_test)
    plot_history(history, save_path='app/training_history.png')

    # 6. Save artifacts
    print("\n--- Saving artifacts ---")
    save_artifacts(model, vectorizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake News Detection Pipeline")
    parser.add_argument("dataset", help="Path to the train_news.csv file")
    args = parser.parse_args()

    if os.path.exists(args.dataset):
        main(args.dataset)
    else:
        print(f"Error: Dataset not found at {args.dataset}")
