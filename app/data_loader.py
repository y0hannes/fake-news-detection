import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)
    return df

def show_distribution(df):
    print(df['label'].value_counts())
