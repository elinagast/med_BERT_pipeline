from sklearn.model_selection import train_test_split
import pandas as pd

def clean_data(data: pd.DataFrame):
    if 'Unnamed: 1' in data.columns:
        return data.drop('Unnamed: 1', axis=1).dropna(axis=0)
    return data.dropna(axis=0)

def extract_text(data: pd.DataFrame, texts: list) -> list:
    data['texts'] = data[texts].agg('\n'.join, axis=1)
    return data['texts'].tolist()

def split_data(texts, labels):
    return train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

def extract_labels(data: pd.DataFrame, text: list):
    text_columns = text + ['texts']  # Create a new list instead of modifying the original
    return data.drop(text_columns, axis=1).values.tolist()