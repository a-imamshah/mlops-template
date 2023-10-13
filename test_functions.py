from sklearn.model_selection import train_test_split
import pandas as pd


def test_split_len():
    CSV_FILE = "dataset.csv"
    df = pd.read_csv(CSV_FILE)
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
    assert len(train_df) == 232
    assert len(valid_df) == 58
