import json
import re
import pandas as pd


def clean_text(text):

    # lowercase
    text = text.lower()

    # remove content inside brackets
    # example: (Vessel name : MSC ORION)
    text = re.sub(r"\(.*?\)", "", text)

    # remove voyage numbers
    # example: / 248E
    text = re.sub(r"/\s*\w+", "", text)

    # remove special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # remove extra spaces
    text = " ".join(text.split())

    return text


def load_dataset(filepath):

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    return df


def preprocess_dataframe(df):

    # remove nulls
    df = df.dropna()

    # clean text
    df["cleaned_text"] = df["externalStatus"].apply(clean_text)

    return df


if __name__ == "__main__":

    df = load_dataset("../data/dataset.json")

    df = preprocess_dataframe(df)

    print(df.head())