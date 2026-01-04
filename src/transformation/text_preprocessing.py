import pandas as pd
import os
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


class TextPreprocessor:
    def __init__(self, input_path="data/processed/news_data.csv",
                 output_path="data/processed/clean_news_data.csv"):
        self.input_path = input_path
        self.output_path = output_path
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^a-z\s]", "", text)

        words = text.split()
        words = [
            self.lemmatizer.lemmatize(word)
            for word in words
            if word not in self.stop_words
        ]

        return " ".join(words)

    def transform(self):
        df = pd.read_csv(self.input_path, low_memory=False)

        print("📌 Available columns:", df.columns.tolist())

        if "text" in df.columns:
            text_column = "text"
        elif "title" in df.columns:
            text_column = "title"
        else:
            raise ValueError("❌ No valid text column found")

        print(f"✅ Using '{text_column}' column for NLP processing")

        print("🔄 Cleaning text data...")

        df["clean_text"] = df[text_column].astype(str).apply(self.clean_text)

        df.to_csv(self.output_path, index=False)

        print("✅ Text preprocessing completed")
        print(f"📁 Clean data saved to: {self.output_path}")

        return df


if __name__ == "__main__":
    processor = TextPreprocessor()
    processor.transform()
