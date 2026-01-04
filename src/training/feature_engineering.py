import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle


class FeatureEngineering:
    def __init__(self,
                 input_path="data/processed/clean_news_data.csv",
                 output_dir="data/processed"):
        self.input_path = input_path
        self.output_dir = output_dir

    def build_features(self):
        df = pd.read_csv(self.input_path, low_memory=False)

        if "clean_text" not in df.columns or "label" not in df.columns:
            raise ValueError("❌ Required columns not found (clean_text, label)")

        # 🔒 Handle NaN / empty text
        df["clean_text"] = df["clean_text"].fillna("")
        df = df[df["clean_text"].str.strip() != ""]

        X = df["clean_text"]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        os.makedirs(self.output_dir, exist_ok=True)

        with open(os.path.join(self.output_dir, "tfidf_vectorizer.pkl"), "wb") as f:
            pickle.dump(vectorizer, f)

        with open(os.path.join(self.output_dir, "X_train.pkl"), "wb") as f:
            pickle.dump(X_train_tfidf, f)

        with open(os.path.join(self.output_dir, "X_test.pkl"), "wb") as f:
            pickle.dump(X_test_tfidf, f)

        with open(os.path.join(self.output_dir, "y_train.pkl"), "wb") as f:
            pickle.dump(y_train, f)

        with open(os.path.join(self.output_dir, "y_test.pkl"), "wb") as f:
            pickle.dump(y_test, f)

        print("✅ Feature engineering completed")
        print("📦 TF-IDF features and labels saved")

        return X_train_tfidf, X_test_tfidf, y_train, y_test


if __name__ == "__main__":
    fe = FeatureEngineering()
    fe.build_features()
