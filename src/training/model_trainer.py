import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score


class ModelTrainer:
    def __init__(self, data_dir="data/processed"):
        self.data_dir = data_dir

    def load_data(self):
        with open(os.path.join(self.data_dir, "X_train.pkl"), "rb") as f:
            X_train = pickle.load(f)

        with open(os.path.join(self.data_dir, "X_test.pkl"), "rb") as f:
            X_test = pickle.load(f)

        with open(os.path.join(self.data_dir, "y_train.pkl"), "rb") as f:
            y_train = pickle.load(f)

        with open(os.path.join(self.data_dir, "y_test.pkl"), "rb") as f:
            y_test = pickle.load(f)

        return X_train, X_test, y_train, y_test

    def train_and_evaluate(self):
        X_train, X_test, y_train, y_test = self.load_data()

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Naive Bayes": MultinomialNB()
        }

        results = {}

        for name, model in models.items():
            print(f"\n🚀 Training {name}...")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            results[name] = {
                "accuracy": accuracy,
                "f1_score": f1
            }

            # Save model
            model_path = os.path.join(
                self.data_dir,
                f"{name.replace(' ', '_').lower()}_model.pkl"
            )

            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            print(f"✅ {name} Results:")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   F1 Score: {f1:.4f}")

        return results


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_and_evaluate()
