import pandas as pd
import os

class DataIngestion:
    def __init__(self, raw_data_path="data/raw", processed_data_path="data/processed"):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path

    def ingest_data(self):
        fake_path = os.path.join(self.raw_data_path, "Fake.csv")
        true_path = os.path.join(self.raw_data_path, "True.csv")

        fake_df = pd.read_csv(
            fake_path,
            encoding="latin1",
            engine="python",
            on_bad_lines="skip"
        )

        true_df = pd.read_csv(
            true_path,
            encoding="latin1",
            engine="python",
            on_bad_lines="skip"
        )

        fake_df["label"] = 0   # Fake
        true_df["label"] = 1   # Real

        combined_df = pd.concat([fake_df, true_df], ignore_index=True)

        os.makedirs(self.processed_data_path, exist_ok=True)
        output_path = os.path.join(self.processed_data_path, "news_data.csv")

        combined_df.to_csv(output_path, index=False)

        print("✅ Data ingestion completed successfully")
        print(f"📁 Saved processed data to: {output_path}")

        return combined_df


if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.ingest_data()
