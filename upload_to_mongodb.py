# scripts/upload_to_mongodb.py
import os
import pandas as pd
from pymongo import MongoClient

def upload_fitness_data(csv_path: str = "/Users/Documents/ML_OPS/DAY11/artifact/fitness_dataset.csv"):
    client = MongoClient(os.environ.get("MONGO_URI", "mongodb://localhost:27017/"))
    db = client["fitness_db"]
    collection = db["raw_data"]

    collection.drop()  # Xóa data cũ nếu có

    data = pd.read_csv(csv_path)
    collection.insert_many(data.to_dict("records"))

    print(f"✅ Uploaded {len(data)} records to MongoDB")
    client.close()

if __name__ == "__main__":
    upload_fitness_data()