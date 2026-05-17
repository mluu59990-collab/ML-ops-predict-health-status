from datetime import datetime
from bson import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash


class UserModel:
    """Quản lý User trong MongoDB"""

    def __init__(self, db):
        self.col = db["users"]
        self.col.create_index("email", unique=True)
        self.col.create_index("username", unique=True)

    def create(self, username: str, email: str, password: str):
        doc = {
            "username": username,
            "email": email.lower().strip(),
            "password_hash": generate_password_hash(password),
            "created_at": datetime.utcnow(),
        }
        result = self.col.insert_one(doc)
        doc["_id"] = result.inserted_id
        return doc

    def find_by_email(self, email: str):
        return self.col.find_one({"email": email.lower().strip()})

    def find_by_id(self, user_id: str):
        try:
            return self.col.find_one({"_id": ObjectId(user_id)})
        except Exception:
            return None

    def verify_password(self, user: dict, password: str) -> bool:
        return check_password_hash(user["password_hash"], password)

    def email_exists(self, email: str) -> bool:
        return self.col.find_one({"email": email.lower().strip()}) is not None

    def username_exists(self, username: str) -> bool:
        return self.col.find_one({"username": username}) is not None


class HistoryModel:
    """Lưu lịch sử dự đoán theo từng user"""

    def __init__(self, db):
        self.col = db["prediction_history"]
        self.col.create_index("user_id")
        self.col.create_index("created_at")

    def save(self, user_id: str, input_data: dict, result: str, model_version: str = None):
        """
        result        : "Fit" hoặc "Not Fit"
        input_data    : dict các feature người dùng nhập
        model_version : version MLflow được dùng để dự đoán (None = latest)
        """
        doc = {
            "user_id":       str(user_id),
            "input_data":    input_data,
            "result":        result,
            "model_version": model_version or "latest",
            "created_at":    datetime.utcnow(),
        }
        inserted = self.col.insert_one(doc)
        doc["_id"] = str(inserted.inserted_id)
        return doc

    def get_by_user(self, user_id: str, limit: int = 50):
        cursor = (
            self.col.find({"user_id": str(user_id)})
            .sort("created_at", -1)
            .limit(limit)
        )
        records = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            records.append(doc)
        return records

    def delete(self, record_id: str, user_id: str) -> bool:
        res = self.col.delete_one(
            {"_id": ObjectId(record_id), "user_id": str(user_id)}
        )
        return res.deleted_count > 0

    def stats(self, user_id: str) -> dict:
        pipeline = [
            {"$match": {"user_id": str(user_id)}},
            {"$group": {"_id": "$result", "count": {"$sum": 1}}},
        ]
        out = {"total": 0, "fit": 0, "not_fit": 0}
        for item in self.col.aggregate(pipeline):
            out["total"] += item["count"]
            if item["_id"] == "Fit":
                out["fit"] = item["count"]
            elif item["_id"] == "Not Fit":
                out["not_fit"] = item["count"]
        return out