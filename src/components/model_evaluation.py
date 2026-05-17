import os, sys
from sklearn.metrics import precision_score, recall_score, f1_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from src.logger.logger import logging
from src.exception.exception import CustomException
from src.utils.utils import load_object

class ModelEvaluationConfig:
    def __init__(self):
        logging.info("evaluation started")

    def eval_metrics(self, actual, pred):
        precision = precision_score(actual, pred, pos_label=0)
        recall    = recall_score(actual, pred, pos_label=0)
        f1        = f1_score(actual, pred, pos_label=0)
        return precision, recall, f1

    def initiate_model_evaluation(self, test_array):
        try:
            x_test, y_test = test_array[:, :-1], test_array[:, -1]

            model = load_object(os.path.join("artifact", "model.pkl"))

            mlflow.set_tracking_uri("http://127.0.0.1:5000")

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            # → bây giờ scheme = "http", không còn là "file" nữa

            with mlflow.start_run():
                mlflow.log_params(model.get_params())

                prediction = model.predict(x_test)
                pre, recall, f1 = self.eval_metrics(y_test, prediction)

                print(f"Precision : {pre:.4f}")
                print(f"Recall    : {recall:.4f}")
                print(f"F1 Score  : {f1:.4f}")

                mlflow.log_metric("precision", pre)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)

                # ✅ Sửa lỗi 2: nhánh if bây giờ sẽ được thực thi
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(
                        model, "model",
                        registered_model_name="FitnessModel"  # ← model sẽ xuất hiện ở đây
                    )
                else:
                    mlflow.sklearn.log_model(model, "model")

        except Exception as e:
            raise CustomException(e, sys)