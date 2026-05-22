import os
import sys
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.metrics import precision_score, recall_score, f1_score
from src.logger.logger import logging
from src.exception.exception import CustomException
from src.utils.utils import save_object

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
IMPROVE_THRESHOLD   = 0.01   # model mới phải tốt hơn ít nhất 1%


class ModelEvaluationConfig:
    best_model_path: str = os.path.join("artifact", "model.pkl")


class ModelEvaluation:
    def __init__(self):
        self.eval_config = ModelEvaluationConfig()
        logging.info("ModelEvaluation started")

    def eval_metrics(self, actual, pred):
        precision = precision_score(actual, pred, pos_label=0)
        recall    = recall_score(actual, pred, pos_label=0)
        f1        = f1_score(actual, pred, pos_label=0)
        return precision, recall, f1

    def _compare_with_production(self, client, new_version: str, new_f1: float):
        """
        So sánh model mới với Production hiện tại.
        - Tốt hơn THRESHOLD → promote lên Staging (chờ approve)
        - Không tốt hơn     → Archive
        """
        prod_versions = client.get_latest_versions(
            "FitnessModel", stages=["Production"]
        )

        if not prod_versions:
            # Chưa có Production → promote thẳng lên Staging
            client.transition_model_version_stage(
                name             = "FitnessModel",
                version          = new_version,
                stage            = "Staging",
                archive_existing_versions=False,
            )
            print(f"  Chưa có Production — promote version {new_version} lên Staging")
            print(f"   Kiểm tra xong chạy: promote_to_production({new_version})")
            return

        # Lấy F1 của model đang Production
        current_prod   = prod_versions[0]
        current_run    = client.get_run(current_prod.run_id)
        current_f1     = float(current_run.data.metrics.get("test_f1", 0))
        current_ver    = current_prod.version

        print(f"\nSo sánh với Production hiện tại:")
        print(f"   Production v{current_ver} : F1 = {current_f1:.4f}")
        print(f"   Model mới   v{new_version} : F1 = {new_f1:.4f}")
        print(f"   Threshold               : +{IMPROVE_THRESHOLD:.2f}")

        if new_f1 > current_f1 + IMPROVE_THRESHOLD:
            # Model mới tốt hơn → lên Staging chờ approve
            client.transition_model_version_stage(
                name             = "FitnessModel",
                version          = new_version,
                stage            = "Staging",
                archive_existing_versions=False,
            )
            print(f" Model mới tốt hơn {(new_f1 - current_f1):.4f} → lên Staging (version {new_version})")
            print(f"   Approve xong chạy: promote_to_production({new_version})")
        else:
            # Không tốt hơn → Archive, giữ Production cũ
            client.transition_model_version_stage(
                name             = "FitnessModel",
                version          = new_version,
                stage            = "Archived",
                archive_existing_versions=False,
            )
            print(f" Model mới không cải thiện đủ → Archive version {new_version}")
            print(f"   Production v{current_ver} vẫn được giữ nguyên")

    def initiate_model_evaluation(self, trained_models: dict, test_array):
        try:
            x_test, y_test = test_array[:, :-1], test_array[:, -1]

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            client = MlflowClient(MLFLOW_TRACKING_URI)

            best_f1         = -1
            best_model      = None
            best_model_name = None
            results         = {}

            # ── Đánh giá từng model ───────────────────────────────────────────
            for name, info in trained_models.items():
                model  = info["estimator"]
                cv_f1  = info["cv_f1"]
                params = info["best_params"]

                with mlflow.start_run(run_name=name) as run:
                    mlflow.log_params(params)
                    mlflow.log_param("model_name", name)
                    mlflow.log_param("cv_f1", round(cv_f1, 4))

                    prediction = model.predict(x_test)
                    precision, recall, f1 = self.eval_metrics(y_test, prediction)

                    print(f"\n=== {name} ===")
                    print(f"   CV F1 (train) : {cv_f1:.4f}")
                    print(f"   Precision     : {precision:.4f}")
                    print(f"   Recall        : {recall:.4f}")
                    print(f"   F1 (test)     : {f1:.4f}")

                    mlflow.log_metric("test_precision", precision)
                    mlflow.log_metric("test_recall",    recall)
                    mlflow.log_metric("test_f1",        f1)
                    mlflow.log_metric("cv_f1",          cv_f1)

                    # Log model lên MinIO qua MLflow (1 lần duy nhất)
                    mlflow.sklearn.log_model(model, artifact_path="model")

                    results[name] = {
                        "precision": precision,
                        "recall":    recall,
                        "f1":        f1,
                        "cv_f1":     cv_f1,
                        "run_id":    run.info.run_id,
                        "estimator": model,
                    }

                    if f1 > best_f1:
                        best_f1         = f1
                        best_model      = model
                        best_model_name = name

            # ── Bảng tổng hợp ────────────────────────────────────────────────
            print("\n" + "=" * 55)
            print("     KẾT QUẢ SO SÁNH TRÊN TEST SET")
            print("=" * 55)
            print(f"{'Model':<25} {'CV F1':>8} {'Test F1':>8} {'Precision':>10} {'Recall':>8}")
            print("-" * 55)
            for name, r in results.items():
                marker = "" if name == best_model_name else ""
                print(
                    f"{name:<25} {r['cv_f1']:>8.4f} {r['f1']:>8.4f} "
                    f"{r['precision']:>10.4f} {r['recall']:>8.4f}{marker}"
                )
            print("=" * 55)
            print(f"\nWinner: {best_model_name} | Test F1 = {best_f1:.4f}")

            # ── Bước 1: Lưu pkl local ─────────────────────────────────────────
            save_object(file_path=self.eval_config.best_model_path, obj=best_model)
            logging.info(f"Saved best model local: {self.eval_config.best_model_path}")

            # ── Bước 2: Register vào MLflow Registry ─────────────────────────
            if tracking_url_type_store != "file":
                best_run_id = results[best_model_name]["run_id"]
                model_uri   = f"runs:/{best_run_id}/model"

                mv = mlflow.register_model(
                    model_uri = model_uri,
                    name      = "FitnessModel",
                )
                logging.info(
                    f"Registered '{best_model_name}' as FitnessModel v{mv.version}"
                )
                print(f"\n Registered FitnessModel version {mv.version}")

                # ── Bước 3: So sánh với Production hiện tại ──────────────────
                self._compare_with_production(client, mv.version, best_f1)
            else:
                logging.info("File store — bỏ qua registry")

            return best_model_name, best_f1, results

        except Exception as e:
            raise CustomException(e, sys)


def promote_to_production(version: int):
    """
    Gọi thủ công sau khi QA team kiểm tra Staging xong.
    promote_to_production(2)
    """
    client = MlflowClient(MLFLOW_TRACKING_URI)
    client.transition_model_version_stage(
        name                     = "FitnessModel",
        version                  = str(version),
        stage                    = "Production",
        archive_existing_versions= True,  # tự động Archive version Production cũ
    )
    print(f" FitnessModel version {version} đã lên Production!")
    print(f"   Version Production cũ đã được Archive tự động")