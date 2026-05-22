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

    def _get_production_f1(self, client) -> tuple:
        """Lấy F1 và version của model đang Production. Trả về (f1, version) hoặc (None, None)."""
        prod_versions = client.get_latest_versions("FitnessModel", stages=["Production"])
        if not prod_versions:
            return None, None
        current_prod = prod_versions[0]
        current_run  = client.get_run(current_prod.run_id)
        current_f1   = float(current_run.data.metrics.get("test_f1", 0))
        return current_f1, current_prod.version

    def _register_and_promote(self, client, run_id: str, new_f1: float, prod_ver, prod_f1):
        """Register model mới và tự động promote thẳng lên Production."""
        model_uri = f"runs:/{run_id}/model"

        mv = mlflow.register_model(model_uri=model_uri, name="FitnessModel")
        logging.info(f"Registered FitnessModel v{mv.version}")
        print(f"\n✅ Registered FitnessModel version {mv.version}")

        # Promote thẳng lên Production, tự động Archive version cũ
        client.transition_model_version_stage(
            name                     = "FitnessModel",
            version                  = mv.version,
            stage                    = "Production",
            archive_existing_versions= True,
        )

        if prod_f1 is None:
            print(f" Version {mv.version} → Production (lần đầu tiên)")
        else:
            print(f" Version {mv.version} → Production (F1: {prod_f1:.4f} → {new_f1:.4f}, cải thiện +{new_f1 - prod_f1:.4f})")
            print(f"   Version {prod_ver} đã được Archive tự động")

        return mv.version

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
                    f"{r['precision']:>10.4f} {r['recall']:>8.4f} {marker}"
                )
            print("=" * 55)
            print(f"\nWinner: {best_model_name} | Test F1 = {best_f1:.4f}")

            # ── Lưu pkl local ─────────────────────────────────────────────────
            save_object(file_path=self.eval_config.best_model_path, obj=best_model)
            logging.info(f"Saved best model local: {self.eval_config.best_model_path}")

            # ── So sánh với Production TRƯỚC khi register ────────────────────
            if tracking_url_type_store != "file":
                prod_f1, prod_ver = self._get_production_f1(client)

                print(f"\n{'=' * 55}")
                if prod_f1 is None:
                    # Lần đầu tiên → register và promote thẳng lên Production
                    print("  Chưa có Production → register và promote lần đầu")
                    self._register_and_promote(
                        client, results[best_model_name]["run_id"],
                        best_f1, prod_ver, prod_f1
                    )
                elif best_f1 > prod_f1 + IMPROVE_THRESHOLD:
                    # Model mới tốt hơn → register và promote lên Production
                    print(f"So sánh với Production v{prod_ver}:")
                    print(f"   Production F1 : {prod_f1:.4f}")
                    print(f"   Model mới F1  : {best_f1:.4f}  (+{best_f1 - prod_f1:.4f} > threshold {IMPROVE_THRESHOLD})")
                    self._register_and_promote(
                        client, results[best_model_name]["run_id"],
                        best_f1, prod_ver, prod_f1
                    )
                else:
                    # Không tốt hơn → bỏ qua hoàn toàn
                    print(f"So sánh với Production v{prod_ver}:")
                    print(f"   Production F1 : {prod_f1:.4f}")
                    print(f"   Model mới F1  : {best_f1:.4f}")
                    print(f" Không cải thiện đủ {IMPROVE_THRESHOLD:.0%} → bỏ qua, giữ Production v{prod_ver}")
                    logging.info(f"Skipped: new_f1={best_f1:.4f} <= prod_f1={prod_f1:.4f} + {IMPROVE_THRESHOLD}")
            else:
                logging.info("File store — bỏ qua registry")

            return best_model_name, best_f1, results

        except Exception as e:
            raise CustomException(e, sys)


def promote_to_production(version: int):
    """Gọi thủ công nếu cần promote 1 version cụ thể lên Production."""
    client = MlflowClient(MLFLOW_TRACKING_URI)
    client.transition_model_version_stage(
        name                     = "FitnessModel",
        version                  = str(version),
        stage                    = "Production",
        archive_existing_versions= True,
    )
    print(f"✅ FitnessModel version {version} đã lên Production!")
    print(f"   Version Production cũ đã được Archive tự động")