import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, make_scorer
from src.logger.logger import logging
from src.exception.exception import CustomException
import os
import sys
from dataclasses import dataclass, field
from src.utils.utils import save_object
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold


@dataclass
class ModelTrainerConfig:
    # Thư mục lưu từng model riêng biệt thay vì 1 file duy nhất
    model_dir: str = os.path.join("artifact", "models")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array) -> dict:
        """
        Train cả 3 model bằng GridSearchCV + StratifiedKFold.
        Trả về dict {model_name: best_estimator} để ModelEvaluation
        đánh giá trên test set rồi mới chọn winner.
        """
        try:
            x_train, y_train = train_array[:, :-1], train_array[:, -1]

            models = {
                "SVM": SVC(),
                "logistic_regression": LogisticRegression(),
                "random_forest": RandomForestClassifier(random_state=42),
            }

            param_grids = {
                "SVM": {
                    "C": [0.1, 1, 10],
                    "kernel": ["rbf", "linear"],
                },
                "logistic_regression": {
                    "C": [0.01, 0.1, 1, 10, 100],
                    "max_iter": [200],
                },
                "random_forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                },
            }

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = make_scorer(f1_score, pos_label=0)

            os.makedirs(self.model_trainer_config.model_dir, exist_ok=True)

            # ── Lưu tất cả best estimator sau GridSearch ──────────────────
            trained_models = {}

            for name, model in models.items():
                print(f"\n Tuning {name}...")
                gs = GridSearchCV(
                    model,
                    param_grids[name],
                    scoring=scoring,
                    cv=cv,
                    n_jobs=-1,
                    verbose=1,
                )
                gs.fit(x_train, y_train)

                print(f"   Best params   : {gs.best_params_}")
                print(f"   Best CV F1    : {gs.best_score_:.4f}")
                logging.info(f"{name} | best_params={gs.best_params_} | cv_f1={gs.best_score_:.4f}")

                # Lưu từng model để ModelEvaluation có thể load lại
                model_path = os.path.join(
                    self.model_trainer_config.model_dir, f"{name}.pkl"
                )
                save_object(file_path=model_path, obj=gs.best_estimator_)

                trained_models[name] = {
                    "estimator": gs.best_estimator_,
                    "cv_f1": gs.best_score_,
                    "best_params": gs.best_params_,
                    "model_path": model_path,
                }

            logging.info("Train xong ca 3 model, chuyen sang ModelEvaluation de chon winner")
            print("\n Train xong ca 3 model. Chuyen sang evaluation tren test set...")

            # Trả về dict để ModelEvaluation dùng — KHÔNG chọn winner ở đây
            return trained_models

        except Exception as e:
            logging.info("Loi o ModelTrainer")
            raise CustomException(e, sys)