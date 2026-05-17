import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, make_scorer, make_scorer
from src.logger.logger import logging
from src.exception.exception import CustomException
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from src.utils.utils import save_object, load_object
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array):  # ← thụt vào trong class
        try:
            x_train, y_train = train_array[:, :-1], train_array[:, -1]

            models = {
                'SVM': SVC(),
                'logistic_regression': LogisticRegression(),
                'random_forest': RandomForestClassifier(random_state=42),
            }

            param_grids = {
                'SVM': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                },
                'logistic_regression': {
                    'C': [0.01, 0.1, 1, 10,100],
                    'max_iter': [200],
                },
                'random_forest': {
                    'n_estimators': [100, 200,300],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                },
            }

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            best_score = -1
            best_model = None
            best_model_name = None
            scoring = make_scorer(f1_score, pos_label=0)
            for name, model in models.items():
                print(f"\n Tuning {name}...")
                gs = GridSearchCV(
                    model,
                    param_grids[name],
                    scoring=scoring,  # Sử dụng F1 macro để đánh giá toàn bộ lớp
                    cv=cv,
                    n_jobs=-1,
                    verbose=1,
                )
                gs.fit(x_train, y_train)

                print(f"   Best params : {gs.best_params_}")
                print(f"   Best F1 score : {gs.best_score_:.4f}")

                if gs.best_score_ > best_score:
                    best_score = gs.best_score_
                    best_model = gs.best_estimator_
                    best_model_name = name

            print(f"\nBest model: {best_model_name} | F1 Score(CV): {best_score:.4f}")
            logging.info(f"Best model: {best_model_name}, F1 Score: {best_score:.4f}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info('loi')
            raise CustomException(e, sys)