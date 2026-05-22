import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from src.exception.exception import CustomException
from src.logger.logger import logging
from src.utils.utils import load_object

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME          = "FitnessModel"


def get_mlflow_client():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


def get_available_versions():
    """Lấy danh sách tất cả version của FitnessModel."""
    try:
        client   = get_mlflow_client()
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        result   = []
        for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
            result.append({
                "version": v.version,
                "stage":   v.current_stage,   # None / Staging / Production / Archived
                "run_id":  v.run_id,
            })
        return result
    except Exception:
        return []


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features, model_version=None):
        try:
            preprocessor = load_object(os.path.join("artifact", "preprocessor.pkl"))

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            if model_version:
                # Load đúng version chỉ định (dùng khi A/B test)
                model_uri = f"models:/{MODEL_NAME}/{model_version}"
            else:
                model_uri = f"models:/{MODEL_NAME}/Production"

            logging.info(f"Loading model from MLflow: {model_uri}")
            model = mlflow.sklearn.load_model(model_uri)

            scaled_fea = preprocessor.transform(features)
            return model.predict(scaled_fea)

        except Exception as e:
            logging.warning(f"MLflow load thất bại, fallback sang local pkl: {e}")
            # Fallback: MLflow server không chạy → dùng file local
            model        = load_object(os.path.join("artifact", "model.pkl"))
            preprocessor = load_object(os.path.join("artifact", "preprocessor.pkl"))
            scaled_fea   = preprocessor.transform(features)
            return model.predict(scaled_fea)


class CustomData:
    def __init__(self,
                 tuoi:      int,
                 c_cao:     float,
                 c_nang:    float,
                 nhip_tim:  float,
                 huyet_ap:  float,
                 gio_ngu:   float,
                 dinh_duong:float,
                 hd:        float,
                 gt:        str,
                 hutthuoc:  str):

        self.tuoi       = tuoi
        self.c_cao      = c_cao
        self.c_nang     = c_nang
        self.nhip_tim   = nhip_tim
        self.huyet_ap   = huyet_ap
        self.gio_ngu    = gio_ngu
        self.dinh_duong = dinh_duong
        self.hd         = hd
        self.gt         = gt
        self.hutthuoc   = hutthuoc

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "age":               [self.tuoi],
                "height_cm":         [self.c_cao],
                "weight_kg":         [self.c_nang],
                "heart_rate":        [self.nhip_tim],
                "blood_pressure":    [self.huyet_ap],
                "sleep_hours":       [self.gio_ngu],
                "nutrition_quality": [self.dinh_duong],
                "activity_index":    [self.hd],
                "gender":            [self.gt],
                "smokes":            [self.hutthuoc],
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info(f"Dataframe Gathered. Columns: {list(df.columns)}")
            return df
        except Exception as e:
            logging.info("Exception Occured in prediction pipeline")
            raise CustomException(e, sys)