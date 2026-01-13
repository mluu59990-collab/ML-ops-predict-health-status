import os
import sys
import pandas as pd
from src.exception.exception import CustomException
from src.logger.logger import logging
from src.utils.utils import load_object


class PredictPipeline:
    def __init__(self):
        print("init.. the object")

    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifact","preprocessor.pkl")
            model_path=os.path.join("artifact","model.pkl")

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            scaled_fea=preprocessor.transform(features)
            pred=model.predict(scaled_fea)
            return pred

        except Exception as e:
            raise CustomException(e,sys)
class CustomData:
    def __init__(self,
                tuoi:int,
                c_cao:float,
                c_nang:float,
                nhip_tim:float,
                huyet_ap:float,
                gio_ngu:int,
                dinh_duong:int,
                hd:int,
                gt:str,
                hutthuoc:str):
        
        self.tuoi=tuoi
        self.c_cao=c_cao
        self.c_nang=c_nang
        self.nhip_tim=nhip_tim
        self.huyet_ap=huyet_ap
        self.gio_ngu=gio_ngu
        self.dinh_duong = dinh_duong
        self.hd = hd
        self.gt = gt
        self.hutthuoc = hutthuoc
            
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "age": [self.tuoi],
                "height_cm": [self.c_cao],
                "weight_kg": [self.c_nang],
                "heart_rate": [self.nhip_tim],
                "blood_pressure": [self.huyet_ap],
                "sleep_hours": [self.gio_ngu],
                "nutrition_quality": [self.dinh_duong],
                "activity_index": [self.hd],
                "gender": [self.gt],
                "smokes": [self.hutthuoc],
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info(f"Dataframe Gathered. Columns: {list(df.columns)}")
            return df
        except Exception as e:
            logging.info("Exception Occured in prediction pipeline")
            raise CustomException(e, sys)