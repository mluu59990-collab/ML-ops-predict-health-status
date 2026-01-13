import pandas as pd
import numpy as np
from src.logger.logger import logging
from src.exception.exception import CustomException
import os
import sys
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder,FunctionTransformer,Normalizer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline 
from src.utils.utils import save_object
from dataclasses import dataclass
from pathlib import Path
@dataclass
class DataTransformationConfig:
    preprocessor_obj_filepath = os.path.join("artifact","preprocessor.pkl")
class DataTransformation:
    def __init__(self):
        self.data_transform_config = DataTransformationConfig()
    def get_data_transformation(self):
        try:
            categorical_cols = ["smokes", "gender"]
            numerical_cols = [
                "age","height_cm","weight_kg","heart_rate","blood_pressure",
                "sleep_hours","nutrition_quality","activity_index"
            ]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_cols),
                    ("cat", cat_pipeline, categorical_cols),
                ],
                remainder="drop"
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) 
    def initialize_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Doc train,test hoan tat")
            logging.info(f"Train data frame head:\n{train_df.head().to_string()}")
            logging.info(f"Test data frame head:\n{test_df.head().to_string()}")
            preprocessing_obj = self.get_data_transformation()
            target_column_name = 'is_fit'
            drop_columns = [target_column_name]
            input_feature_train_df = train_df.drop(columns=drop_columns)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=drop_columns)
            target_feature_test_df = test_df[target_column_name]
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info("Ap dung preprocessing cho train,test data")
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            save_object(
                file_path = self.data_transform_config.preprocessor_obj_filepath,
                obj = preprocessing_obj
            )
            logging.info("preprocessing pickle file saved")
            return (
                train_arr,
                test_arr
            )
        except Exception as e:
            logging.info("exception")
            raise CustomException(e,sys)


