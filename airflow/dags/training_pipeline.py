from __future__ import annotations
import json
from textwrap import dedent
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from src.pipeline.training_pipeline import TrainingPipeline
training_pipeline = TrainingPipeline()

with DAG(
    dag_id="training_fitness",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    schedule="@weekly",
    catchup=False,
) as dag:

    def data_ingestion(**kwargs):
        ti = kwargs["ti"]
        train_path, test_path = training_pipeline.data_ingestion()
        ti.xcom_push(
            key="data_ingestion_artifact",
            value={"train_data": train_path, "test_data": test_path}
        )

    def data_trans(**kwargs):
        ti = kwargs["ti"]
        artifact = ti.xcom_pull(
            task_ids="ingestion",
            key="data_ingestion_artifact"
        )
        train_arr, test_arr = training_pipeline.data_trans(
            artifact["train_data"],
            artifact["test_data"]
        )
        ti.xcom_push(
            key="data_transformation_artifact",
            value={
                "train_arr": train_arr.tolist(),
                "test_arr": test_arr.tolist()
            }
        )

    def model_train(**kwargs):
        import numpy as np
        ti = kwargs["ti"]
        artifact = ti.xcom_pull(
            task_ids="trans",
            key="data_transformation_artifact"
        )
        training_pipeline.model_train(
            np.array(artifact["train_arr"]),
            np.array(artifact["test_arr"])
        )

    ingestion = PythonOperator(task_id="ingestion", python_callable=data_ingestion)
    trans = PythonOperator(task_id="trans", python_callable=data_trans)
    train = PythonOperator(task_id="model_train", python_callable=model_train)

    ingestion >> trans >> train
