from __future__ import annotations

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import timedelta


def data_ingestion(**kwargs):
    import sys, os
    sys.path.insert(0, "/Users/Documents/ML_OPS/DAY11")
    os.chdir("/Users/Documents/ML_OPS/DAY11")
    from src.pipeline.training_pipeline import TrainingPipeline
    pipeline = TrainingPipeline()

    ti = kwargs["ti"]
    train_path, test_path = pipeline.data_ingestion()
    ti.xcom_push(
        key="data_ingestion_artifact",
        value={"train_data": train_path, "test_data": test_path}
    )


def data_trans(**kwargs):
    import sys, os, numpy as np
    sys.path.insert(0, "/Users/Documents/ML_OPS/DAY11")
    os.chdir("/Users/Documents/ML_OPS/DAY11")
    from src.pipeline.training_pipeline import TrainingPipeline
    pipeline = TrainingPipeline()

    ti = kwargs["ti"]
    artifact = ti.xcom_pull(task_ids="ingestion", key="data_ingestion_artifact")
    train_arr, test_arr = pipeline.data_trans(artifact["train_data"], artifact["test_data"])

    os.makedirs("/Users/Documents/ML_OPS/DAY11/artifact", exist_ok=True)
    np.save("/Users/Documents/ML_OPS/DAY11/artifact/train_arr.npy", train_arr)
    np.save("/Users/Documents/ML_OPS/DAY11/artifact/test_arr.npy", test_arr)

    ti.xcom_push(
        key="data_transformation_artifact",
        value={
            "train_arr_path": "/Users/Documents/ML_OPS/DAY11/artifact/train_arr.npy",
            "test_arr_path":  "/Users/Documents/ML_OPS/DAY11/artifact/test_arr.npy",
        }
    )


def model_train(**kwargs):
    import sys, os, numpy as np
    import json, pickle
    sys.path.insert(0, "/Users/Documents/ML_OPS/DAY11")
    os.chdir("/Users/Documents/ML_OPS/DAY11")
    from src.pipeline.training_pipeline import TrainingPipeline
    pipeline = TrainingPipeline()

    ti = kwargs["ti"]
    artifact = ti.xcom_pull(task_ids="trans", key="data_transformation_artifact")

    train_arr = np.load(artifact["train_arr_path"])
    trained_models = pipeline.model_train(train_arr)

    # trained_models là dict chứa object — lưu ra file, push path vào XCom
    models_path = "/Users/Documents/ML_OPS/DAY11/artifact/trained_models.pkl"
    import pickle
    with open(models_path, "wb") as f:
        pickle.dump(trained_models, f)

    ti.xcom_push(
        key="data_train_artifact",
        value={
            "trained_models_path": models_path,
            "test_arr_path": artifact["test_arr_path"],
        }
    )


def model_eval(**kwargs):
    import sys, os, numpy as np, pickle
    from airflow.models import DagModel
    from airflow.utils.session import create_session

    sys.path.insert(0, "/Users/Documents/ML_OPS/DAY11")
    os.chdir("/Users/Documents/ML_OPS/DAY11")
    from src.pipeline.training_pipeline import TrainingPipeline
    pipeline = TrainingPipeline()

    ti = kwargs["ti"]
    artifact = ti.xcom_pull(task_ids="model_train", key="data_train_artifact")

    if artifact is None:
        raise ValueError("XCom pull thất bại! artifact là None. Kiểm tra task model_train đã push đúng chưa.")

    # Load trained_models từ file
    with open(artifact["trained_models_path"], "rb") as f:
        trained_models = pickle.load(f)

    test_arr = np.load(artifact["test_arr_path"])
    should_stop = pipeline.model_eval(trained_models, test_arr)

    if should_stop:
        print("Early stopping! Pause DAG.")
        with create_session() as session:
            dag_model = session.query(DagModel).filter(
                DagModel.dag_id == "training_fitness"
            ).first()
            if dag_model:
                dag_model.is_paused = True
                session.commit()


with DAG(
    dag_id="training_fitness",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    schedule="0 */2 * * *",
    max_active_runs=1,
    catchup=False,
    default_args={
        "execution_timeout": timedelta(minutes=120)
    }
) as dag:

    ingestion = PythonOperator(task_id="ingestion",    python_callable=data_ingestion)
    trans      = PythonOperator(task_id="trans",        python_callable=data_trans)
    train      = PythonOperator(task_id="model_train",  python_callable=model_train)
    eval_task = PythonOperator(
    task_id="model_eval",
    python_callable=model_eval,
    execution_timeout=timedelta(hours=3),  # tăng lên
)

    ingestion >> trans >> train >> eval_task