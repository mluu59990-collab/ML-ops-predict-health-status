#!/bin/sh
set -e

export AIRFLOW_HOME=/app/airflow
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////app/airflow/airflow.db
export AIRFLOW__CORE__LOAD_EXAMPLES=False

airflow db init || airflow db migrate

airflow users create \
  -u admin \
  -p admin \
  -f Minh \
  -l Luu \
  -r Admin \
  -e mluu59990@gmail.com || true

airflow scheduler &

exec airflow webserver -p 8080
