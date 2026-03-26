FROM python:3.10-slim-buster

USER root

RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /app
WORKDIR /app

COPY . /app/

RUN pip install --no-cache-dir -r requirements.txt

ENV AIRFLOW_HOME=/app/airflow
ENV AIRFLOW_CORE_DAGBAG_IMPORT_TIMEOUT=1000
ENV AIRFLOW_CORE_ENABLE_XCOM_PICKLING=True
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False

RUN chmod +x start.sh

CMD ["./start.sh"]
