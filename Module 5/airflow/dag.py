import subprocess
from datetime import timedelta

import pandas as pd
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

from airflow import DAG
from deployment.app.utils import load_artifacts, preprocess, predict

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        dag_id="batch_pipeline_with_docker",
        default_args=default_args,
        description="Step-by-step batch pipeline with docker build",
        schedule_interval="0 3 * * *",
        start_date=days_ago(1),
        catchup=False,
) as dag:
    def read_input_data(**context):
        df = pd.read_csv("data/input/sample.csv")
        context["ti"].xcom_push(key="raw_df", value=df.to_json())


    def preprocess_step(**context):
        df_json = context["ti"].xcom_pull(key="raw_df", task_ids="read_input")
        df = pd.read_json(df_json)
        df = preprocess(df)
        context["ti"].xcom_push(key="preprocessed_df", value=df.to_json())


    def predict_step(**context):
        df_json = context["ti"].xcom_pull(key="preprocessed_df", task_ids="preprocess")
        df = pd.read_json(df_json)

        vectorizer, model = load_artifacts()
        y_pred = predict(df, vectorizer, model)

        context["ti"].xcom_push(key="y_pred", value=y_pred.tolist())
        context["ti"].xcom_push(key="df", value=df.to_json())


    def save_predictions(**context):
        df = pd.read_json(context["ti"].xcom_pull(key="df", task_ids="predict"))
        y_pred = context["ti"].xcom_pull(key="y_pred", task_ids="predict")
        df["pred"] = y_pred
        df.to_csv("data/output/predictions.csv", index=False)


    def build_docker_image():
        subprocess.run(["docker", "build", "-t", "module_5:latest", "."], check=True)


    t1 = PythonOperator(task_id="read_input", python_callable=read_input_data, provide_context=True)
    t2 = PythonOperator(task_id="preprocess", python_callable=preprocess_step, provide_context=True)
    t3 = PythonOperator(task_id="predict", python_callable=predict_step, provide_context=True)
    t4 = PythonOperator(task_id="save_output", python_callable=save_predictions, provide_context=True)
    t5 = PythonOperator(task_id="build_docker", python_callable=build_docker_image)

    t1 >> t2 >> t3 >> t4 >> t5
