import os
import subprocess
from datetime import timedelta

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from utils import load_artifacts, preprocess, make_predict

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(days=1),
}

with DAG(
        dag_id='batch_pipeline',
        # default_args=default_args,
        start_date=days_ago(1),
        schedule='@once',
        catchup=False,
) as dag:
    def read_input(path='opt/airflow/data/input/sample.csv', **kwargs):
        df = pd.read_csv(path)
        kwargs['ti'].xcom_push(key='raw_df', value=df.to_json())


    def preprocess_step(**kwargs):
        df_json = kwargs['ti'].xcom_pull(key='raw_df', task_ids='read_input')
        df = pd.read_json(df_json)
        df = preprocess(df)
        kwargs['ti'].xcom_push(key='preprocessed_df', value=df.to_json())


    def predict_step(**kwargs):
        df_json = kwargs['ti'].xcom_pull(key='preprocessed_df', task_ids='preprocess')
        df = pd.read_json(df_json)

        vectorizer, model = load_artifacts('opt/airflow/artifacts/count_vectorizer.joblib',
                                           'opt/airflow/artifacts/logistic_regression.joblib')
        y_pred = make_predict(df, vectorizer, model)

        kwargs['ti'].xcom_push(key='y_pred', value=y_pred)
        kwargs['ti'].xcom_push(key='df', value=df.to_json())


    def save_predictions(path='opt/airflow/data/output/preds.csv', **kwargs):
        df = pd.read_json(kwargs['ti'].xcom_pull(key='df', task_ids='predict'))
        y_pred = kwargs['ti'].xcom_pull(key='y_pred', task_ids='predict')
        df['pred'] = y_pred
        df.to_csv(path, index=False)


    def build_docker_image():
        os.chdir('/opt/airflow')
        subprocess.run(
            ['docker', 'build', '-f', 'online/Dockerfile.online', '-t', 'module-5-online:latest', '.'],
            check=True)


    t1 = PythonOperator(task_id='read_input', python_callable=read_input)
    t2 = PythonOperator(task_id='preprocess', python_callable=preprocess_step)
    t3 = PythonOperator(task_id='predict', python_callable=predict_step)
    t4 = PythonOperator(task_id='save_output', python_callable=save_predictions)
    t5 = PythonOperator(task_id='build_docker_image', python_callable=build_docker_image)

    t1 >> t2 >> t3 >> t4 >> t5
