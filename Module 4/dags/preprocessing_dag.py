from datetime import datetime

from airflow import DAG
from airflow.operators.python_operator import PythonOperator

from tasks.clean import clean_data
from tasks.load import load_data
from tasks.merge import merge_data
from tasks.save import save_data
from tasks.split import split_data
from tasks.transform import transform_data

with DAG(dag_id='adult_income_preprocessing',
         start_date=datetime(2025, 6, 9),
         schedule='@once',
         catchup=False) as dag:
    t1 = PythonOperator(
        task_id='load_data',
        python_callable=load_data
    )
    t2 = PythonOperator(
        task_id='merge_data',
        python_callable=merge_data
    )
    t3 = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data
    )
    t4 = PythonOperator(
        task_id='split_data',
        python_callable=split_data
    )
    t5 = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data
    )
    t6 = PythonOperator(
        task_id='save_data',
        python_callable=save_data
    )

    t1 >> t2 >> t3 >> t4 >> t5 >> t6
