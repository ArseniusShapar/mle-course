import pandas as pd


def load_data(path='/opt/airflow/data/input/adult.csv', **kwargs):
    df = pd.read_csv(path)
    kwargs['ti'].xcom_push(key='raw_data', value=df.to_dict())
