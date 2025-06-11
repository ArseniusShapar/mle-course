import pandas as pd


def save_data(path='/opt/airflow/data/output', **kwargs):
    ti = kwargs['ti']
    train_df = pd.DataFrame(ti.xcom_pull(key='transformed_train_data', task_ids='transform_data'))
    test_df = pd.DataFrame(ti.xcom_pull(key='transformed_test_data', task_ids='transform_data'))

    train_df.to_csv(f"{path}/train.csv", index=False)
    test_df.to_csv(f"{path}/test.csv", index=False)
