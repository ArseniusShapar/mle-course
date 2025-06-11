import pandas as pd


def merge_data(**kwargs):
    ti = kwargs['ti']
    df = pd.DataFrame(ti.xcom_pull(key='raw_data', task_ids='load_data'))
    work_data = df[['workclass', 'fnlwgt', 'occupation', 'capital-gain', 'capital-loss', 'hours-per-week']]
    df = pd.concat([df, work_data], axis=1)
    ti.xcom_push(key='merged_data', value=df.to_dict())
