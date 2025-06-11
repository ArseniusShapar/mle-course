import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(**kwargs):
    ti = kwargs['ti']
    df = pd.DataFrame(ti.xcom_pull(key='cleaned_data', task_ids='clean_data'))
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    ti.xcom_push(key='train_data', value=train_df.to_dict())
    ti.xcom_push(key='test_data', value=test_df.to_dict())
