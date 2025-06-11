import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


def transform_data(**kwargs):
    ti = kwargs['ti']
    train_df = pd.DataFrame(ti.xcom_pull(key='train_data', task_ids='split_data'))
    test_df = pd.DataFrame(ti.xcom_pull(key='test_data', task_ids='split_data'))

    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender',
                        'native-country']
    numeric_cols = ["age", "educational-num", "capital-gain", "capital-loss", "hours-per-week"]

    processor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False), categorical_cols),
            ('num', MinMaxScaler(), numeric_cols)
        ],
        remainder='passthrough'
    )
    processor.set_output(transform='pandas')

    train_df = processor.fit_transform(train_df)
    test_df = processor.transform(test_df)

    ti.xcom_push(key="transformed_train_data", value=train_df.to_dict())
    ti.xcom_push(key="transformed_test_data", value=test_df.to_dict())
