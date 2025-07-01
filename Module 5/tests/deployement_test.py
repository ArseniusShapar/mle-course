import os
import unittest

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from deployment.utils import preprocess, load_artifacts, make_predict


class TestDeployment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.chdir('../deployment')

    def test_loading_artifacts(self):
        vectorizer, model = load_artifacts()
        self.assertIsInstance(vectorizer, CountVectorizer)
        self.assertIsInstance(model, LogisticRegression)

    def test_preprocess(self):
        df = pd.DataFrame({
            'text': [' Hello, world!   ', 'Text... with $$$ symbols!!!', '\nNew\nLine\tTest ']
        })
        expected = ['Hello world', 'Text with symbols', 'New Line Test']

        df_processed = preprocess(df)

        self.assertIn('prepocessed_text', df_processed.columns)
        self.assertEqual(df_processed['prepocessed_text'].tolist(), expected)

    def test_predict(self):
        vectorizer, model = load_artifacts()

        df = pd.DataFrame({'text': ['example text']})
        pred = make_predict(df, vectorizer, model)[0]

        self.assertIn(pred, [0, 1])


if __name__ == '__main__':
    unittest.main()
