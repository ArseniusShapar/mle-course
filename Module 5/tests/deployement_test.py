import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

from deployment.app.utils import preprocess, load_artifacts, predict


class TestDeployment(unittest.TestCase):

    @patch("deployment.app.utils.joblib.load")
    def test_loading_artifacts(self, mock_load):
        mock_vectorizer = MagicMock()
        mock_model = MagicMock()
        mock_load.side_effect = [mock_vectorizer, mock_model]

        vectorizer, model = load_artifacts()

        self.assertEqual(vectorizer, mock_vectorizer)
        self.assertEqual(model, mock_model)
        self.assertEqual(mock_load.call_count, 2)

    def test_preprocess(self):
        df = pd.DataFrame({
            'text': [' Hello, world!   ', 'Text... with $$$ symbols!!!', '\nNew\nLine\tTest ']
        })
        expected = ['Hello world', 'Text with symbols', 'New Line Test']

        df_processed = preprocess(df)

        self.assertIn('prepocessed_text', df_processed.columns)
        self.assertEqual(df_processed['prepocessed_text'].tolist(), expected)

    def test_predict(self):
        mock_vectorizer = MagicMock()
        mock_model = MagicMock()

        mock_vectorizer.transform.return_value = "fake_vector"
        mock_model.predict.return_value = [0]

        df = pd.DataFrame({'text': ['example text']})

        preds = predict(df, mock_vectorizer, mock_model)

        mock_vectorizer.transform.assert_called_once_with(df['text'])
        mock_model.predict.assert_called_once_with("fake_vector")
        self.assertEqual(preds, [0])


if __name__ == '__main__':
    unittest.main()
