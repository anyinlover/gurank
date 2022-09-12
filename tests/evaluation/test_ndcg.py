from gurank.evaluation.ndcg import NDCG
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import ndcg_score

import numpy as np
import unittest

class NDCGTest(unittest.TestCase):

    def test_ndcg(self):
        """Test ndcg computed correct"""
        k = 10
        ndcg = NDCG(k)
        y_true = np.random.randint(0, 2, 10)
        y_pred = np.random.random((10,2))
        eval_prediction = EvalPrediction(y_pred, y_true)
        ndcg_compute = ndcg(eval_prediction)
        pred = (np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True))[:,-1]
        sklearn_ndcg = ndcg_score([y_true], [pred], k=k)
        self.assertAlmostEqual(ndcg_compute[f"ndcg@{k}"], sklearn_ndcg)


if __name__ == "__main__":
    unittest.main()