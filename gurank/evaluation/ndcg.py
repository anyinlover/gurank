import numpy as np
from sklearn.metrics import ndcg_score
from transformers.trainer_utils import EvalPrediction

class NDCG:
    def __init__(self, k: int):
        self.k = k

    def __call__(self, pred: EvalPrediction):
        labels = pred.label_ids
        pred_exp = np.exp(pred.predictions)
        preds = (pred_exp / pred_exp.sum(axis=1, keepdims=True))[:,-1]
        ndcg = ndcg_score([labels], [preds], k = self.k)
        return {f"ndcg@{self.k}": ndcg}