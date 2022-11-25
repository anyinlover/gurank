import torch
import logging
from transformers import BertForSequenceClassification, BertPreTrainedModel
from typing import Tuple

logger = logging.getLogger(__name__)

class MonoBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = BertForSequenceClassification(config)
        self.post_init()
    
    def forward(self, input_ids, attention_mask, labels) -> Tuple[torch.Tensor]:
        return self.model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)