import torch
import logging
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers.modeling_utils import PreTrainedModel
from typing import Tuple

logger = logging.getLogger(__name__)

class MonoBert(PreTrainedModel):
    def __init__(self, model_name):
        self.config = AutoConfig.from_pretrained(model_name, return_dict=False, num_labels=2)
        super().__init__(self.config)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config)
    
    def forward(self, input_ids, attention_mask, labels) -> Tuple[torch.Tensor]:
        return self.model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)