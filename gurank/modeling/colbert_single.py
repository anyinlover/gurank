import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Tuple

class ColBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, config.dim, bias=False)
        self.post_init()
    
    def forward(self,
                input_ids,
                attention_mask) -> Tuple[torch.Tensor]:
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)[0]
        outputs = self.linear(outputs)
        outputs = F.normalize(outputs, dim=2)
        query_outputs = outputs[-1:]
        docs_outputs = outputs[:-1]

        outputs = docs_outputs @ query_outputs.permute(0, 2, 1)
        outputs = outputs.max(1).values
        logits = outputs.sum(-1)

        return SequenceClassifierOutput(logits=logits)
