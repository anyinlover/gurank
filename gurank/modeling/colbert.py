import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertPreTrainedModel, BertModel
from typing import Tuple

class ColBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, config.dim, bias=False)
        self.post_init()
    
    def forward(self,
                query_input_ids,
                query_attention_mask,
                docs_input_ids,
                docs_attention_mask,
                labels) -> Tuple[torch.Tensor]:
        query_outputs = self._forward(input_ids=query_input_ids,
                            attention_mask=query_attention_mask)

        docs_outputs = self._forward(input_ids=docs_input_ids,
                                attention_mask=docs_attention_mask)
        
        query_outputs.repeat_interleave(self.config.nways, dim=0).contiguous()

        outputs = docs_outputs @ query_outputs.permute(0, 2, 1)
        outputs = outputs.max(1).values
        logits = outputs.sum(-1)

        loss = None
        if labels:
            loss = nn.CrossEntropyLoss(logits.view(-1, self.config.nways), labels.view(-1))
        
        output = (logits,) + docs_outputs[:2]
        return ((loss,) + output) if loss else output


    def _forward(self, input_ids, attention_mask) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)[0]
        outputs = self.linear[outputs]
        outputs = F.normalize(outputs, dim=2)
        return outputs

        
