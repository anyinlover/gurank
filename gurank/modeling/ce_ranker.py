import torch
from torch import nn
from tranformers import PreTrainedModel
from typing import Optional, Tuple

class CERanker(nn.Module):
    def __init__(
        self,
        hf_model: PreTrainedModel,
        docs_per_query: Optional[int] = None,
        loss_fct: nn.Module = nn.CrossEntropyLoss(),
    ):
        super().__init__()
        self.hf_model = hf_model
        self.docs_per_query = docs_per_query
        self.loss_fct = loss_fct
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        logits: torch.Tensor = self.hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).logits
        if labels is not None:
            group_logits = logits.view(-1, self.docs_per_query)
            group_labels = labels.view(-1, self.docs_per_query)
            loss = self.loss_fct(group_logits, group_labels).squeeze()
            return loss, logits
        
        return None, logits