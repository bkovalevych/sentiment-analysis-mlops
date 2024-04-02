import torch.nn as nn
import transformers
from torch import Tensor
transformer_name = "bert-base-uncased"


class SentimentModel(nn.Module):
    def __init__(self, output_dim=6):
        super().__init__()

        self.transformer = transformers.AutoModel.from_pretrained(transformer_name)
        hidden_dim = self.transformer.config.hidden_size
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids: Tensor, attention_mask: Tensor, token_type_ids=None) -> Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        prediction = self.classifier(pooled_output)

        return prediction
