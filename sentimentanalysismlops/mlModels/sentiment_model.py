import torch.nn as nn
from transformers.models.bert.modeling_bert import BertModel
transformer_name = "bert-base-uncased"


class SentimentModel(nn.Module):
    def __init__(self, transformer_instance: BertModel, output_dim: int, freeze: bool):
        super().__init__()
        self.transformer = transformer_instance
        hidden_dim = transformer_instance.config.hidden_size
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.name = "Sentiment model"
        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def say_hello(self) -> str:
        return f"hello {self.name}"
