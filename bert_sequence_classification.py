import torch 
import torch.nn as nn
import torch.nn.functional as F
from bert_model import *
class BertForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel(config=config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.emb_size, config.n_classes)
        
    def forward(self, input_ids, token_type_ids ,attention_mask = None):
        _, pooled_out = self.bert(input_ids, token_type_ids, attention_mask)
        
        pooled_out = self.dropout(pooled_out)
        logits = self.classifier(pooled_out)
        if self.config.return_pooler_output:
            return pooled_out, logits
        return logits