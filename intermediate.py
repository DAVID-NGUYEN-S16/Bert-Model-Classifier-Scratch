import torch 
import torch.nn as nn
import torch.nn.functional as F
from bert_attention import BertAttention
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.emb_size, config.intermediate_size)
        self.gelu = nn.GELU()
    def forward(self, att_out):
        x = self.dense(att_out)
        out = self.gelu(x)
        return out
        
class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.emb_size)
        self.dropout = nn.Dropout(config.dropout)
        self.LayerNorm = nn.LayerNorm(config.emb_size, eps=config.layer_norm_eps)
        
    def forward(self, intermediate_out, att_out):
        x = self.dense(intermediate_out)
        x = self.dropout(x)
        x = x + att_out
        out = self.LayerNorm(x)
        return out
class BertLayer(nn.Module):
    def __init__(self, config, layer_i):
        super().__init__()
        
        self.attention = BertAttention(config=config, layer_i=layer_i)
        self.intermediate = BertIntermediate(config=config)
        self.output = BertOutput(config=config)
    def forward(self, emb, att_mask):
        att_out = self.attention(emb, att_mask)
        intermediate_out = self.intermediate(att_out)
        out = self.output(intermediate_out, att_out)
        return out
        