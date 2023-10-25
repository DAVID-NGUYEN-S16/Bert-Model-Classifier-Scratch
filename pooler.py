import torch
import torch.nn as nn

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.emb_size, config.emb_size)
        self.tanh= nn.Tanh()
        
    def forward(self, encoder_out):
        pool_first_token = encoder_out[:, 0]
        out= self.dense(pool_first_token)
        out = self.tanh(out)
        return out