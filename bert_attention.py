import torch
import torch.nn as nn
import torch.nn.functional as F
class BertSelfAttention(nn.Module):
    def __init__(self, config, layer_i):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads[layer_i]
        self.head_size = config.emb_size // self.n_heads
        self.query = nn.Linear(config.emb_size, config.emb_size)
        self.key = nn.Linear(config.emb_size, config.emb_size)
        self.value = nn.Linear(config.emb_size, config.emb_size)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, emb, att_mask):
        B, T, C = emb.shape # batch size, sequence lenght, embedding size
        '''
        Instead of performing a single attention function with d_model-dimensional keys, values, and queries, we found it beneficial to linearly project the queries, keys, and values h times with different, learned linear projections to d_k, d_k, and d_v dimensions, respectively. On each of these projected versions of queries, keys, and values, we then perform the attention function in parallel, yielding d_v-dimensional.
        

        
        '''
        q = self.query(emb).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        k = self.query(emb).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = self.query(emb).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        '''
        k.transpose(-2, -1):   so transpose had  3 part is batch size, keys_dim, num_keys
        
        with (-2, -1) is swap and transpose in  keys_dim, num_keys
        
        `
        k = torch.tensor([
                            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                            [[10, 11, 12], [13, 14, 15], [16, 17, 18]]
                        ])
        k_transposed = k.transpose(1, 2)

        Output:
                            tensor([[[ 1,  4,  7],
                            [ 2,  5,  8],
                            [ 3,  6,  9]],

                            [[10, 13, 16],
                            [11, 14, 17],
                            [12, 15, 18]]])
        `
        self.head_size**-0.5: scare value by divide to sqrt(dim)
        '''
        weights = q @ k.transpose(-2, -1) * self.head_size**-0.5
        
        # set the pad tokens to -inf se that they equal zero after softmax
        if att_mask !=None:
            '''
            unsqueeze will add dimensional of row example with 1D of (N, ) to 2D of (N, 1)
            
            repeat is dulicated current matrix 
            1, 1, 1      1, 1, 1    1, 1, 1   1, 1, 1
            1, 1, 1 -->{ 1, 1, 1 } {1, 1, 1} {1, 1, 1}
            1, 1, 1      1, 1, 1    1, 1, 1   1, 1, 1
            '''
            att_mask = (att_mask > 0 ).unsqueeze(1).repeat(1, att_mask.size(1), 1).unsqueeze(1)
            weights = weights.masked_fill(att_mask == 0, float('-inf'))
            
        '''
        dim = -1 sure that softmax apply compute to each row (item in batch)
        
        '''
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        
        emb_rich = weights @ v
        emb_rich = emb_rich.transpose(1, 2).contiguous().view(B, T, C)
        return emb_rich
    
        
        
        
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.emb_size, config.emb_size)
        self.dropout = nn.Dropout(config.dropout)
        self.LayerNorm = nn.LayerNorm(config.emb_size, eps=config.layer_norm_eps)
        
    def forward(self, emb_rich, emb):
        x = self.dense(emb_rich)
        x = self.dropout(x)
        x = x + emb
        out = self.LayerNorm(x)
        return out
    
class BertAttention(nn.Module):
    def __init__(self, config, layer_i):
        super().__init__()
        self.self = BertSelfAttention(config=config, layer_i=layer_i)
        self.output= BertSelfOutput(config=config)
        
    def forward(self, emb, att_mask):
        emb_rich = self.self(emb, att_mask)
        out = self.output(emb_rich, emb)
        return out