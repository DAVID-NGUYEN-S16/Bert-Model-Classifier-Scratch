import torch.nn as nn
import torch 
class BertEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.emb_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_seq_length, config.emb_size)
        self.token_type_embeddings = nn.Embedding(2, config.emb_size)
        self.LayerNorm = nn.LayerNorm(config.emb_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
        self.register_buffer("position_ids", torch.arange(config.max_seq_length).expand((1, -1)) )
        
        
    def forward(self, input_ids, token_type_ids):
        word_emb = self.word_embeddings(input_ids)
        pos_emb = self.position_embeddings(self.position_ids)
        type_emb = self.token_type_embeddings(token_type_ids)
        
        emb = word_emb + pos_emb + type_emb
        emb = self.LayerNorm(emb)
        emb = self.dropout(emb)
        
        return emb
        