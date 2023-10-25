from bert_embedding import *
from pooler import *
from bert_encoder import *

class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbedding(config=config)
        self.encoder = BertEncoder(config=config)
        self.pooler = BertPooler(config=config)
        
    def forward(self, input_ids, token_type_ids, att_mask):
        emb = self.embeddings(input_ids, token_type_ids)
        out = self.encoder(emb, att_mask)
        pooled_out = self.pooler(out)
        return out, pooled_out