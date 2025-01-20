##@ All required imports are here in this cell
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        "The embedding is done in pytorch using just a simple nn.Embedding function"
        self.embed= nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embed(x) * torch.sqrt(self.d_model, dtype=torch.float32)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_length: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)
        
        ## Create a positional encoding matrix of size seq_length x d_model
        pe = torch.zeros(seq_length, d_model)
        ## Create a vector of shape (seq_length, 1)
        pos = torch.arange(0, seq_length).unsqueeze(1)  ## The numerator part 
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)) ## The denominator part
        
        ## Apply sine to even positions
        pe[:, 0::2] = torch.sin(pos * div)
        
        ## Apply cosine to odd positions
        pe[:, 1::2] = torch.cos(pos * div)
        
        ## Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)     
        
        ## Register the positional encoding as a buffer
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x= x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, epsilon: float = 10**-6):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(1))  # The scaler
        self.beta = nn.Parameter(torch.zeros(1))  # The shifter aka bias
        
    def forward(self, x):
        mean = x.mean(dim= -1, keepdim=True)
        std = x.std(dim= -1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # The first linear layer with W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # The second linear layer with W2 and b2
    
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model= d_model
        self.h= h
        assert d_model % h ==0, "d_model is not divisible by h"
        
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        
        attention_scores = query @ key.transpose(-2, -1) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        
        query = query.view(query.size[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.size[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.size[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        
        ## (Batch_size, h, seq_length, d_k) -> (Batch_size, seq_length, h, d_k) -> (Batch_size, seq_length, d_model)
        x= x.transpose(1, 2).contiguous().view(x.size(0), -1, self.h * self.d_k)  #@ self.h * self.d_k = d_model
        
        #@ Now finally we multiply this with the output weights:
        return self.w_o(x)       
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
class EncoderBlock (nn.Module):
    def __init__(self, self_attn: MultiHeadAttention, feed_forward: FeedForward, dropout: float):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self,x, mask): 
        ''' why apply mask here?:
        The mask is applied to the self attention mechanism to prevent the model from attending to the future tokens.'''
        x= self.residual_connection[0](x, lambda x: self.self_attn(x, x, x, mask))
        x= self.residual_connection[1](x, self.feed_forward)
        return x
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x= layer(x, mask)
        return self.norm(x)  #Since after every layer we have Add and Norm layer.. 🤷
class DecoderBlock(nn.Module):
    def __init__(self, self_attn: MultiHeadAttention, cross_attn: MultiHeadAttention, \
                 feed_forward: FeedForward, dropout: float):
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        
    def forward(self, x, enc_output, src_mask, trgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attn(x, x, x, trgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attn(x, enc_output, enc_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward)
        return x
''' Now coding the Decoder class '''

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, enc_output, src_mask, trgt_mask):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, trgt_mask)
        return self.norm(x)
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        ''' Also  the softmax is applied here in forward method'''
        return torch.log_softmax(self.projection(x), dim=-1)
class Transformers(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, \
                 trgt_embed: InputEmbeddings, src_pos: PositionalEncoding, trgt_pos:PositionalEncoding,\
                 projection: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trgt_embed = trgt_embed
        self.src_pos = src_pos
        self.trgt_pos = trgt_pos
        self.projection = projection
    
    def encode(self, src, src_mask):
        src= self.src_embed(src)
        src= self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, trgt, enc_output, src_mask, trgt_mask):
        trgt= self.trgt_embed(trgt)
        trgt= self.trgt_pos(trgt)
        return self.decoder(trgt, enc_output, src_mask, trgt_mask)

    def project(self, x):
        return self.projection(x)
def build_transformer(src_vocab_size: int, trgt_vocab_size: int, src_seq_len: int,\
                    trgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8,\
                    dropout: float = 0.1,d_ff: int = 2048)  -> Transformers:
    ## Create the embedding layers:
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    trgt_embed = InputEmbeddings(d_model, trgt_vocab_size)
    
    ## Positional  Encoding layers:
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    trgt_pos = PositionalEncoding(d_model, trgt_seq_len, dropout)
    
    ## Create the encoder and decoder blocks:
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttention(d_model, h, dropout)
        feed_forward= FeedForward(d_model, d_ff, dropout)
        encoder_block= EncoderBlock(encoder_self_attention, feed_forward, dropout)
        encoder_blocks.append(encoder_block)
    
    decoder_blocks= []
    for _ in range(N):
        decoder_self_attention = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention = MultiHeadAttention(d_model, h, dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention, decoder_cross_attention, feed_forward, dropout)
        decoder_blocks.append(decoder_block)
        
    ## Creting Encoder and Decoder:
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    ## Create the projection layer:
    projection = ProjectionLayer(d_model, trgt_vocab_size)
    
    ## Create the transformer model:
    transformer = Transformers(encoder, decoder, src_embed, trgt_embed, src_pos, trgt_pos, projection)
    
    ## Initialize the parameters:
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    '''
    The xavier_uniform_ function initializes the weights of the model. The weights are initialized using a uniform distribution
    '''
    return transformer

