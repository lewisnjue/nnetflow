# %%
import os 
import numpy as np 
import sys 


# %%
# Get parent directory path
parent_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath('.'))), 'nnetflow')
# Add parent directory to Python path if not already there
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# %%
parent_dir

# %%
print(sys.path)

# %%
from nnetflow.engine import Tensor 
from nnetflow import layers

# %%
GPT_CONFIG_124M = {
"vocab_size": 50257,
"context_length": 1024,
"emb_dim": 768,
"n_heads": 12,
"n_layers": 12,
"drop_rate": 0.1,
"qkv_bias": False
}

# %%


# %%


# %%
class FeedForward:
    def __init__(self,cfg:dict): 
        super().__init__() 
        self.layers = [
            layers.Linear(cfg['emb_dim'],4*cfg['emb_dim']),
            layers.Linear(4 * cfg['emb_dim'], cfg['emb_dim'])
        ]
    def __call__(self,x): 
        return self.layers[1](self.layers[0](x).gelu())
    def parameters(self):
        parameters = [] 
        parameters.append(self.layers[0].parameters())
        parameters.append(self.layers[-1].parameters())
        return parameters

# %%
class MultiHeadAttention:
    def __init__(self,d_in,d_out,context_length,num_heads,dropout,qkv_bias=False):
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads" 
        self.d_out = d_out
        self.num_heads = num_heads 
        self.head_dim = d_out // num_heads 
        self.W_query = layers.Linear(d_in,d_out,bias=qkv_bias) 
        self.W_key = layers.Linear(d_in,d_out,bias=qkv_bias) 
        self.W_value = layers.Linear(d_in,d_out,bias=qkv_bias) 
        self.out_proj = layers.Linear(d_out,d_out)
        self.dropout = layers.Dropout(dropout) 
        mask = np.triu(np.ones((context_length, context_length)), k=1)  # Create upper triangular mask
        self.mask = Tensor(mask,requires_grad=False) 
    
    def __call__(self,x):
        B,T,D_in = x.shape 
        Q = self.W_query(x) # (B,T,D_out) 
        K = self.W_key(x) 
        V = self.W_value(x) 
        Q = Q.view((B,T,self.num_heads,self.head_dim)).transpose((1,2))
        K = K.view((B,T,self.num_heads,self.head_dim)).transpose((1,2)) 
        V = V.view((B,T,self.num_heads,self.head_dim)).transpose((1,2))

        # attention scores 
        attn_scores = (Q @ K.transpose((-2,-1))) / (self.head_dim ** 0.5)  
        mask  = self.mask[:T,:T].bool()
        attn_scores = attn_scores.masked_fill(mask[None,None,:,:],float('-inf'))
        #sotfmax and dropout 
        attn_weights = attn_scores.softmax(axis=-1)
        attn_weights = self.dropout(attn_weights)
        context  = attn_weights @ V 
        context = context.transpose((1,2)).view((B,T,self.d_out)) 
        context = self.out_proj(context) 
        return context 
    def parameters(self):
        parameters = [] 
        parameters.append(self.W_key.parameters())
        parameters.append(self.W_query.parameters())
        parameters.append(self.W_value.parameters())
        parameters.append(self.out_proj.parameters())
        
    


# %%
class TransformerBlock: 
    def __init__(self,config:dict):
        self.att = MultiHeadAttention(
            d_in= config['emb_dim'],
            d_out = config['emb_dim'], 
            context_length=config['context_length'], 
            num_heads = config['n_heads'], 
            dropout = config['drop_rate'], 
            qkv_bias=config['qkv_bias'] 
        ) 

        self.ff = FeedForward(config) 
        # Add embedding dimension to LayerNorm
        self.norm1 = layers.LayerNorm(dim=config['emb_dim'])
        self.norm2 = layers.LayerNorm(dim=config['emb_dim'])
        self.drop_shortcut = layers.Dropout(config['drop_rate']) 
    def __call__(self,x):
        shortcut = x 
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x 

    def parameters(self):
        parameters = []
        parameters.append(self.ff.parameters())
        parameters.append(self.norm1.parameters())
        parameters.append(self.norm2.parameters())
        parameters.append(self.drop_shortcut.parameters())
        parameters.append(self.att.parameters())
        
    
    

# %%
class GPT2:
    def __init__(self,config:dict): 
        self.tok_emb = layers.Embedding(config['vocab_size'],config['emb_dim']) 
        self.pos_emb = layers.Embedding(config['context_length'],config['emb_dim'])
        self.drop_emb = layers.Dropout(config['drop_rate']) 
        self.trf_blocks = [TransformerBlock(config) for _ in range(config['n_layers'])]
        # Add embedding dimension to final LayerNorm
        self.final_norm = layers.LayerNorm(dim=config['emb_dim'])
        self.out_head = layers.Linear( 
            config['emb_dim'], config['vocab_size'],bias=False
        )

    def parameters(self):
        params = [] 
        params += self.tok_emb.parameters() 
        params += self.pos_emb.parameters() 
        params += self.final_norm.parameters() 
        params += self.out_head.parameters() 
        for block in self.trf_blocks:
            params += block.parameters() 
        return params
    
    def __call__(self,in_idx:Tensor): 
        batch_size , seq_len = in_idx.shape 
        tok_embeds = self.tok_emb(in_idx) 
        pos_embeds = self.pos_emb(
            Tensor(np.arange(seq_len))
        )
        x  = tok_embeds + pos_embeds # broadcasting will happen here 
        x = self.drop_emb(x) 
        for block in self.trf_blocks:
            x = block(x) 
        x = self.final_norm(x) 
        logits = self.out_head(x) 
        return logits


# %%
model = GPT2(GPT_CONFIG_124M)

# %%



