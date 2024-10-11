from simple_lib.nn.modules import PositionalEncoding1D
import torch
from torch import nn
from dataclasses import dataclass

class TransformerEncoder(nn.Module):
    @dataclass
    class Args:
        vocab_size: int
        embedding_dim: int
        n_heads: int
        n_layers: int
        
    def __init__(self, args: Args):
        super().__init__()
        self.args = args

        encoder_layer = nn.TransformerEncoderLayer(d_model=args.embedding_dim, nhead=args.n_heads, dim_feedforward=4*args.embedding_dim, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.n_layers, enable_nested_tensor=False)
        self.encoder_pos_enc = PositionalEncoding1D(args.embedding_dim)

        self.embedding = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.embedding_dim)
        
    def encode(self, x: torch.Tensor, *, pad_id: int) -> torch.Tensor:
        B, N = x.shape
        src_key_padding_mask = x == pad_id
        x = self.embedding(x) * (self.args.embedding_dim)**(1/2)
        x = x + self.encoder_pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x
