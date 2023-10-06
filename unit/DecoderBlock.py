from torch import nn

from unit.AttentionBlock import SelfAttention
from unit.TransformerBlock import TransformerBlock


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size).to(device)
        self.attention = SelfAttention(embed_size, heads=heads).to(device)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        ).to(device)
        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out
