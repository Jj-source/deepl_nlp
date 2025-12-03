# model.py
"""
Minimal model and tokenizer utilities used by agent.py / the notebook.

Students must implement the neural architecture so that checkpoints
trained in the notebook load and decode with the same code here.

Keep the public API stable:

- SPECIAL_TOKENS : Dict[str, str]
- simple_tokenize(s: str) -> List[str]
- encode(tokens: List[str], stoi: Dict[str, int], add_sos_eos: bool=False) -> List[int]
- class Encoder(nn.Module): forward(src, src_lens)
- class Decoder(nn.Module): forward(tgt_in, hidden)
- class Seq2Seq(nn.Module):
    - forward(src, src_lens, tgt_in) -> logits [B,T,V]
    - greedy_decode(src, src_lens, max_len, sos_id, eos_id) -> LongTensor[B, max_len]
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn


# -------------------------
# Tokenization utilities
# -------------------------

SPECIAL_TOKENS = {
    "pad": "<pad>",
    "sos": "<sos>",
    "eos": "<eos>",
    "unk": "<unk>",
}


def simple_tokenize(s: str) -> List[str]:
    """Lowercase whitespace tokenizer used by both training and inference."""
    return s.strip().lower().split()


def encode(tokens: List[str], stoi: Dict[str, int], add_sos_eos: bool = False) -> List[int]:
    """Map tokens to ids using `stoi`. Optionally wrap with <sos>/<eos>."""
    ids = [stoi.get(t, stoi[SPECIAL_TOKENS["unk"]]) for t in tokens]
    if add_sos_eos:
        ids = [stoi[SPECIAL_TOKENS["sos"]]] + ids + [stoi[SPECIAL_TOKENS["eos"]]]
    return ids


# -------------------------
# Model scaffolding
# -------------------------


class EncoderBidirectionalAttn(nn.Module):
  
  def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=1, dropout=0.1):
    
    super().__init__()
    self.num_layers = num_layers
    self.hid_dim = hid_dim
    self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
    self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=num_layers, batch_first=True, bidirectional=True,
                       dropout=dropout if num_layers > 1 else 0.0)
    
  def forward(self, src, src_lens):
    
    emb = self.emb(src)
    packed = nn.utils.rnn.pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
    
    out, h = self.rnn(packed)
    out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    h = h.view(self.num_layers, 2, -1, self.hid_dim).sum(dim=1)
    return out, h

 
class Decoder(nn.Module):
  
  def __init__(self, vocab_size, emb_dim, hid_dim, num_layers=1, dropout=0.1):
      
    super().__init__()
    self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
    self.attention = LuongAttention(hid_dim, 2 * hid_dim, score_type='general')
    self.rnn = nn.GRU(emb_dim + 2 * hid_dim, hid_dim, num_layers=num_layers, batch_first=True, 
                     dropout=dropout if num_layers > 1 else 0.0)
    
    self.proj = nn.Linear(hid_dim, vocab_size)
    self.dropout = nn.Dropout(dropout)
    
  def forward_step(self, tgt_in, hidden, encoder_outputs):
      emb = self.dropout(self.emb(tgt_in))  
        
      
      h = hidden[-1:].permute(1, 0, 2)
        
      context, attn_weights = self.attention(h, encoder_outputs)
      emb = torch.cat([emb, context], dim=2)  
        
      out, hidden = self.rnn(emb, hidden)
      logits = self.proj(out)
      return logits, hidden, attn_weights
        
  def forward(self, tgt_in, hidden, encoder_outputs):
    
    batch_size, seq_len = tgt_in.size()
    outputs = []
        
    for i in range(seq_len):
      token = tgt_in[:, i:i+1]
      logits, hidden, _ = self.forward_step(token, hidden, encoder_outputs)
      outputs.append(logits)
        
    return torch.cat(outputs, dim=1), hidden
        
class Seq2Seq(nn.Module):
  
    def __init__(self, enc, dec):
      super().__init__()
      self.encoder = enc
      self.decoder = dec
      
    def forward(self, src, src_lens, tgt_in):
      enc_out , h = self.encoder(src, src_lens)
      logits, _ = self.decoder(tgt_in, h, enc_out)
      return logits
      
    @torch.no_grad()
    def greedy_decode(self, src, src_lens, max_len, sos_id, eos_id):
      
        B = src.size(0)
        
        enc_out , h = self.encoder(src, src_lens)
        
        inputs = torch.full((B, 1), sos_id, dtype=torch.long, device=src.device)
        outs = []
        
        for _ in range(max_len):
            
            logits, h, _ = self.decoder.forward_step(inputs, h, enc_out)
            
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            outs.append(nxt)
            inputs = nxt
            
        
        seqs = torch.cat(outs, dim=1)
        
        for i in range(B):
            row = seqs[i]
            
            if (row == eos_id).any():
              
              idx = (row == eos_id).nonzero(as_tuple=False)[0].item()
              
              row[idx+1:] = eos_id
        return seqs
      
      
class LuongAttention(nn.Module):
  def __init__(self, query_dim, key_dim, score_type='general'):
    super().__init__()
    self.score_type = score_type
    if score_type == 'general':
        self.Wa = nn.Linear(query_dim, key_dim, bias=False)  

  def forward(self, query, keys):
      if self.score_type == 'dot' and query.size(-1) == keys.size(-1):
          scores = torch.bmm(query, keys.transpose(1, 2))
      elif self.score_type == 'general':
          scores = torch.bmm(self.Wa(query), keys.transpose(1, 2))
      
      weights = F.softmax(scores, dim=-1)
      context = torch.bmm(weights, keys)  
      
      return context, weights