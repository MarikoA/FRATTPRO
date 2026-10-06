import torch
import torch.nn as nn

class fPLG(nn.Module):
    def __init__(self, esm_model, frattvae_model):
        super(fPLG, self).__init__()
        self.esm_model = esm_model
        self.frattvae_model = frattvae_model


    def forward(self, length, toks, features, positions, conditions: dict,
                tgt_mask: torch.Tensor= None, tgt_pad_mask: torch.Tensor= None,
                frag_ecfps: torch.Tensor= None, ndummys: torch.Tensor= None,
                max_nfrags: int= 20, free_n: bool= False, use_full_rep: bool= False, sequential: bool= None):
        # esm
        with torch.no_grad():
            out = self.esm_model(toks, repr_layers=[self.esm_model.num_layers])
        token_emb = out["representations"][self.esm_model.num_layers]
        if use_full_rep:
            seq_embs = [token_emb[1:l+1] for l in length]
            seq_emb = torch.stack(seq_embs)
        else:
            seq_embs = [token_emb[1:l+1].mean(0) for l in length]
            seq_emb = torch.stack(seq_embs)

        sequential = not self.training if sequential is None else sequential

        conditions = torch.stack([self.frattvae_model.cond_embs[key](value) for key, value in conditions.items()], dim= 1)

        # frattvae decoder
        if sequential:
            output = self.frattvae_model.vae.sequential_decode(seq_emb, frag_ecfps, ndummys, conditions= conditions, max_nfrags= max_nfrags, free_n= free_n)
        else:
            output = self.frattvae_model.vae.decode(seq_emb, features, positions, tgt_mask, tgt_pad_mask, conditions)
        
        return output
    
class fPLG_fast(nn.Module):
    def __init__(self, frattvae_model):
        super(fPLG_fast, self).__init__()
        self.frattvae_model = frattvae_model


    def forward(self, vectors, features, positions, conditions: dict,
                tgt_mask: torch.Tensor= None, tgt_pad_mask: torch.Tensor= None,
                frag_ecfps: torch.Tensor= None, ndummys: torch.Tensor= None,
                max_nfrags: int= 20, free_n: bool= False, use_full_rep: bool= False, sequential: bool= None):

        sequential = not self.training if sequential is None else sequential

        conditions = torch.stack([self.frattvae_model.cond_embs[key](value) for key, value in conditions.items()], dim= 1)

        # frattvae decoder
        if sequential:
            output = self.frattvae_model.vae.sequential_decode(vectors, frag_ecfps, ndummys, conditions= conditions, max_nfrags= max_nfrags, free_n= free_n)
        else:
            output = self.frattvae_model.vae.decode(vectors, features, positions, tgt_mask, tgt_pad_mask, conditions)
        
        return output
