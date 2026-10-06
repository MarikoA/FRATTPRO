import numpy as np

from joblib import Parallel, delayed

import torch
import torch.nn as nn

from FRATTVAE.scripts.utils.apps import torch_fix_seed
from FRATTVAE.scripts.utils.construct import constructMol


def generate(dataloader,
             labels: list,
             frag_ecfps: torch.Tensor,
             ndummys: torch.Tensor,
             model: nn.Module,
             name_conditions: list,
             max_nfrags: int= 30, 
             useChiral: bool= True,
             free_n: bool= False,
             n_jobs: int= -1,
             random: bool= False,
             device: torch.device= torch.device('cpu'),
             seed: int= 0
            ):
    torch_fix_seed(seed)

    labels = np.array(labels)
    frag_idxs_list, adjs_list = [], []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            z = data[2].to(device)
            conditions = {key: data[3][:, i].float().to(device) for i, key in enumerate(name_conditions)}

            # decode
            tree_list = model.frattvae_model.sequential_decode(z, conditions, frag_ecfps, ndummys, max_nfrags, free_n) 

            # stock in list
            # z_list.append(z_dash.cpu())
            frag_idxs, adjs = zip(*[(tree.dgl_graph.ndata['fid'].squeeze(-1).tolist(), tree.adjacency_matrix().tolist()) for tree in tree_list])
            frag_idxs_list += list(frag_idxs)
            adjs_list += list(adjs)

            # if ((i+1) % 10 == 0) | (i == 0) | ((i+1) == len(dataloader)):
            #     print(f'[{i+1:0=3}/{len(dataloader):0=3}] latent_similarity: {cosines.mean().item():.6f}, elapsed time: {second2date(time.time()-s)}', flush= True)
            #     s = time.time()

    # constructMol
    dec_smiles = Parallel(n_jobs= n_jobs)(delayed(constructMol)(labels[idxs].tolist(), adj, useChiral= useChiral) for idxs, adj in zip(frag_idxs_list, adjs_list))

    torch.cuda.empty_cache()

    return dec_smiles