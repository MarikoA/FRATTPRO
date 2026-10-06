import argparse
import os
import datetime
import pickle
import time
import warnings
from joblib import Parallel, delayed
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from FRATTVAE.scripts.models.frattvae import FRATTVAE
from FRATTVAE.scripts.models.wrapper import CVAEwrapper
from FRATTVAE.scripts.utils.apps import second2date
from FRATTVAE.scripts.utils.chem_metrics import get_all_metrics, METRICS_DICT
from FRATTVAE.scripts.utils.preprocess import SmilesToMorganFingetPrints
from utils.data import fPLG_fast_Dataset, get_emb, check_columns_exist, expand_vectors, ProteinParser, CalculateAffinity, load_vector, calculate_tanimoto_similarity
from utils.process import generate
from models.fplg import fPLG_fast

from esm import pretrained

# set parser
parser = argparse.ArgumentParser(description= 'please enter paths')
parser.add_argument('yml', type= str, help= 'yml file')
parser.add_argument('pdb_dir', type= str, help= 'A group of directories containing pdb for test proteins and sdf files for reference compounds')
parser.add_argument('condition_table', type= str, help= 'condition table')
parser.add_argument('--finetuning', type= str, default= None, help= 'fine-tuning dir name')
parser.add_argument('--load_epoch', type= int, default= None, help= 'load model at load epoch')
parser.add_argument('--gen', action= 'store_true', help= 'only generation')
parser.add_argument('--dock', action= 'store_true', help= 'only docking')

parser.add_argument('--gpu', type= int, default= None, help= 'gpu device ids')
parser.add_argument('--n_jobs', type= int, default= 1, help= 'the number of cpu for parallel, default 24')
parser.add_argument('--free_n', action= 'store_true')
parser.add_argument('--smina', type= str, default= '/home/miniconda-user/workspace/smina-code/build-fPLG/smina', help= 'smina comand')
args = parser.parse_args()


if args.gen == args.dock:
    args.gen = args.dock = True

yml_file = args.yml

print(f'---{datetime.datetime.now()}: start.---', flush= True)
start = time.time()
# check environments
if args.gpu is None:
    device = 'cpu'
    print(f'using CPU\n', flush= True)
elif torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.set_device(args.gpu)
    print(f'GPU [{args.gpu}] is available: {torch.cuda.is_available()}\n', flush= True)
else:
    device = 'cpu'
    print(f'using CPU\n', flush= True)

## load hyperparameters
with open(yml_file) as yml:
    params = yaml.safe_load(yml)
print(f'load: {yml_file}', flush= True)
# path
data_path = params['data_path'] # protein-ligand data
model_path= params['model_path'] # pretrained FRATTVAE
result_path = params['result_path'] # result directory

# hyperparameters for decomposition and tree-fragments
decomp_params = params['decomp']
n_bits = decomp_params['n_bits']
max_nfrags = decomp_params['max_nfrags']
dupl_bits = decomp_params['dupl_bits']
radius = decomp_params['radius']
max_depth = decomp_params['max_depth']
max_degree = decomp_params['max_degree']
useChiral = decomp_params['useChiral']
ignore_dummy = decomp_params['ignore_dummy']

# hyperparameters for model
model_params = params['model']
d_model = model_params['d_model']
d_ff = model_params['d_ff']
num_layers = model_params['nlayer']
num_heads = model_params['nhead']
activation = model_params['activation']
latent_dim = model_params['latent']
feat_dim = model_params['feat']
props = model_params['property']
pnames = list(props.keys())
dropout = model_params['dropout']

# esm
esm_params = params['esm']
model_name = esm_params['model_name']
toks_per_batch = esm_params['toks_per_batch']
d_ReLU = esm_params['d_ReLU']

# train
train_params = params['train']
batch_size = train_params['batch_size']

os.makedirs(os.path.join(result_path, 'test'), exist_ok=True)


dname = data_path.split('/')[-1].split('.')[0]
directory_path = os.path.dirname(data_path)
par_path = os.path.dirname(directory_path)

os.makedirs(os.path.join(par_path, 'smina-output'), exist_ok=True)

test_pdblist = sorted(os.listdir(args.pdb_dir))
pro_file = ['%s/%s/%s_protein.pdb' % (args.pdb_dir, pdb, pdb) for pdb in test_pdblist]
ligand_file = ['%s/%s/%s_ligand.sdf' % (args.pdb_dir, pdb, pdb) for pdb in test_pdblist]

emb_path = os.path.join(par_path, 'embedding', os.path.basename(args.pdb_dir))

df_condition = pd.read_csv(args.condition_table)
existence_check = check_columns_exist(df_condition, pnames)
num_condition = len(df_condition)

gen_smiles = os.path.join(result_path, 'test', 'generated_smiles.csv')

# generation
if args.gen:
    s = time.time()
    print(f'---{datetime.datetime.now()}: Generation start.---', flush= True)

    print(f'For each {len(pro_file)} proteins, {num_condition}(num_condition) ligand candidates are generated.\n', flush=True)

    ## load data
    df = pd.read_csv(os.path.join(result_path, 'input_data', 'SMILES_inputed.csv'))
    df_frag = pd.read_csv(os.path.join(result_path, 'input_data', 'fragments.csv'))
    uni_fragments = df_frag.SMILES.tolist()
    freq_label = df_frag['frequency'].tolist()
    with open(os.path.join(result_path, 'input_data', 'dataset.pkl'), 'rb') as f:
        dataset = pickle.load(f)
    try:
        with open(os.path.join(result_path, 'input_data', 'csr_ecfps.pkl'), 'rb') as f:
                frag_ecfps = pickle.load(f).toarray()
                frag_ecfps = torch.from_numpy(frag_ecfps).float()
                num_labels = frag_ecfps.shape[0]
    except Exception as e:
        print(e, flush= True)
        frag_ecfps = torch.tensor(SmilesToMorganFingetPrints(uni_fragments[1:], n_bits= n_bits, dupl_bits= dupl_bits, radius= radius, 
                                                            ignore_dummy= ignore_dummy, useChiral= useChiral, n_jobs= n_jobs)).float()
        frag_ecfps = torch.vstack([frag_ecfps.new_zeros(1, n_bits+dupl_bits), frag_ecfps])      # padding feature is zero vector
    prop_dim = sum(list(props.values())) if pnames else None
    ndummys = torch.tensor(df_frag['ndummys'].tolist()).long()

    print(f'test protein data directory: {args.pdb_dir}', flush= True)

    # define model
    esm_model, _ = pretrained.load_model_and_alphabet(model_name)
    embedding_dim = esm_model.embed_dim
    print(f'pretrained esm model loaded: {model_name}', flush= True)

    frattvae_model = FRATTVAE(num_labels, max_depth, max_degree, feat_dim, latent_dim, 
                    d_model, d_ff, num_layers, num_heads, activation, dropout).to(device)
    frattvae_model.fc_memory = nn.Sequential(
        nn.Linear(embedding_dim, d_ReLU),
        nn.ReLU(),
        nn.Linear(d_ReLU, d_model)
    )
    frattvae_model = CVAEwrapper(frattvae_model, pnames, list(props.values())).to(device)


    # load model
    model = fPLG_fast(frattvae_model).to(device)

    if args.finetuning:
        if args.load_epoch:
            load_epoch = args.load_epoch
            model.load_state_dict(torch.load(os.path.join(result_path, args.finetuning, f'model_finetuning_iter{load_epoch}.pth'), map_location= device))
            print(f'model loaded: {os.path.join(result_path, args.finetuning, f"model_finetuning_iter{load_epoch}.pth")}\n', flush= True)
        else:
            load_epoch = '_best'
            model.load_state_dict(torch.load(os.path.join(result_path, args.finetuning, f'model_finetuning_best.pth'), map_location= device))
            print(f'model loaded: {os.path.join(result_path, args.finetuning, f"model_finetuning_best.pth")}\n', flush= True)
    else:
        if args.load_epoch:
            load_epoch = args.load_epoch
            model.load_state_dict(torch.load(os.path.join(result_path, 'models', f'model_iter{load_epoch}.pth'), map_location= device))
            print(f'model loaded: {os.path.join(result_path, "models", f"model_iter{load_epoch}.pth")}\n', flush= True)
        else:
            load_epoch = '_best'
            model.load_state_dict(torch.load(os.path.join(result_path, 'models', f'model_best.pth'), map_location= device))
            print(f'model loaded: {os.path.join(result_path, "models", f"model_best.pth")}\n', flush= True)
    model.frattvae_model.vae.PE._update_weights()
    model.eval()

    # make fasta for test pdb
    output_fasta = os.path.join(directory_path, os.path.basename(args.pdb_dir) + ".fasta")
    if not os.path.exists(output_fasta):
        records = []
        for (pdb, protein) in zip(test_pdblist, pro_file):
            protein_seq = ProteinParser(pdb, protein) 
            record = SeqRecord(Seq(protein_seq), id=pdb, description="")
            records.append(record)
        with open(output_fasta, "w") as output_handle:
            SeqIO.write(records, output_handle, "fasta")
        print(f"fasta file: {output_fasta}", flush=True)

    # make embbeding
    if not os.path.exists(emb_path):
        nogpu = False
        get_emb(output_fasta, emb_path, toks_per_batch, model_name, device)
        print("make emb file done", flush=True)

    vectors = Parallel(n_jobs=args.n_jobs)(delayed(load_vector)(emb_path, idx) for idx in test_pdblist)
    expanded_vectors = expand_vectors(vectors, num_condition)
    expanded_pdbids = expand_vectors(test_pdblist, num_condition)

    ## props
    df = pd.concat([df_condition] * len(test_pdblist), ignore_index=True)
    prop = df[pnames].to_numpy().tolist()
    props = torch.tensor(prop).reshape(len(df), -1).float()

    num_samples = len(props)
    frag_indices = [0] * num_samples  # 適当なインデックスリスト
    positions = [0] * num_samples 
    dataset = fPLG_fast_Dataset(frag_indices, positions, expanded_vectors, props)
    dataloader = DataLoader(dataset, batch_size= batch_size, shuffle= False)
    dec_smiles = generate(dataloader, uni_fragments, frag_ecfps, ndummys, model, pnames,
                                max_nfrags, useChiral, args.free_n, args.n_jobs, False, device) 
    print('one of generated smiles: ', dec_smiles[0], flush=True)

    # QED, logP等の計算
    properties = Parallel(n_jobs= args.n_jobs)(delayed(get_all_metrics)(s) for s in dec_smiles)
    prop_dict = {f'{key}': list(prop) for key, prop in zip(METRICS_DICT.keys(), zip(*properties))}
    print('calc Metrics done.', flush=True)


    # dec_smilesからN個ずつのグループを作成し、それぞれのグループで重複を削除
    all_groups = [list(set(dec_smiles[i:i+num_condition])) for i in range(0, len(dec_smiles), num_condition)]

    # all_groupsを一つのリストにまとめる
    all_combined = [item for group in all_groups for item in group]

    # uniqueリストを作成（重複を削除）
    unique = list(set(all_combined))

    # allとuniqueの割合を計算
    all_count = len(all_combined)
    unique_count = len(unique)
    uniqueness = unique_count / all_count if all_count > 0 else 0
    print('Uniqueness: ', uniqueness, flush=True)

    df = pd.DataFrame({
            **{prop_name: props[:, i] for i, prop_name in enumerate(pnames)}, 
            "smiles": dec_smiles,
            "PDBIDs": expanded_pdbids,
            **prop_dict
        })
    df.to_csv(gen_smiles, index= False)
    print("generated smiles can be saved\n", flush=True)
    print(f'Average {", ".join([f"{key}: {np.nanmean(values):.4f}" for key, values in prop_dict.items()])} (elapsed time: {second2date(time.time()-s)})\n', flush= True)
    print(f'---{datetime.datetime.now()}: Generation done. (elapsed time: {second2date(time.time()-s)})---\n', flush= True)


if args.dock:
    ## Docking
    print(f'---{datetime.datetime.now()}: Docking start.---', flush= True)
    start = s = time.time()

    df = pd.read_csv(gen_smiles)
    dec_smiles = df['smiles'].tolist()

    results = Parallel(n_jobs=args.n_jobs, verbose=10)(delayed(CalculateAffinity)(smile, file_protein=pro_file[idx // num_condition], \
        file_lig_ref=ligand_file[idx // num_condition], out_path=os.path.join(par_path, 'smina-output'), smina_path=args.smina, calc_ref=True) for idx, smile in enumerate(dec_smiles))
    affinities = [result[0] for result in results]
    ref_affinities = [result[1] for result in results]

    print('length of affinities: ', len(affinities), flush=True)

    if len(affinities) == len(dec_smiles):
        df['affinities'] = affinities
        df['ref_affinities'] = ref_affinities
        df.to_csv(os.path.join(result_path, 'test', f'Affinities.csv'), index= False)
        print("affinities table can be saved\n", flush=True)
        # 各ターゲット化合物のうち、affinityが最小の化合物のみによるdataframeの作成
        df['group'] = df.index // num_condition
        grouped = df.groupby('group', group_keys=True)
        filtered = grouped.filter(lambda g: not g['affinities'].isna().all())
        top_df = filtered.loc[filtered.groupby('group')['affinities'].idxmin()]
        unique_all_df = df.groupby('group').apply(lambda x: x.drop_duplicates(subset='smiles')).reset_index(drop=True)
        df_unique = unique_all_df.drop_duplicates(subset='smiles')
        df_unique_top = top_df.drop_duplicates(subset='smiles')
        top_df = top_df.drop(columns=['group'])
        # High Affinity
        df_clean = df.dropna(subset=['affinities', 'ref_affinities'])
        df_clean['greater'] = df_clean['affinities'] < df_clean['ref_affinities']
        percentage_true = df_clean['greater'].mean() # trueが1, falseが0として扱われるため、平均の計算は実質割合の計算になる。
        top_df_clean = top_df.dropna(subset=['affinities', 'ref_affinities'])
        top_df_clean['greater'] = top_df_clean['affinities'] < top_df_clean['ref_affinities']
        top_percentage_true = top_df_clean['greater'].mean() # trueが1, falseが0として扱われるため、平均の計算は実質割合の計算になる。
        # Tanimoto Similarity (top only)
        similarity_results = calculate_tanimoto_similarity(top_df, args.pdb_dir)
        mean_similarity = similarity_results["Similarity"].mean()
        print(f"[all] Docking Score: {df['affinities'].mean()}, High Affinity: {percentage_true}, Uniqueness: {len(df_unique)/len(unique_all_df)}, logP: {df['logP'].mean()}, QED: {df['QED'].mean()}, SA: {df['SA'].mean()}, NP: {df['NP'].mean()}", flush=True)
        print(f"[top] Docking Score: {top_df['affinities'].mean()}, High Affinity: {top_percentage_true}, Uniqueness: {len(df_unique_top)/len(top_df)}, similarity: {mean_similarity}, logP: {top_df['logP'].mean()}, QED: {top_df['QED'].mean()}, SA: {top_df['SA'].mean()}, NP: {top_df['NP'].mean()}", flush=True)

    else:
        print("failed to save an affinites' tabel\n", flush=True)



    s = time.time()
    print(f'---{datetime.datetime.now()}: Docking done. (elapsed time: {second2date(time.time()-start)})---\n', flush= True)




print(f'---{datetime.datetime.now()}: all process done. (elapsed time: {second2date(time.time()-start)})---\n', flush= True)