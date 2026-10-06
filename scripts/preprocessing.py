import os
import argparse
import copy
import numpy as np
import pandas as pd
import time
import datetime
import re
import moses
import warnings
warnings.simplefilter('ignore')

from rdkit import Chem
import pickle
import yaml
from scipy import sparse
from joblib import Parallel, delayed

import torch

from FRATTVAE.scripts.utils.tree import get_tree_features
from FRATTVAE.scripts.utils.apps import second2date
from FRATTVAE.scripts.utils.preprocess import parallelMolsToBRICSfragments, smiles2mol, SmilesToMorganFingetPrints
from FRATTVAE.scripts.utils.chem_metrics import normalize
from utils.data import fPLG_fast_Dataset, DictProcessor, check_columns_exist, get_emb, load_vector

# set parser
parser = argparse.ArgumentParser(description= 'please enter paths')
parser.add_argument('data_path', type= str, help= 'csv file')
parser.add_argument('esm', type= str, help= 'pretrained esm-2 model name')
parser.add_argument('--gpu', type= int, default= None, help= 'GPU device ID. If u want to use CPU, you should not use this')
parser.add_argument('--pretrained_model', type= str, default= None, help= 'directory of pretrained FRATTVAE. ex) ~/FRATTVAE/result/ChEMBLE_DB_tg_standardized_struct. if not, we dont use pretrained model.')
parser.add_argument('--seed', type= int, default= 0, help= 'random seed')
parser.add_argument('--n_jobs', type= int, default= 1, help= 'the number of cpu for parallel, default 1')
parser.add_argument('--standardized', action= 'store_true')
parser.add_argument('--normalize', action= 'store_true', help= 'Normalize(min-max) properties[MW, QED, SA, NP, TPSA, BertzCT] using default norm-parameters. Properties not included in the list are not processed.')

# decompose
parser.add_argument('--maxDepth', type= int, default= 32, help= 'max number of fragments per a mol')
parser.add_argument('--maxDegree', type= int, default= 16, help= 'max number of degree per a fragment')
parser.add_argument('--minSize', type= int, default= 1, help= 'min number of atoms per a fragment')
parser.add_argument('--ecfpBits', type= int, default= 2048, help= 'number of ecfp bits')
parser.add_argument('--ecfpRadius', type= int, default= 2, help= 'number of ecfp bits')
parser.add_argument('--useChiral', type= int, default= 1, help= 'use chirality')
parser.add_argument('--breakDouble', action= 'store_false')
parser.add_argument('--ignoreDummy', action= 'store_true')
parser.add_argument('--free_n', action= 'store_true')

# frattvae model (if u dont use pretrained model)
parser.add_argument('--d_model', type= int, default= 512, help= 'd_model')
parser.add_argument('--d_ff', type= int, default= 2048, help= 'd_ff')
parser.add_argument('--nlayer', type= int, default= 6, help= 'num_layers')
parser.add_argument('--nhead', type= int, default= 8, help= 'num_heads')
parser.add_argument('--d_latent', type= int, default= 256, help= 'd_latent')
parser.add_argument('--dropout', type= float, default= 0.1, help= 'dropout')
parser.add_argument('--activation', type= str, default= 'gelu', choices= ['relu', 'gelu'], help= "activation func. please choice from ['relu', 'gelu']")

# fPLG
parser.add_argument('--len', type= int, default= 2046, help= 'length of sequence')
parser.add_argument('--toks_per_batch', type= int, default= 4096, help= 'batch size for esm')
parser.add_argument('--d_ReLU', type= int, default= 256, help= 'Intermediate dimension of the conversion layer')
parser.add_argument('--property', default= {}, action= DictProcessor, help= 'property_column_name: num_categories. if the task is regression or binary classification, num_categories = 1. ex. QED:1')

# main trainning parameters
parser.add_argument('--epoch', type= int, default= 1000, help= 'num_epoch')
parser.add_argument('--batch_size', type= int, default= 512, help= 'batch size for fPLG')
parser.add_argument('--lr', type= float, default= 0.0001, help= 'learning rate')
args = parser.parse_args()

print(f'---{datetime.datetime.now()}: pre-processing start.---', flush= True)
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

start = time.time()

if bool(args.property):
    props = args.property
else:
    props = {}
pnames = list(props.keys())
spl = '_' if bool(pnames) else ''

data_path = args.data_path
df = pd.read_csv(data_path)
dname = data_path.split('/')[-1].split('.')[0]
directory_path = os.path.dirname(data_path)
par_path = os.path.dirname(directory_path)
today = datetime.datetime.today()
if args.pretrained_model is None:
    pretrain = "noPretrain"

    min_size = args.minSize
    max_depth = args.maxDepth
    max_degree = args.maxDegree
    useChiral = bool(args.useChiral)
    breakDouble = args.breakDouble

    d_model = args.d_model
    d_ff = args.d_ff
    nlayer = args.nlayer
    nhead = args.nhead
    d_latent = args.d_latent
    dropout = args.dropout
    activation = args.activation

else:
    pretrain = "Pretrain"

    with open(os.path.join(args.pretrained_model, 'input_data', 'params.yml')) as yml:
        params = yaml.safe_load(yml)

    decomp_params = params['decomp']
    min_size = decomp_params['min_size']
    max_depth = decomp_params['max_depth']
    max_degree = decomp_params['max_degree']
    useChiral = decomp_params['useChiral']
    breakDouble = decomp_params['ignore_double']

    model_params = params['model']
    d_model = model_params['d_model']
    d_ff = model_params['d_ff']
    nlayer = model_params['nlayer']
    nhead = model_params['nhead']
    d_latent = model_params['latent']
    dropout = model_params['dropout']
    activation = model_params['activation']

result_path = os.path.abspath(os.path.join("..", "results", f"{dname}{spl}{'_'.join(pnames)}_fast_{pretrain}_{today.month:0>2}{today.day:0>2}"))
main_dir = os.path.abspath("..")
os.makedirs(os.path.join(result_path, 'input_data'), exist_ok= False)

print(f'loaded data path: {data_path} (N = {len(df)})', flush= True)
print(f'result directry path: {result_path}', flush= True)

# check colum names
columns_to_check = ['ID', 'SMILES', 'Sequence']
check_columns_exist(df, columns_to_check)
check_columns_exist(df, pnames)
df.dropna(subset=["ID", "SMILES", "Sequence"], inplace=True)
print(f'after drop nan: N = {len(df)}', flush= True)

# check the letter of sequences
pattern = re.compile('^[ACDEFGHIKLMNPQRSTVWY]+$')
df = df[df['Sequence'].apply(lambda x: bool(pattern.match(x)))]
print(f"Number of sequences with the correct letter: {len(df)}", flush=True)

# length of sequences
df['length'] = df['Sequence'].apply(len)
df = df[df['length'] <= args.len]
print(f"Number of Sequence less than {args.len} in length: {len(df)}\n", flush=True)

# remove chirality if u need
if useChiral == 0:
    df['SMILES'] = [Chem.CanonSmiles(s, useChiral= useChiral) for s in df.SMILES]

# decompose mols
mols = Parallel(n_jobs= args.n_jobs)(delayed(smiles2mol)(s) for s in df.SMILES)

if args.pretrained_model is None:
    df_frag = []
    print(f'now we dont use pretrained FRATTVAE', flush= True)
else:
    frag_path = os.path.join(args.pretrained_model, 'input_data', f'fragments.csv')
    df_frag = pd.read_csv(frag_path)
    print(f'fragments load from pretrained model: {frag_path}', flush= True)

fragments_list, bondtypes_list, bondMapNums_list \
, recon_flag, uni_fragments, freq_label = parallelMolsToBRICSfragments(mols,
                                                                       minFragSize = min_size, maxFragNums= max_depth, maxDegree= max_degree,
                                                                       useChiral= bool(useChiral), ignore_double= args.breakDouble, 
                                                                       df_frag= df_frag, asFragments= False,
                                                                       n_jobs= args.n_jobs, verbose= 0)
frag_lens = list(map(len, fragments_list))
recon_flag = np.array(recon_flag)

if len(df_frag) != len(uni_fragments):
    print(f'class increment: {len(df_frag)} -> {len(uni_fragments)}', flush= True)
    df_frag = pd.DataFrame({'SMILES': uni_fragments, 'frequency': freq_label, 'ndummys': [f.count('*') for f in uni_fragments]})
    frag_path = os.path.join(result_path, 'input_data', 'fragments.csv')
    df_frag.to_csv(frag_path, index= False)

if not np.all(recon_flag>0):
    # if there are any compounds that are not reconstructable
    df = df.loc[recon_flag>0].reset_index(drop= True)
    df = df.assign(nfrags= frag_lens, recon= recon_flag[recon_flag>0])
else:
    df_tmp = pd.DataFrame({'idx': df.index.tolist(), 'nfrags': frag_lens, 'recon': recon_flag[recon_flag>0]})
    df_tmp.to_csv(os.path.join(result_path, 'input_data', 'num_nodes.csv'), index= False)
    del df_tmp

if sum(recon_flag>1) == 0:
    useChiral = False

print(f'reconstruct2D: {sum(recon_flag>0)/len(df.SMILES):.4f} ({sum(recon_flag>0)}/{len(df.SMILES)})', flush= True)
print(f'reconstruct3D: {sum(recon_flag==3)/sum(recon_flag>1):.4f} ({sum(recon_flag==3)}/{sum(recon_flag>1)})\n', flush= True)

# fragments to ECFP
dupl_bits = 0
frag_ecfps = np.array(SmilesToMorganFingetPrints(uni_fragments[1:], n_bits= args.ecfpBits, dupl_bits= dupl_bits, radius= args.ecfpRadius, 
                                                 ignore_dummy= args.ignoreDummy, useChiral= bool(args.useChiral), n_jobs= args.n_jobs))
frag_ecfps = np.vstack([np.zeros((1, args.ecfpBits+dupl_bits)), frag_ecfps])      # padding feature is zero vector
assert frag_ecfps.shape[0] == len(uni_fragments)
csr_ecfps = sparse.csr_matrix(frag_ecfps)
with open(os.path.join(result_path, 'input_data', 'csr_ecfps.pkl'), 'wb') as f:
    pickle.dump(csr_ecfps, f)

if not 'test' in df.columns:
    df['orig_index'] = df.index
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_total = len(df_shuffled)
    n_train = int(n_total * 0.8)
    n_valid = int(n_total * 0.1)
    n_test = n_total - n_train - n_valid
    df_shuffled['test'] = np.nan
    df_shuffled.loc[:n_train - 1, 'test'] = 0          # train
    df_shuffled.loc[n_train:n_train + n_valid - 1, 'test'] = -1  # valid
    df_shuffled.loc[n_train + n_valid:, 'test'] = 1       # test
    df = df_shuffled.sort_values('orig_index').reset_index(drop=True)
    df = df.drop(columns='orig_index')

print(f"total: {len(df)}", flush=True)
print(f"train: {len(df[df['test'] == 0])}", flush=True)
print(f"valid: {len(df[df['test'] == -1])}", flush=True)
print(f"test: {len(df[df['test'] == 1])}", flush=True)
print(f'useChiral: {useChiral}, max_depth: {max_depth}, max_degree: {max_degree}, n_jobs: {args.n_jobs}', flush= True)


df.to_csv(os.path.join(result_path, 'input_data', 'SMILES_inputed.csv'), index= False)
print('SMILES_inputed.csv saved\n', flush= True)

# normalize
if pnames:
    if args.normalize:
        prop = np.array([normalize(df[p].to_numpy(), p) for p in pnames]).T.tolist()
        print('normalized done\n', flush=True)
    else:
        prop = df[pnames].to_numpy().tolist()
        print('normalization process is NOT performed\n', flush=True)
    prop_dim = sum(list(props.values()))
else:
    prop = [[float('nan')] for _ in range(len(df))]
    prop_dim = None
    print('there is no property\n', flush=True)

# esm-2
print(f'pretrained esm-2 model is {args.esm}', flush= True)

# get frag_indices and positions
features = Parallel(n_jobs= args.n_jobs)(delayed(get_tree_features)(f, torch.zeros(len(f), 1).float(), b, m, max_depth, max_degree, args.free_n) for f, b, m in zip(fragments_list, bondtypes_list, bondMapNums_list))
frag_indices, _, positions = zip(*features)

fasta_path=os.path.join(directory_path, f'{dname}.fasta')
df_unique = df.drop_duplicates(subset=['ID'], keep='first')
# make Fasta
if not os.path.exists(fasta_path):
    with open(fasta_path, 'w') as f:
        for index, row in df_unique.iterrows():
            # FASTA形式でファイルに書き込む
            f.write(f'>{row["ID"]}\n{row["Sequence"]}\n')
    print("make fasta file done", flush=True)
else:
    print("fasta file already exist", flush=True)
# make embbeding
emb_path = os.path.join(par_path, 'embedding', f'{dname}')
if not os.path.exists(emb_path):
    get_emb(fasta_path, emb_path, args.toks_per_batch, args.esm, device)
    print("make emb file done", flush=True)
else:
    print("embedding directory already exist", flush=True)
# get emb
vectors = Parallel(n_jobs=args.n_jobs)(delayed(load_vector)(emb_path, idx) for idx in df['ID'])
print("get emb file done", flush=True)

prop = torch.tensor(prop).reshape(len(df), -1).float()
print(f'Intermediate dimension of the conversion layer: {args.d_ReLU}', flush= True)


dataset = fPLG_fast_Dataset(frag_indices, positions, vectors, prop)
assert len(df) == len(dataset)
with open(os.path.join(result_path, 'input_data', 'dataset.pkl'), 'wb') as f:
    pickle.dump(dataset, f)


print(f'fragments: {len(uni_fragments)}, feature: {args.ecfpBits+dupl_bits}, degree: {df_frag.ndummys.max()}', flush= True)
print(f'fragments per mol: {min(frag_lens)} - {max(frag_lens)} (mean: {np.mean(frag_lens):.1f}, median: {np.median(frag_lens):.1f}), mol as a fragment: {frag_lens.count(1)}', flush= True)
print("Dataset creation complete\n", flush=True)

# make yaml
with open(os.path.join(result_path, 'input_data', 'params.yml'), 'w') as yf:
    yf.write(f'# Date of creation: {datetime.datetime.now()}\n')
    yaml.dump({
        'data_path': args.data_path,
        'model_path': args.pretrained_model, 
        'result_path': result_path,
        'seed': args.seed,

        'decomp': {
            'min_size': min_size,
            'max_nfrags': max_degree,
            'n_bits': args.ecfpBits,
            'dupl_bits': dupl_bits,
            'radius': args.ecfpRadius,
            'max_depth': max_depth,
            'max_degree': max_degree,
            'useChiral': useChiral,
            'ignore_double': breakDouble,
            'ignore_dummy': args.ignoreDummy,
        },

        'model': {
            'd_model': d_model,
            'd_ff': d_ff,
            'nlayer': nlayer,
            'nhead': nhead,
            'latent': args.d_latent,
            'feat': args.ecfpBits + dupl_bits,
            'property': props,
            'activation': activation,
            'dropout': dropout,
        },

        'esm': {
            'model_name': args.esm,
            'toks_per_batch':args.toks_per_batch,
            'd_ReLU': args.d_ReLU,
        },

        'train': {
            'epoch': args.epoch,
            'batch_size': args.batch_size,
            'lr': args.lr,
        }
        }, yf, default_flow_style= False, sort_keys= False)
    
print(f"Config is saved to {os.path.join(result_path, 'input_data', 'params.yml')}", flush= True)


print(f'---{datetime.datetime.now()}: pre-processing done. (elapsed time: {second2date(time.time()-start)})---\n', flush= True)





