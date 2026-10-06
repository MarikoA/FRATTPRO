import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import time
import datetime
import yaml
import pickle
import gc

from esm import pretrained
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from FRATTVAE.scripts.models.frattvae import FRATTVAE
from FRATTVAE.scripts.models.wrapper import CVAEwrapper
from FRATTVAE.scripts.utils.apps import second2date, torch_fix_seed
from FRATTVAE.scripts.utils.preprocess import SmilesToMorganFingetPrints
from FRATTVAE.scripts.utils.mask import create_mask
from utils.data import collate_fn_fPLG_fast, list2pdData
from models.fplg import fPLG_fast


# set parser
parser = argparse.ArgumentParser(description= 'please enter paths')
parser.add_argument('yml', type= str, help= 'yml file')
parser.add_argument('--pretrained_fplg', type= str, default= None, help= 'directory of pretrained fPLG. ex) ~/fPLG/result/BDB_pretrained_0201. if not, we dont use pretrained model.')
parser.add_argument('--gpus', type= int, default= [0], nargs='*', help= 'a list of gpu device ids. if len(ids) > 1, use DDP')
parser.add_argument('--n_jobs', type= int, default= 1, help= 'the number of cpu for parallel, default 24')
parser.add_argument('--load_epoch', type= int, default= 0, help= 'load model at load epoch, default epoch= 0')
parser.add_argument('--save_interval', type= int, default= 200, help= 'save model every N epochs, default N= 20')
parser.add_argument('--select_metric', type= int, default= 1, help= 'metric to select best model. 1: label loss, 2: label acc, default 0')
parser.add_argument('--valid', action= 'store_true', help= 'select models based on validation data.')
parser.add_argument('--use_full_rep', action= 'store_true', help= 'When using the output vector of esm as is. Not necessary when the output vector is averaged over the lengthwise direction of the protein.')
parser.add_argument('--master_port', type= str, default= '12356', help= 'if you use DDP, select master port. default= "12356"')
parser.add_argument('--seed', type= str, default= 0, help= 'random seed.')
args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpus))


def train(rank, 
          yml_file: str,
          load_epoch: int= 0,
          save_epoch: int= 20,
          validation: bool= True,
          n_jobs: int= 1,
          seed: int= 0,
          ) -> torch.Tensor:
    torch_fix_seed(seed)

    # set device
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.set_device(rank)
        n_gpu = torch.cuda.device_count()
    else:
        device = 'cpu'
        n_gpu = 1
    multigpu = n_gpu > 1
    if multigpu:
        dist.init_process_group(backend= "nccl", init_method='env://', rank= rank, world_size= n_gpu)

    ## preparation
    # load hyperparameters
    with open(yml_file) as yml:
        params = yaml.safe_load(yml)
    if rank == 0:
        print(f'---{datetime.datetime.now()}: Loading data. ---', flush= True)
        print(f'load: {yml_file}\n', flush= True)
    s = time.time()

    # path
    data_path = params['data_path'] # protein-ligand data
    model_path= params['model_path'] # pretrained FRATTVAE
    result_path = params['result_path'] # result directory

    # hyperparameters for decomposition and tree-fragments
    decomp_params = params['decomp']
    n_bits = decomp_params['n_bits']
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
    d_ReLU = esm_params['d_ReLU']

    # hyperparameters for training
    train_params = params['train']
    epochs = train_params['epoch']
    batch_size = train_params['batch_size'] // n_gpu if train_params['batch_size'] > n_gpu else 1
    lr = train_params['lr']

    # Creating Subdirectories
    os.makedirs(os.path.join(result_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(result_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(result_path, 'visualize'), exist_ok=True)

    ## load data
    df = pd.read_csv(os.path.join(result_path, 'input_data', 'SMILES_inputed.csv'))

    df_frag = pd.read_csv(os.path.join(result_path, 'input_data', 'fragments.csv'))
    uni_fragments = df_frag.SMILES.tolist()
    freq_label = df_frag['frequency'].tolist()
    with open(os.path.join(result_path, 'input_data', 'dataset.pkl'), 'rb') as f:
        dataset = pickle.load(f)
    try:
        if model_path is None:
            with open(os.path.join(result_path, 'input_data', 'csr_ecfps.pkl'), 'rb') as f:
                frag_ecfps = pickle.load(f).toarray()
                frag_ecfps = torch.from_numpy(frag_ecfps).float()
                num_labels = frag_ecfps.shape[0]
        else:   
            with open(os.path.join(result_path, 'input_data', 'csr_ecfps.pkl'), 'rb') as f:
                frag_ecfps = pickle.load(f).toarray()
                frag_ecfps = torch.from_numpy(frag_ecfps).float()
                new_num_labels = frag_ecfps.shape[0]
            with open(os.path.join(model_path, 'input_data', 'csr_ecfps.pkl'), 'rb') as f:
                pre_frag_ecfps = pickle.load(f).toarray()
                pre_frag_ecfps = torch.from_numpy(pre_frag_ecfps).float()
                num_labels = pre_frag_ecfps.shape[0]
    except Exception as e:
        if rank==0: print(e, flush= True)
        frag_ecfps = torch.tensor(SmilesToMorganFingetPrints(uni_fragments[1:], n_bits= n_bits, dupl_bits= dupl_bits, radius= radius, 
                                                            ignore_dummy= ignore_dummy, useChiral= useChiral, n_jobs= n_jobs)).float()
        frag_ecfps = torch.vstack([frag_ecfps.new_zeros(1, n_bits+dupl_bits), frag_ecfps])      # padding feature is zero vector
    prop_dim = sum(list(props.values())) if pnames else None

    # train valid split
    train_data = Subset(dataset, df.loc[df.test==0].index.tolist())
    valid_data = Subset(dataset, df.loc[df.test==-1].index.tolist()) if validation & np.any(df.test==-1) else None

    # make data loader
    sampler = DistributedSampler(train_data, num_replicas= n_gpu, rank= rank, shuffle= True) if multigpu else None
    train_loader = DataLoader(train_data, batch_size= batch_size, shuffle= not multigpu,
                              sampler= sampler, pin_memory= True, collate_fn= collate_fn_fPLG_fast)
    valid = bool(valid_data)
    if valid:
        valid_sampler = DistributedSampler(valid_data, num_replicas= n_gpu, rank= rank, shuffle= False) if multigpu else None
        valid_loader = DataLoader(valid_data, batch_size= batch_size, shuffle= False,
                                  sampler= valid_sampler, pin_memory= True, collate_fn= collate_fn_fPLG_fast)
        
    if rank == 0:
        print(f'data: {data_path}', flush= True)
        print(f'result path: {result_path}', flush= True)
        print(f'train: {len(train_data)}, valid: {sum(df.test==-1)}, test: {sum(df.test==1)}, useChiral: {useChiral}, n_jobs: {n_jobs}', flush= True)
        print(f'fragments: {len(uni_fragments)}, feature: {frag_ecfps.shape[-1]}, tree: ({max_depth}, {max_degree}), prop: {prop_dim}', flush= True)
        print(f'---{datetime.datetime.now()}: Loading data done. (elapsed time: {second2date(time.time()-s)})---\n', flush= True)

    # load models
    if rank==0:
        print(f'---{datetime.datetime.now()}: defining the model. ---', flush= True)
    s = time.time()
    esm_model, _ = pretrained.load_model_and_alphabet(model_name)
    embedding_dim = esm_model.embed_dim
    if rank == 0:
        print(f'pretrained esm model loaded: {model_name}', flush= True)


    frattvae_model = FRATTVAE(num_labels, max_depth, max_degree, feat_dim, latent_dim, 
                   d_model, d_ff, num_layers, num_heads, activation, dropout).to(device)
    if model_path:
        frattvae_model.load_state_dict(torch.load(os.path.join(model_path, 'models', 'model_best.pth'), map_location= "cpu"))
        frattvae_model.PE._update_weights()  # initialization
        frattvae_model.fc_dec = nn.Linear(d_model, new_num_labels) 
        for name, param in frattvae_model.named_parameters():
            if "fc_dec" in name or "fc_memory" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        if rank==0: print(f'pretrained frattvae model loaded: {os.path.join(model_path, "models", "model_best.pth")}', flush= True)
    frattvae_model.fc_memory = nn.Sequential(
        nn.Linear(embedding_dim, d_ReLU),
        nn.ReLU(),
        nn.Linear(d_ReLU, d_model)
    )
    frattvae_model = CVAEwrapper(frattvae_model, pnames, list(props.values())).to(device)
    for name, param in frattvae_model.named_parameters():
        if "cond_embs" in name:
            param.requires_grad = True

    # define model
    model = fPLG_fast(frattvae_model).to(device)
    if args.pretrained_fplg is not None:
        model.load_state_dict(torch.load(os.path.join(args.pretrained_fplg, 'models', f'model_best.pth'), map_location=device))
        if rank==0: print(f'Loaded pretrained fPLG model: {args.pretrained_fplg}')
    elif load_epoch:
        model.load_state_dict(torch.load(os.path.join(result_path, 'models', f'model_iter{load_epoch}.pth'), map_location= device))
        model.PE._update_weights()      # initialization
        if rank==0: print(f'model loaded: {os.path.join(result_path, "models", f"model_iter{load_epoch}.pth")}', flush= True)
    else:
        if rank==0: print(f'we dont use pretrained fPLG model.', flush= True)
    if multigpu: 
        flag = True     #prop_dim is not None
        model = DDP(model, device_ids= [rank], find_unused_parameters= flag)


    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr= lr, eps= 1e-3)
    if load_epoch and (args.pretrained_fplg is None):
        state_dict = torch.load(os.path.join(result_path, 'models', f'optim_iter{load_epoch}.pth'), map_location= device)
        if state_dict['param_groups'][0]['lr'] == lr:
            optimizer.load_state_dict(state_dict)
            if rank==0: print(f'optimizer loaded: {os.path.join(result_path, "models", f"optim_iter{load_epoch}.pth")}', flush= True)
        del state_dict

    if rank == 0:
        print(f'---{datetime.datetime.now()}: Model definition complete. (elapsed time: {second2date(time.time()-s)})---\n', flush= True)

    # define loss
    num_labels = len(freq_label)
    freq_label = torch.tensor(freq_label)
    freq_label[freq_label > 1000] = 1000                              # limitation
    loss_weight_label = freq_label.max() / freq_label
    loss_weight_label[loss_weight_label == float('Inf')]  = 0.001     # padding weight
    loss_weight_label = loss_weight_label.to(device) if loss_weight_label is not None else None
    criterion = nn.CrossEntropyLoss(weight= loss_weight_label)

    # release memory
    del df, df_frag, uni_fragments, freq_label, dataset, train_data, valid_data
    del loss_weight_label
    gc.collect()
    if multigpu: dist.barrier()

    ## training
    if rank == 0:
        print(f'---{datetime.datetime.now()}: Training start (valid: {valid}).---', flush= True)
        print(f'{load_epoch + 1} epoch -> {load_epoch + epochs} epoch', flush= True)
        print(f'batch_size: {batch_size}, learning_rate: {lr}, dropout: {dropout}, save: {save_epoch}', flush= True)

    metrics = ['epoch', 'loss', 'label_acc', 'label_pad_acc']
    TRAIN_LOSS, VALID_LOSS = [], []
    start = time.time()

    if rank==0:
        filename = os.path.join(result_path, 'train', f'train_{datetime.date.today()}.txt')

    metric = args.select_metric      # metric for select best model. 1: label loss, 2: label acc
    before = 0 if metric > 1 else float('-inf')
    patient = 0
    best_state_dict = model.state_dict()
    epochs = load_epoch + epochs
    ep=0
    for epoch in range(load_epoch, epochs):
        model.train()
        if multigpu: sampler.set_epoch(epoch)
        train_losses = []
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()

            frag_indices = data[0]
            features = frag_ecfps[frag_indices.flatten()].reshape(frag_indices.shape[0], frag_indices.shape[1], -1).to(device)
            positions = data[1].to(device)
            vectors = data[2].to(device)
            conditions = {key: data[3][:, i].to(device) for i, key in enumerate(pnames)}

            target = torch.hstack([frag_indices.detach(), torch.zeros(frag_indices.shape[0], 1)]).flatten().long().to(device)

            # make mask
            nan_mask = torch.where(data[3].isnan(), 0, 1)
            frag_indices = torch.hstack([nan_mask, torch.full((frag_indices.shape[0], 1), -1), frag_indices.detach()]).to(device)   # for super root
            src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(vectors, frag_indices, pad_idx= 0, batch_first= True)
            tgt_mask[:, :nan_mask.shape[-1]+1] = 0 

            # forward
            output = model(vectors, features, positions, conditions,
                            tgt_mask, tgt_pad_mask,
                            frag_ecfps=None, ndummys=None,
                            max_nfrags=None, free_n=False, use_full_rep=False) 
            
            # backward
            loss = criterion(input= output.view(-1, num_labels), target= target)
            loss.backward() 
            optimizer.step() 

            # calc accuracy
            equals = output.argmax(dim= -1).flatten().eq(target)
            label_acc = equals[target!=0].sum() / (target!=0).sum()
            label_pad_acc = equals.sum() / target.shape[0]

            if multigpu:
                dist.all_reduce(loss, op= dist.ReduceOp.SUM)
                dist.all_reduce(label_acc, op= dist.ReduceOp.SUM)
                dist.all_reduce(label_pad_acc, op= dist.ReduceOp.SUM)

            if rank==0:
                train_losses.append([epoch+1, loss.item()/n_gpu, label_acc.item()/n_gpu, label_pad_acc.item()/n_gpu])

        # validation and model save
        if (epoch == 0) | ((epoch+1) % save_epoch == 0) | ((epoch+1) == epochs):
            if multigpu: dist.barrier() 
            if valid:
                model.eval()
                valid_losses = []
                with torch.no_grad():
                    for i, data in enumerate(valid_loader):
                        optimizer.zero_grad()

                        frag_indices = data[0]
                        features = frag_ecfps[frag_indices.flatten()].reshape(frag_indices.shape[0], frag_indices.shape[1], -1).to(device)
                        positions = data[1].to(device)
                        vectors = data[2].to(device)
                        conditions = {key: data[3][:, i].to(device) for i, key in enumerate(pnames)}

                        target = torch.hstack([frag_indices.detach(), torch.zeros(frag_indices.shape[0], 1)]).flatten().long().to(device)

                        # make mask
                        nan_mask = torch.where(data[3].isnan(), 0, 1)
                        frag_indices = torch.hstack([nan_mask, torch.full((frag_indices.shape[0], 1), -1), frag_indices.detach()]).to(device)  # for super root
                        src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(vectors, frag_indices, pad_idx= 0, batch_first= True)
                        tgt_mask[:, :nan_mask.shape[-1]+1] = 0 

                        # forward
                        output = model(vectors, features, positions, conditions,
                                        tgt_mask, tgt_pad_mask,
                                        frag_ecfps=None, ndummys=None,
                                        max_nfrags=None, free_n=False, use_full_rep=False, sequential=False)
                        
                        # calc loss
                        loss = criterion(input= output.view(-1, num_labels), target= target)

                        # calc accuracy
                        equals = output.argmax(dim= -1).flatten().eq(target)
                        label_acc = equals[target!=0].sum() / (target!=0).sum()
                        label_pad_acc = equals.sum() / target.shape[0]

                        if multigpu:
                            dist.all_reduce(loss, op= dist.ReduceOp.SUM)
                            dist.all_reduce(label_acc, op= dist.ReduceOp.SUM)
                            dist.all_reduce(label_pad_acc, op= dist.ReduceOp.SUM)

                        if rank==0:
                            valid_losses.append([epoch+1, loss.item()/n_gpu, label_acc.item()/n_gpu, label_pad_acc.item()/n_gpu])
                
                if rank==0: 
                    VALID_LOSS.append([np.mean(losses) for losses in zip(*valid_losses)])
                    print(f'<valid> ' + ', '.join([f'{m}: {l:.4f}' for m, l in zip(metrics, VALID_LOSS[-1])]) + f', elapsed time: {second2date(time.time()-s)}', flush= True)
            
            # save models
            if rank == 0:
                if multigpu:
                    torch.save(model.module.state_dict(), os.path.join(result_path, 'models', f'model_iter{epoch+1}.pth'))
                else:
                    torch.save(model.state_dict(), os.path.join(result_path, 'models', f'model_iter{epoch+1}.pth'))
                torch.save(optimizer.state_dict(), os.path.join(result_path, 'models', f'optim_iter{epoch+1}.pth'))
                print(f'model_iter{epoch+1} saved.', flush= True)

        # best model save
        if rank == 0:
            TRAIN_LOSS.append([np.mean(losses) for losses in zip(*train_losses)])
            selected_metric = TRAIN_LOSS[-1][metric]
            tmp = selected_metric if (metric > 1) else -1 * selected_metric
            if tmp > before:
                ep = epoch + 1
                before = tmp
                if multigpu:
                    best_state_dict = deepcopy(model.module.state_dict())
                else: 
                    best_state_dict = deepcopy(model.state_dict())
                patient = 0
            else:
                patient += 1
           
            # print and save
            with open(filename, 'a') as f:
                f.write(f'[{epoch+1:0=3}/{epochs:0=3}] <train> ' + ', '.join([f'{m}: {l:.4f}' for m, l in zip(metrics, TRAIN_LOSS[-1])]) + f', elapsed time: {second2date(time.time()-s)}\n')
            if (epoch == 0) | ((epoch+1) % 5 == 0):
                print(f'[{epoch+1:0=3}/{epochs:0=3}] <train> ' + ', '.join([f'{m}: {l:.4f}' for m, l in zip(metrics, TRAIN_LOSS[-1])]) + f', elapsed time: {second2date(time.time()-s)}', flush= True)

            if (patient > 5) | ((epoch+1) == epochs):
                torch.save(best_state_dict, os.path.join(result_path, 'models', f'model_best.pth'))
                print(f'model_iter{ep} saved as best. [{metrics[metric]}: {abs(before):.4f}]', flush= True)
                patient = 0

    # save loss and reconstruction
    if rank==0:
        TRAIN_LOSS = list(zip(*TRAIN_LOSS))
        df_train = list2pdData(TRAIN_LOSS, metrics)
        df_train.to_csv(os.path.join(result_path, 'train', f'train_loss{load_epoch}-{epochs}.csv'), index= False)
        if VALID_LOSS:
            VALID_LOSS = list(zip(*VALID_LOSS))
            df_valid = list2pdData(VALID_LOSS, metrics)
            df_valid.to_csv(os.path.join(result_path, 'train', f'valid_loss{load_epoch}-{epochs}.csv'), index= False)
        print(f'---{datetime.datetime.now()}: Training done. (elapsed time: {second2date(time.time()-start)})---\n', flush= True)

    # save loss graph
    if rank==0:
        # label_loss
        plt.figure(figsize=(4, 5))  # グラフのサイズ指定
        plt.plot(df_train['epoch'], df_train['loss'], label='Train')
        if valid:
            plt.plot(df_valid['epoch'], df_valid['loss'], label='Valid') 
        plt.title('label_loss') 
        plt.xlabel('Epoch') 
        plt.ylabel('label_loss') 
        plt.legend()  
        plt.grid(True)
        plt.savefig(os.path.join(result_path, 'visualize', f'label_loss.png'), bbox_inches='tight', facecolor= 'white') 

        # label_acc
        plt.figure(figsize=(4, 5)) 
        plt.plot(df_train['epoch'], df_train['label_acc'], label='Train')
        if valid:
            plt.plot(df_valid['epoch'], df_valid['label_acc'], label='Valid') 
        plt.title('label_acc')
        plt.xlabel('Epoch')  
        plt.ylabel('label_acc') 
        plt.legend()  
        plt.grid(True)
        plt.savefig(os.path.join(result_path, 'visualize', f'label_acc.png'), bbox_inches='tight', facecolor= 'white') 

        # label_pad_acc
        plt.figure(figsize=(4, 5))
        plt.plot(df_train['epoch'], df_train['label_pad_acc'], label='Train')
        if valid:
            plt.plot(df_valid['epoch'], df_valid['label_pad_acc'], label='Valid') 
        plt.title('label_pad_acc')
        plt.xlabel('Epoch') 
        plt.ylabel('label_pad_acc') 
        plt.legend()  
        plt.grid(True)
        plt.savefig(os.path.join(result_path, 'visualize', f'label_pad_acc.png'), bbox_inches='tight', facecolor= 'white')

        print('all figures have been saved. \n', flush=True)
            



if __name__ == '__main__':
    yml_file = args.yml
    # yml_file = ''

    start = time.time()
    print(f'---{datetime.datetime.now()}: start.---', flush= True)

    ## check environments
    n_gpu = torch.cuda.device_count()
    print(f'GPU [{",".join([str(g) for g in args.gpus])}] is available: {torch.cuda.is_available()}', flush= True)
    if n_gpu > 1:
        print(f'DDP is available: {dist.is_available()}\n', flush= True)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.master_port
        os.environ['MKL_THREADING_LAYER'] = 'GNU'

    if n_gpu > 1:
        n_jobs = args.n_jobs // n_gpu if args.n_jobs > n_gpu else 1
        mp_args = (yml_file, args.load_epoch, args.save_interval, args.valid, n_jobs, args.seed)
        mp.spawn(train, nprocs= n_gpu, args= mp_args, join=True)
    else:
        train(0, yml_file, args.load_epoch, args.save_interval, args.valid, args.n_jobs, args.seed)

    print(f'---{datetime.datetime.now()}: all process done. (elapsed time: {second2date(time.time()-start)})---', flush= True)














