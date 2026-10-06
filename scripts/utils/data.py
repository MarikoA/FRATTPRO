import pandas as pd
import argparse
import copy
import os
import sys
import subprocess
import uuid
import moses
from Bio import PDB
from pathlib import Path
from molvs import  Standardizer

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem, DataStructs

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from esm import FastaBatchedDataset, pretrained, MSATransformer

class fPLG_Dataset(Dataset):
    def __init__(self, frag_indices, positions, sequences, prop, esm) -> None:
        super().__init__()
        self.frag_indices = frag_indices
        self.positions = positions
        self.sequences = sequences
        self.prop = prop

        self.model, self.alphabet = pretrained.load_model_and_alphabet(esm)
        self.batch_converter = self.alphabet.get_batch_converter()

    def __len__(self):
        return len(self.frag_indices)
    
    def __getitem__(self, index):
        frag_idx = self.frag_indices[index]
        pos = self.positions[index]
        seq = self.sequences[index]  # ここがアミノ酸配列
        prop = self.prop[index]

        # ESM 用のトークン化（バッチ処理に対応）
        _, _, toks = self.batch_converter([(str(index), seq)])
        length = len(seq)
        return torch.tensor(frag_idx), torch.tensor(pos), torch.tensor(length), toks.squeeze(0), torch.tensor(prop)
    
class fPLG_fast_Dataset(Dataset):
    def __init__(self, frag_indices, positions, emb_seq, interactions) -> None:
        super().__init__()
        self.frag_indices = frag_indices
        self.positions = positions
        self.emb_seq = emb_seq
        self.interactions = interactions

    def __len__(self):
        return len(self.frag_indices)
    
    def __getitem__(self, index) -> [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.frag_indices[index], self.positions[index], self.emb_seq[index], self.interactions[index]

class DictProcessor(argparse.Action):
    def __call__(self, parser, namespace, values, option_strings=None):
        param_dict = getattr(namespace,self.dest,[])
        if param_dict is None:
            param_dict = {}

        k, v = values.split(":")
        param_dict[k] = int(v)
        setattr(namespace, self.dest, param_dict)

def get_emb(fasta_path, emb_path, toks_per_batch, esm, device):
    model, alphabet = pretrained.load_model_and_alphabet(esm)
    model.eval()
    if isinstance(model, MSATransformer):
        raise ValueError(
            "This script currently does not handle models with MSA input (MSA Transformer)."
        )
    model = model.to(device)

    dataset = FastaBatchedDataset.from_file(Path(fasta_path))
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
    )
    print(f"Read {fasta_path} with {len(dataset)} sequences", flush=True)

    Path(emb_path).mkdir(parents=True, exist_ok=True)

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in [-1])

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            if (batch_idx == 0) | ((batch_idx + 1) % 1000 == 0) | ((batch_idx + 1) == len(batches)):
                print(f"Processing batch {batch_idx + 1} of {len(batches)}", flush=True)
            if torch.cuda.is_available():
                toks = toks.to(device, non_blocking=True)
            out = model(toks, repr_layers=[model.num_layers - 1])  # Requesting only the last layer
            representations = out["representations"][model.num_layers - 1].to(device="cpu")

            for i, label in enumerate(labels):
                output_file = Path(emb_path) / f"{label}.pt"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                per_pro_vector = representations[i, 1:len(strs[i]) + 1].mean(0).clone()  # Clone to ensure it's not a view
                torch.save(per_pro_vector, output_file)

def load_vector(emb_path, idx):
    file_path = os.path.join(emb_path, f"{idx}.pt")
    if os.path.exists(file_path):
        vector = torch.load(file_path)
        return vector
    else:
        print(f'error: there is no such a file. (ID :{idx})', flush=True)
        sys.exit()

def check_columns_exist(df, columns_to_check):
    missing_columns = [column for column in columns_to_check if column not in df.columns]
    
    if missing_columns:
        raise ValueError(f"These column names cannot be found: {', '.join(missing_columns)}")
    
def collate_fn_fPLG(batch):
    frag_indices, positions, length, toks, prop = zip(*batch)
    frag_indices = pad_sequence(frag_indices, batch_first= True, padding_value= 0)
    positions = pad_sequence(positions, batch_first= True, padding_value= 0)
    length = torch.tensor(length)
    toks = pad_sequence(toks, batch_first= True, padding_value= 0)
    prop = torch.stack(prop)

    return frag_indices, positions, length, toks, prop

def collate_fn_fPLG_fast(batch):
    frag_indices, positions, vectors, prop = zip(*batch)
    frag_indices = pad_sequence(frag_indices, batch_first= True, padding_value= 0)
    positions = pad_sequence(positions, batch_first= True, padding_value= 0)
    vectors = pad_sequence(vectors, batch_first= True, padding_value= 0)
    prop = torch.stack(prop)

    return frag_indices, positions, vectors, prop

def clearAtomMapNums(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

def metrics(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        mol = copy.deepcopy(mol)
        clearAtomMapNums(mol)
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        # mol = stand.disconnect_metals(mol)
        mol = Standardizer().normalize(mol)
        mol = Standardizer().reionize(mol)
        # Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        smi_std = Chem.MolToSmiles(mol)

        natom = mol.GetNumAtoms()
        mw = moses.metrics.weight(mol)
        logP = moses.metrics.logP(mol)
        qed = moses.metrics.QED(mol)
        sa = moses.metrics.SA(mol)
        npl = moses.metrics.NP(mol)
        tpsa = Chem.Descriptors.TPSA(mol)
        ct = Descriptors.BertzCT(mol)
    except:
        mol = smi_std = None
        natom = mw = logP = qed = sa = npl = tpsa = ct =  None
    
    return smi, smi_std, natom, mw, logP, qed, sa, npl, tpsa, ct

def list2pdData(loss_list: list, metrics: list) -> pd.DataFrame:
    loss_dict = {}
    for key, values in zip(metrics, loss_list):
        loss_dict[key] = values
    return pd.DataFrame(loss_dict)

def expand_vectors(vectors, N):
    new_vectors = []
    
    for vector in vectors:
        new_vectors.extend([vector] * N)
    
    return new_vectors

def calculate_tanimoto_similarity(smiles_df, sdf_dir):
    results = []

    for _, row in smiles_df.iterrows():
        pdb_id = row['PDBIDs']
        smiles = row['smiles']

        # SMILESから分子オブジェクトを生成
        mol_smiles = Chem.MolFromSmiles(smiles)
        if mol_smiles is None:
            print(f"PDB {pdb_id}: SMILESの変換に失敗しました ({smiles})")
            continue

        # 分子指紋を生成
        fp_smiles = AllChem.GetMorganFingerprintAsBitVect(mol_smiles, radius=2)

        # 対応するSDFファイルのパスを生成
        sdf_path = os.path.join(sdf_dir, f"{pdb_id}/{pdb_id}_ligand.sdf")
        if not os.path.exists(sdf_path):
            print(f"PDB {pdb_id}: SDFファイルが見つかりません ({sdf_path})", flush=True)
            continue

        # SDFファイルから分子を読み込む
        sdf_supplier = Chem.SDMolSupplier(sdf_path)
        for sdf_mol in sdf_supplier:
            if sdf_mol is None:
                print(f"PDB {pdb_id}: SDFの分子読み込みに失敗しました")
                continue

            # SDF分子の指紋を生成
            fp_sdf = AllChem.GetMorganFingerprintAsBitVect(sdf_mol, radius=2)

            # Tanimoto類似度を計算
            similarity = DataStructs.TanimotoSimilarity(fp_smiles, fp_sdf)
            results.append({"PDBIDs": pdb_id, "SMILES": smiles, "Similarity": similarity})

    return pd.DataFrame(results)

def ProteinParser(pdbid, pro_file):
    parser = PDB.PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(pdbid, pro_file)
    ppb = PDB.PPBuilder()
    
    seq = ''
    for pp in ppb.build_peptides(structure):
        seq += pp.get_sequence()
    
    return seq

def CalculateAffinity(smi, file_protein='./1zys.pdb', file_lig_ref = './1zys_D_199.sdf', out_path = './', prefix='', smina_path='./smina', calc_ref=True):
    try:
        mol = Chem.MolFromSmiles(smi)
        m2=Chem.AddHs(mol)
        AllChem.EmbedMolecule(m2)
        m3 = Chem.RemoveHs(m2)
    except Exception as e:
        print(e, flush= True)
        affinity = None
        ref_affinity = None
        return affinity, ref_affinity
    unique_id = str(uuid.uuid4())
    file_genChem = os.path.join(out_path, prefix + unique_id + '.pdb')

    Chem.MolToPDBFile(m3, file_genChem)

        # file_drug="sdf_ligand_"+str(pdb_id)+str(i)+".sdf"
    smina_cmd_output = os.path.join(out_path, prefix + unique_id + ".out")  # 標準出力ファイル
    smina_error_output = os.path.join(out_path, prefix + unique_id + ".err")  # エラー出力ファイル

    # sminaのコマンドを作成
    launch_args = [
        smina_path,  # smina のパスを指定
        "-r", file_protein,
        "-l", file_genChem,
        "--autobox_ligand", file_lig_ref,
        "--autobox_add", "10",
        "--seed", "1000",
        "--exhaustiveness", "9"
    ]

    launch_string = ' '.join(launch_args) + f" >> {smina_cmd_output} 2>> {smina_error_output}"
    # print(launch_string, flush=True)
    p = subprocess.Popen(launch_string, shell=True)
    p.communicate()

    affinity = 500

    with open(smina_cmd_output, 'r') as f:
        for lines in f.readlines():
            lines = lines.split()
            if len(lines) == 4 and lines[0] == '1':
                affinity = float(lines[1])
    if affinity == 500:
        affinity = None
        print(f'affinity error: {unique_id}, smiles: {smi}', flush=True)
        files_to_remove = [file_genChem]
        for file in files_to_remove:
            if os.path.exists(file):
                os.remove(file)
    else: 
        files_to_remove = [file_genChem, smina_cmd_output, smina_error_output]
        for file in files_to_remove:
            if os.path.exists(file):
                os.remove(file)

    # 参考分子
    if calc_ref:
        # file_mol2_ref = os.path.splitext(file_lig_ref)[0] + '.mol2'
        smina_cmd_output_ref = os.path.join(out_path, prefix + unique_id + '_ref' + '.out')  # 標準出力ファイル
        smina_error_output_ref = os.path.join(out_path, prefix + unique_id + '_ref' + '.err')  # エラー出力ファイル
        if not os.path.exists(smina_cmd_output_ref):
            # sminaのコマンドを作成
            launch_args = [
                smina_path,  # smina のパスを指定
                "-r", file_protein,
                "-l", file_lig_ref,
                "--autobox_ligand", file_lig_ref,
                "--autobox_add", "10",
                "--seed", "1000",
                "--exhaustiveness", "9"
            ]

            launch_string = ' '.join(launch_args) + f" >> {smina_cmd_output_ref} 2>> {smina_error_output_ref}"
            # print(launch_string, flush=True)
            p = subprocess.Popen(launch_string, shell=True)
            p.communicate()

        ref_affinity = 500
        with open(smina_cmd_output_ref, 'r') as f:
            for lines in f.readlines():
                lines = lines.split()
                if len(lines) == 4 and lines[0] == '1':
                    ref_affinity = float(lines[1])

        if ref_affinity == 500:
            ref_affinity = None
            print(f'ref_affinity error: {unique_id}', flush=True)
        else: 
            files_to_remove = [smina_cmd_output_ref, smina_error_output_ref]
            for file in files_to_remove:
                if os.path.exists(file):
                    os.remove(file)
    else: ref_affinity = None
     
    return affinity, ref_affinity