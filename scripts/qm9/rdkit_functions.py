from rdkit import Chem
import numpy as np
from qm9.bond_analyze import get_bond_order, geom_predictor
from . import dataset
import torch, time, traceback
from configs.datasets_config import get_dataset_info
import pickle, json
import os, sys
from rdkit.Chem.QED import qed as qed_
from rdkit import RDConfig, Chem, RDLogger
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from rdkit.Chem import AllChem, rdDetermineBonds, DataStructs
from qm9.sdf2pdbqt import sdf2pdbqt
import subprocess, shutil
from datetime import datetime
# import shutil
# import os
sys.path.append("/data/lab_ph/kyle/projects/DrugDesign/uncertainty")
from get_pio import get_pio_score


RDLogger.DisableLog('rdApp.*')

def remove_dir(path2dir):
    if os.path.exists(path2dir) and os.path.isdir(path2dir):
        shutil.rmtree(path2dir)
        # print(f"Deleted directory: {path2dir}")
    else:
        print(f"Directory does not exist: {path2dir}")

def remove_file(file_path):
    if os.path.exists(file_path) and os.path.isfile(file_path):
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
    else:
        print(f"File does not exist: {file_path}")

def compute_qm9_smiles(dataset_name, remove_h):
    '''

    :param dataset_name: qm9 or qm9_second_half
    :return:
    '''
    print("\tConverting QM9 dataset to SMILES ...")

    class StaticArgs:
        def __init__(self, dataset, remove_h):
            self.dataset = dataset
            self.batch_size = 1
            self.num_workers = 1
            self.filter_n_atoms = None
            self.datadir = 'qm9/temp'
            self.remove_h = remove_h
            self.include_charges = True
    args_dataset = StaticArgs(dataset_name, remove_h)
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args_dataset)
    dataset_info = get_dataset_info(args_dataset.dataset, args_dataset.remove_h)
    n_types = 4 if remove_h else 5
    mols_smiles = []
    for i, data in enumerate(dataloaders['train']):
        positions = data['positions'][0].view(-1, 3).numpy()
        one_hot = data['one_hot'][0].view(-1, n_types).type(torch.float32)
        atom_type = torch.argmax(one_hot, dim=1).numpy()

        mol = build_molecule(torch.tensor(positions), torch.tensor(atom_type), dataset_info)
        mol = mol2smiles(mol)
        if mol is not None:
            mols_smiles.append(mol)
        if i % 1000 == 0:
            print("\tConverting QM9 dataset to SMILES {0:.2%}".format(float(i)/len(dataloaders['train'])))
    return mols_smiles

def read_smiles_from_smi(file_path):
    smiles_list = []
    with open(file_path, 'r') as f:
        line = f.readline()  # Read the first line
        while line:
            # Strip any extra spaces/newlines and get the entire SMILES string
            smiles = line.strip()  # No need to split if there is only SMILES on each line
            smiles_list.append(smiles)
            line = f.readline()  # Read the next line
    return smiles_list

def find_dir(dirname):
    current_dir = os.getcwd() 
    while current_dir != '/':
        possible_path = os.path.join(current_dir, dirname)
        if os.path.isdir(possible_path):
            return possible_path
        current_dir = os.path.dirname(current_dir)  
    else:
        return None

def retrieve_qm9_smiles(dataset_info):
    dataset_name = dataset_info['name']
    if dataset_info['with_h']:
        pickle_name = dataset_name
    else:
        pickle_name = dataset_name + '_noH'

    print(f"WithH: {dataset_info['with_h']}")
    
    root = os.getcwd()
    file_name = find_dir('datasets') + f"/smiles/{dataset_name}_removeH_{dataset_info['with_h']}_train.smi"
    
    qm9_smiles = read_smiles_from_smi(file_name)

    return qm9_smiles
    # file_name = 'qm9/temp/%s_smiles.pickle' % pickle_name
    # try:
    #     with open(file_name, 'rb') as f:
    #         qm9_smiles = pickle.load(f)
    #     return qm9_smiles
    # except OSError:
    #     try:
    #         os.makedirs('qm9/temp')
    #     except:
    #         pass
    #     qm9_smiles = compute_qm9_smiles(dataset_name, remove_h=not dataset_info['with_h'])
    #     with open(file_name, 'wb') as f:
    #         pickle.dump(qm9_smiles, f)
    #     return qm9_smiles

def remove_all_files(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error while deleting file {file_path}: {e}")

def get_diversity(valid, k=1):
    def mol_from_smiles(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return mol
        else:
            raise ValueError(f"Invalid SMILES: {smiles}")

    def compute_individual_diversity(fps):
        n = len(fps)
        diversity_scores = []
        for i in range(n):
            sims = []
            for j in range(n):
                if i == j:
                    continue
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                sims.append(sim)
            avg_sim = np.mean(sims)
            diversity = 1.0 - avg_sim  
            diversity_scores.append(diversity)
        return diversity_scores

    valid_smiles = []
    valid_indices = []

    for idx, smiles in enumerate(valid):
        if smiles is not None:
            valid_smiles.append(smiles)
            valid_indices.append(idx)

    mols = [mol_from_smiles(s) for s in valid_smiles]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]

    diversity_list = [float('-inf')] * len(valid)
    if len(valid_smiles) >= 2:
        diversity_scores = compute_individual_diversity(fps)
        for i, idx in enumerate(valid_indices):
            diversity_list[idx] = diversity_scores[i]
    else:
        for idx in valid_indices:
            diversity_list[idx] = 0.0
    return diversity_list
    
    # seen_counts = {}
    # weights = []
    # for s in valid_smiles:
    #     seen_counts[s] = seen_counts.get(s, 0) + 1
    #     n = seen_counts[s]
    #     weight = np.exp(-k * (n - 1))
    #     weights.append(weight)
    # diversity_list = [float('-inf')] * len(valid)
    # if len(valid_smiles) >= 2:
    #     diversity_scores = compute_individual_diversity(fps)
    #     for i, idx in enumerate(valid_indices):
    #         diversity_list[idx] = diversity_scores[i] * weights[i]
    # else:
    #     for idx in valid_indices:
    #         diversity_list[idx] = 0.0
    # return diversity_list

#### New implementation ####

def compute_cutoffs(property_dict, quantile=50):
    high_good = {'qed': True, 'sas': False, 'affinity': False}
    
    cutoff_dict = {}
    for prop, values in property_dict.items():
        values = np.array(values)
        values = values[np.isfinite(values)]

        if high_good.get(prop, True):
            cutoff = np.percentile(values, 100 - quantile)
        else:
            cutoff = np.percentile(values, quantile)
        cutoff_dict[prop] = cutoff
    
    return cutoff_dict

bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]


class BasicMolecularMetrics(object):
    def __init__(self, dataset_info, dataset_smiles_list=None):
        self.atom_decoder = dataset_info['atom_decoder']
        self.dataset_smiles_list = dataset_smiles_list
        self.dataset_info = dataset_info

        # Retrieve dataset smiles only for qm9 currently.
        if dataset_smiles_list is None:   # and 'qm9' in dataset_info['name']:
            self.dataset_smiles_list = retrieve_qm9_smiles(
                self.dataset_info)
        
        # ------------------------------------------------
        self.novel_molecules = None
        self.clean_temp_files = True
        # ------------------------------------------------

    def compute_validity_v1(self, generated):
        """ generated: list of couples (positions, atom_types)"""
        valid = []
        atom_decoder = self.dataset_info["atom_decoder"]

        # clean the temp directories
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dir_sdf = f'/data/lab_ph/kyle/projects/DrugDesign/vina_gpu_utils/dockingResults/evaluation_sdf_{timestamp}'
        dir_pdbqt = f'/data/lab_ph/kyle/projects/DrugDesign/vina_gpu_utils/dockingResults/evaluation_{timestamp}'
        output_directory = f'/data/lab_ph/kyle/projects/DrugDesign/vina_gpu_utils/dockingResults/evaluation_out_{timestamp}'
        print(f'dir_sdf: {dir_sdf}')
        print(f'dir_pdbqt: {dir_pdbqt}')
        print(f'output_directory: {output_directory}')
        os.makedirs(dir_sdf, exist_ok=True)
        os.makedirs(dir_pdbqt, exist_ok=True)
        os.makedirs(output_directory, exist_ok=True)
        for directory in [dir_sdf, dir_pdbqt, output_directory]:
            remove_all_files(directory)
        config_content = f"""receptor = /data/lab_ph/kyle/projects/DrugDesign/mockData/6vhn_prepared.pdbqt
                            ligand_directory = {dir_pdbqt}
                            opencl_binary_path = /data/lab_ph/kyle/projects/DrugDesign/Vina-GPU-2.1/QuickVina2-GPU-2.1
                            center_x = -59.460
                            center_y = 8.303
                            center_z = 25.145
                            size_x = 30
                            size_y = 30
                            size_z = 30
                            thread = 8000
                            seed = 2024
                            output_directory = {output_directory}
                            """
        self.clean_temp_files = False
        for idx, graph in enumerate(generated):
            # ------------------ 1 ------------------
            try:
                # form xyz block
                if self.dataset_info["name"] == 'pubchem':
                    raise False
                smiles = None
                ind = []
                pos, atom_type = graph
                pos, atom_type = pos.tolist(), atom_type.tolist()

                nAts = len(atom_type)
                ind.append(f"{nAts}\n")
                ind.append("\n")

                for i in range(nAts):
                    atom = atom_decoder[atom_type[i]]
                    x, y, z = pos[i]
                    ind.append(f"{atom}\t{x:.10f}\t{y:.10f}\t{z:.10f}\n")
                ind = ''.join(ind)

                mol = Chem.MolFromXYZBlock(ind)

                rdDetermineBonds.DetermineBonds(mol)

                Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)

                Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

                smiles = Chem.MolToSmiles(mol, canonical=True)

            except:
            # ------------------ 1 ------------------
                mol = build_molecule(*graph, self.dataset_info)  # mova ahead
                smiles = mol2smiles(mol)  # mova ahead

            if smiles is not None:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())

                # ------------------ 2 ------------------
                try:
                    Chem.SanitizeMol(largest_mol)
                    smiles = Chem.MolToSmiles(largest_mol, canonical=True) # smiles = mol2smiles(largest_mol)

                    largest_mol_withH = Chem.AddHs(largest_mol)
                    Chem.SanitizeMol(largest_mol_withH)
                    
                    # convert to sdf
                    sdf_filename = os.path.join(dir_sdf, f'{str(idx).zfill(6)}.sdf')
                    pdbqt_filename = os.path.join(dir_pdbqt, f'{str(idx).zfill(6)}.pdbqt')
                    Chem.MolToMolFile(largest_mol_withH, sdf_filename) # to sdf

                    # convert to pdbqt
                    sdf2pdbqt(sdf_filename, pdbqt_filename)
                    # ------------------ 2 ------------------

                    valid.append(smiles)
                except:
                    # continue
                    valid.append(None)
            else:
                valid.append(None)

        assert len(generated) == len(valid), "Length unmatched"
        # return 0,0,0,0,0,0
        # docking        
        working_directory = "/data/lab_ph/kyle/projects/DrugDesign/Vina-GPU-2.1/QuickVina2-GPU-2.1"  
        config_file = f'/data/lab_ph/kyle/projects/DrugDesign/Vina-GPU-2.1/QuickVina2-GPU-2.1/configs/evaluation_config_{timestamp}.txt'
        # create a temp config file
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            f.write(config_content)
        command = f"./QuickVina2-GPU-2-1 --config {config_file}"  

        # -------------------------------------------------------------- 
        try:
            # subprocess.run(command, shell=True, check=True, cwd=working_directory)
            subprocess.run(command, shell=True, check=True, cwd=working_directory, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            print('Fail docking')
            # return 0, 0, 0, 0, 0, 0   # not sure if this is right ???
        # keep_try_flag = True
        # while keep_try_flag:
        #     try:
        #         subprocess.run(command, shell=True, check=True, cwd=working_directory)
        #         keep_try_flag = False
        #     except:
        #         error_message = traceback.format_exc()
        #         print(f"Handled Error message: \n{error_message}")
        #         time.sleep(10*60)
        #         print(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}: Waited 10 mins")
        # ---------------------------------------------------------------

        # get valid idx
        all_processed_pdbqt = sorted(os.listdir(output_directory))
        valid_idx = []
        for i in all_processed_pdbqt:
            valid_idx.append(int(i.replace('_out.pdbqt','')))
        valid_idx = sorted(valid_idx)
        filtered_valid = [valid[i] for i in valid_idx]
        valid = filtered_valid
        # print(valid)
        assert None not in valid, "None in Valid List"
        
        # collect docking scores
        affinity = []
        pdbqt_path = []
        for pdbqt in all_processed_pdbqt:
            if pdbqt.endswith('.pdbqt'):
                pdbqt_fullPath = os.path.join(output_directory, pdbqt)
                with open(pdbqt_fullPath, 'r') as f:
                    _ = f.readline().strip()
                    affinity.append(float(f.readline().strip().split()[3]))
                    pdbqt_path.append(pdbqt_fullPath)
        
        assert len(affinity) == len(valid), "Length unmatched"

        # validity
        validity = len(valid) / len(generated)

        if validity > 0:
            # uniqueness
            smiles_with_affinity = {} 
            smiles_with_all = {}
            for i in range(len(valid)):
                smiles_with_affinity[valid[i]] = affinity[i]
                smiles_with_all[valid[i]] = {'qed':-1, 'sas':-1, 'affinity':affinity[i], 'path':pdbqt_path[i]}
            uniqueness = len(smiles_with_affinity) / len(valid)

            # novelty
            num_novel = 0
            novel = []
            qed = []
            sas = []
            affinity_ = []
            for smiles, affi in smiles_with_affinity.items():
                if smiles not in self.dataset_smiles_list:
                    novel.append(smiles)
                    num_novel += 1
                    mol = Chem.MolFromSmiles(smiles)
                    mol = Chem.AddHs(mol)

                    qed_score = qed_(mol)
                    sas_score = sascorer.calculateScore(mol)
                    
                    qed.append(qed_score)
                    sas.append(sas_score)
                    affinity_.append(affi)

                    smiles_with_all[smiles]['qed'] = qed_score
                    smiles_with_all[smiles]['sas'] = sas_score
                else:
                    smiles_with_all.pop(smiles) #????

            novelty = num_novel / len(smiles_with_affinity)

            # copy top molecules to other places, especially pdbqt files
            sorted_mol_list = sorted(smiles_with_all.items(), key=lambda item: item[1]['affinity'], reverse=False)
            self.novel_molecules = dict(sorted_mol_list)
            
            # ------------------------------------------------
            cut=2000
            top20_molecules = dict(sorted_mol_list[:cut])
            # ------------------------------------------------
            
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            path4top20 = f'/data/lab_ph/kyle/projects/DrugDesign/results/top_molecules/{current_time}'
            os.makedirs(path4top20, exist_ok=True)

            print(f'Top {cut} molecules based on affinity:')
            for idx, (smiles, values) in enumerate(top20_molecules.items()):
                print(f'Top {idx+1}: {values}, smiles: {smiles}')
                source_path = values['path']
                file_name = f'{idx}_' + os.path.basename(source_path)
                destination_path = os.path.join(path4top20, file_name)

                try:
                    shutil.copy(source_path, destination_path)
                except Exception as e:
                    print(f"Failed to copy {file_name}: {e}")
            print(f'Top molecules were saved to: {path4top20}')

            # print(f'qed: {qed}')
            # print(f'sas: {sas}')
            # print(f'affinity: {affinity_}')

            # clean temp files and dirs       
            if self.clean_temp_files:     
                print(f'Clean temp files/dirs: {self.clean_temp_files}')
                remove_dir(dir_sdf)
                remove_dir(dir_pdbqt)
                remove_dir(output_directory)
                remove_dir(path4top20)
                remove_file(config_file)

            print(f'qed: {qed}')
            print(f'sas: {sas}')
            print(f'affinity: {affinity_}')
            return validity, uniqueness, novelty, np.mean(qed), np.mean(sas), np.mean(affinity_)
        else:
            print('Validity is ZERO')
            if self.clean_temp_files:    
                print(f'Clean temp files/dirs: {self.clean_temp_files}') 
                remove_dir(dir_sdf)
                remove_dir(dir_pdbqt)
                remove_dir(output_directory)
                remove_file(config_file)

            return 0, 0, 0, 0, 0, 0

        # return valid, len(valid) / len(generated)

    def compute_validity_v2(self, generated, k=1, hyper_paras=None):
        """ generated: list of couples (positions, atom_types)"""
        valid = []
        atom_decoder = self.dataset_info["atom_decoder"]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dir_sdf = f'/data/lab_ph/kyle/projects/DrugDesign/vina_gpu_utils/dockingResults/evaluation_sdf_{timestamp}'
        dir_pdbqt = f'/data/lab_ph/kyle/projects/DrugDesign/vina_gpu_utils/dockingResults/evaluation_{timestamp}'
        output_directory = f'/data/lab_ph/kyle/projects/DrugDesign/vina_gpu_utils/dockingResults/evaluation_out_{timestamp}'
        os.makedirs(dir_sdf, exist_ok=True)
        os.makedirs(dir_pdbqt, exist_ok=True)
        os.makedirs(output_directory, exist_ok=True)
        for directory in [dir_sdf, dir_pdbqt, output_directory]:
            remove_all_files(directory)
        config_content = f"""receptor = /data/lab_ph/kyle/projects/DrugDesign/mockData/6vhn_prepared.pdbqt
                            ligand_directory = {dir_pdbqt}
                            opencl_binary_path = /data/lab_ph/kyle/projects/DrugDesign/Vina-GPU-2.1/QuickVina2-GPU-2.1
                            center_x = -59.460
                            center_y = 8.303
                            center_z = 25.145
                            size_x = 30
                            size_y = 30
                            size_z = 30
                            thread = 8000
                            seed = 2024
                            output_directory = {output_directory}
                            """
        
        for idx, graph in enumerate(generated):
            try:
                if not hyper_paras['enable_blackbox_restore']:
                    raise False
                smiles = None
                ind = []
                pos, atom_type = graph
                pos, atom_type = pos.tolist(), atom_type.tolist()
                nAts = len(atom_type)
                ind.append(f"{nAts}\n")
                ind.append("\n")
                for i in range(nAts):
                    atom = atom_decoder[atom_type[i]]
                    x, y, z = pos[i]
                    ind.append(f"{atom}\t{x:.10f}\t{y:.10f}\t{z:.10f}\n")
                ind = ''.join(ind)
                mol = Chem.MolFromXYZBlock(ind)
                rdDetermineBonds.DetermineBonds(mol)
                Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
                Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
                smiles = Chem.MolToSmiles(mol, canonical=True)
            except:
                mol = build_molecule(*graph, self.dataset_info)
                smiles = mol2smiles(mol)

            if smiles is not None:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                try:
                    Chem.SanitizeMol(largest_mol)
                    smiles = Chem.MolToSmiles(largest_mol, canonical=True)
                    largest_mol_withH = Chem.AddHs(largest_mol)
                    Chem.SanitizeMol(largest_mol_withH)
                    sdf_filename = os.path.join(dir_sdf, f'{str(idx).zfill(6)}.sdf')
                    pdbqt_filename = os.path.join(dir_pdbqt, f'{str(idx).zfill(6)}.pdbqt')
                    Chem.MolToMolFile(largest_mol_withH, sdf_filename)
                    sdf2pdbqt(sdf_filename, pdbqt_filename)
                    valid.append(smiles)
                except:
                    valid.append(None)
            else:
                valid.append(None)

        assert len(generated) == len(valid), "Length unmatched"

        # 初始化属性和状态列表
        qed_list = [float('-inf')] * len(generated)
        sas_list = [float('-inf')] * len(generated)
        affinity_list = [float('-inf')] * len(generated)

        diversity_list = get_diversity(valid, k=k)  

        valid_list = [False] * len(generated)
        unique_list = [False] * len(generated)
        novel_list = [False] * len(generated)

        # 计算所有有效分子的 QED 和 SAS
        # smiles_dict = {}
        # seen_smiles = set()
        # for idx, smiles in enumerate(valid):
        #     if smiles is not None:
        #         valid_list[idx] = True
        #         mol = Chem.MolFromSmiles(smiles)
        #         mol = Chem.AddHs(mol)
        #         qed_list[idx] = qed_(mol)    #### 
        #         sas_list[idx] = sascorer.calculateScore(mol)   ####
        #         smiles_dict[smiles] = idx
        #         unique_list[idx] = smiles not in seen_smiles
        #         novel_list[idx] = smiles not in self.dataset_smiles_list
        #         seen_smiles.add(smiles)

        # Docking
        working_directory = "/data/lab_ph/kyle/projects/DrugDesign/Vina-GPU-2.1/QuickVina2-GPU-2.1"
        config_file = f'/data/lab_ph/kyle/projects/DrugDesign/Vina-GPU-2.1/QuickVina2-GPU-2.1/configs/evaluation_config_{timestamp}.txt'
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            f.write(config_content)
        command = f"./QuickVina2-GPU-2-1 --config {config_file}"
        
        try:
            # subprocess.run(command, shell=True, check=True, cwd=working_directory)
            subprocess.run(command, shell=True, check=True, cwd=working_directory, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # capture_output=True)
            all_processed_pdbqt = sorted(os.listdir(output_directory))
            valid_idx = [int(i.replace('_out.pdbqt', '')) for i in all_processed_pdbqt]
            valid_idx = sorted(valid_idx)
            
            # 更新 affinity
            for pdbqt in all_processed_pdbqt:
                if pdbqt.endswith('.pdbqt'):
                    idx = int(pdbqt.replace('_out.pdbqt', ''))
                    pdbqt_fullPath = os.path.join(output_directory, pdbqt)
                    with open(pdbqt_fullPath, 'r') as f:
                        _ = f.readline().strip()
                        affinity_list[idx] = float(f.readline().strip().split()[3])
        except Exception as e:
            print(f"Docking failed: {e}")  # 记录失败原因


        smiles_dict = {}
        seen_smiles = set()
        for idx, smiles in enumerate(valid):
            if smiles is not None:
                valid_list[idx] = True
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)
                qed_list[idx] = qed_(mol)    #### 
                sas_list[idx] = sascorer.calculateScore(mol)   ####
                smiles_dict[smiles] = idx
                unique_list[idx] = smiles not in seen_smiles
                novel_list[idx] = smiles not in self.dataset_smiles_list
                seen_smiles.add(smiles)

        # ----------------------------------------- fit start ---
        property_list = hyper_paras['property_list'] # ['qed', 'sas', 'affinity']
        if hyper_paras['new_cutoff'] != None:
            new_cutoff = hyper_paras['new_cutoff']
        else:
            new_cutoff = None
             # hyper_paras['new_cutoff'] # {  
        #     'qed' : 0.47, 
        #     'sas' : 6.48, 
        #     'affinity' : -4.6, 
        # }
        # print(f'property_list: {property_list}')
        # print(f'new_cutoff: {new_cutoff}')
        # print(f'qed: {qed_list}')
        # print(f'sas: {sas_list}')
        # print(f'affinity: {affinity_list}')
        
        # property_dict = {'qed' : qed_list,
        #                  'sas' : sas_list,
        #                  'affinity': affinity_list,}
        # compute_cutoffs(property_dict, quantile=50)

        args_fitness = {'valid_smile_list' : valid, 
                        'dataset' : self.dataset_info['name'], 
                        'property_list' : property_list, 
                        'mode' : 'evidential', 
                        'timestamp' : timestamp, 
                        'fitness' : hyper_paras['uncertainty_method'],  # 'pio', 
                        'new_cutoff' : new_cutoff, 
                        'clean_temp_files' : True,
                        'property_lists': {'qed' : qed_list,
                                           'sas' : sas_list,
                                           'affinity': affinity_list,}
                        }
        script_path = '/data/lab_ph/kyle/projects/DrugDesign/uncertainty/get_pio.py'

        try:
            raw_output = subprocess.check_output([
                "conda", "run", "-n", "chemprop", "python3", script_path, json.dumps(args_fitness)
            ])
            # result = json.loads(raw_output.decode().strip())   #######
            # print("Result list from chemprop:", result)
            # fitness_list = get_pio_score(valid, self.dataset_info['name'], property_list, mode='evidential', timestamp=timestamp, fitness='pio', new_cutoff=None, clean_temp_files=True)
            

            # ------ changed

            # fitness_list = json.loads(raw_output.decode().strip())

            result = json.loads(raw_output.decode().strip())
            fitness_list = result["overall_fitness"]
            new_cutoff = result["new_cutoff"]
            print(f'new_cutoff: {new_cutoff}')
            # ------ changed end
        except:
            print('Unhandled error occurred!')
            fitness_list = [0.0] * len(valid)
        # ----------------------------------------- fit end ---



        # 计算统计指标
        validity = sum(valid_list) / len(generated)
        uniqueness, novelty = 0, 0
        if validity > 0:
            unique_count = sum(unique_list)
            uniqueness = unique_count / sum(valid_list) if sum(valid_list) > 0 else 0
            novelty = sum(novel_list) / sum(valid_list) if sum(valid_list) > 0 else 0
            
            # Top 20 molecules  
            smiles_with_all = {smiles: {'qed': qed_list[smiles_dict[smiles]], 
                                    'sas': sas_list[smiles_dict[smiles]], 
                                    'affinity': affinity_list[smiles_dict[smiles]], 
                                    'fitness': fitness_list[smiles_dict[smiles]],      ###############
                                    'diversity': diversity_list[smiles_dict[smiles]],  
                                    } 
                            for smiles in smiles_dict} 
            
            top_flag = hyper_paras['top_flag']  # 'fitness'
            num_top_mol = hyper_paras['num_top_mol']  # 20
            sorted_mol_list = sorted(smiles_with_all.items(), key=lambda item: item[1][top_flag] if item[1][top_flag] != float('-inf') else float('inf'), reverse=True)
            top20_molecules = dict(sorted_mol_list[:num_top_mol])
            
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            path4top20 = f'/data/lab_ph/kyle/projects/DrugDesign/results/top_molecules/{current_time}'
            os.makedirs(path4top20, exist_ok=True)

            print(f'Top {num_top_mol} molecules based on {top_flag}:')
            for idx, (smiles, values) in enumerate(top20_molecules.items()):
                if idx < num_top_mol :   # only print(top 5)
                    print(f'Top {idx+1}: {values}, smiles: {smiles}')
            print(f'Top molecules saved to: {path4top20}')

        if self.clean_temp_files:
            
            try:
                remove_dir(dir_sdf)
                remove_dir(output_directory)
                remove_file(config_file)
                remove_dir(dir_pdbqt)
                remove_dir(path4top20)
            except:
                pass

        # print('in rdkit_funtion:')
        # print('1.', validity, uniqueness, novelty)
        # print('2,', qed_list, sas_list, affinity_list)
        # print('3.', valid_list, unique_list, novel_list)

        return validity, uniqueness, novelty, \
                np.mean([x for x in qed_list if x != float('-inf')]), \
                np.mean([x for x in sas_list if x != float('-inf')]), \
                np.mean([x for x in affinity_list if x != float('-inf')]), \
                np.mean([x for x in diversity_list if x != float('-inf')]), \
                [qed_list, sas_list, affinity_list, diversity_list, fitness_list, valid_list, unique_list, novel_list]

    def compute_validity(self, generated, RL=False, k=1, hyper_paras=None):
        if RL:
            return self.compute_validity_v2(generated, k=k, hyper_paras=hyper_paras)  # use RL
        else:
            return self.compute_validity_v1(generated)  # only diffusion

    def get_novel_molecules(self):
        return self.novel_molecules

    def compute_uniqueness(self, valid):
        """ valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        qed = []
        sas = []
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        # ------------------------------------
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)
                qed.append(qed_(mol))
                sas.append(sascorer.calculateScore(mol))
        # ------------------------------------
        return novel, num_novel / len(unique), np.mean(qed), np.mean(sas)

    def evaluate(self, generated, RL=False, k=1, hyper_paras=None):
        """ generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked. """
        # valid, validity = self.compute_validity(generated)
        # print(f"Validity over {len(generated)} molecules: {validity * 100 :.3f}%")
        # if validity > 0:
        #     unique, uniqueness = self.compute_uniqueness(valid)
        #     print(f"Uniqueness over {len(valid)} valid molecules: {uniqueness * 100 :.3f}%")

        #     if self.dataset_smiles_list is not None:
        #         _, novelty, qed, sas = self.compute_novelty(unique)
        #         print(f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.3f}%")
        #     else:
        #         novelty = 0.0
        # else:
        #     novelty = 0.0
        #     uniqueness = 0.0
        #     unique = None

        if RL:
            return self.compute_validity(generated, RL, k=k, hyper_paras=hyper_paras), []
        else:
            validity, uniqueness, novelty, qed, sas, affinity = self.compute_validity(generated)
            unique = []
            return [validity, uniqueness, novelty, qed, sas, affinity], unique

def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


def build_molecule(positions, atom_types, dataset_info):
    atom_decoder = dataset_info["atom_decoder"]
    X, A, E = build_xae_molecule(positions, atom_types, dataset_info)
    mol = Chem.RWMol()
    # ------------------ 1 ------------------
    positions = positions.tolist()
    conformer = Chem.Conformer(len(positions))
    # ------------------ 1 ------------------
    for i, atom in enumerate(X):
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
        # ------------------ 2 ------------------
        conformer.SetAtomPosition(i, (positions[i][0], positions[i][1], positions[i][2]))
        # ------------------ 2 ------------------
    
    # ------------------ 3 ------------------
    mol.AddConformer(conformer)
    # ------------------ 3 ------------------

    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[E[bond[0], bond[1]].item()])
    
    # --------------------
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_SETAROMATICITY) # aromaticity
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)  # stereo
    except:
        pass
    # --------------------
    return mol


def build_xae_molecule(positions, atom_types, dataset_info):
    """ Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
        args:
        positions: N x 3  (already masked to keep final number nodes)
        atom_types: N
        returns:
        X: N         (int)
        A: N x N     (bool)                  (binary adjacency matrix)
        E: N x N     (int)  (bond type, 0 if no bond) such that A = E.bool()
    """
    atom_decoder = dataset_info['atom_decoder']
    n = positions.shape[0]
    X = atom_types
    A = torch.zeros((n, n), dtype=torch.bool)
    E = torch.zeros((n, n), dtype=torch.int)

    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    for i in range(n):
        for j in range(i):
            pair = sorted([atom_types[i], atom_types[j]])
            order = get_bond_order(atom_decoder[pair[0]], atom_decoder[pair[1]], dists[i, j])
            # if dataset_info['name'] == 'qm9' or dataset_info['name'] == 'qm9_second_half' or dataset_info['name'] == 'qm9_first_half':
            #     order = get_bond_order(atom_decoder[pair[0]], atom_decoder[pair[1]], dists[i, j])
            # elif dataset_info['name'] == 'geom':
            #     order = geom_predictor((atom_decoder[pair[0]], atom_decoder[pair[1]]), dists[i, j], limit_bonds_to_one=True)
            # TODO: a batched version of get_bond_order to avoid the for loop
            if order > 0:
                # Warning: the graph should be DIRECTED
                A[i, j] = 1
                E[i, j] = order
    return X, A, E

if __name__ == '__main__':
    smiles_mol = 'C1CCC1'
    print("Smiles mol %s" % smiles_mol)
    chem_mol = Chem.MolFromSmiles(smiles_mol)
    block_mol = Chem.MolToMolBlock(chem_mol)
    print("Block mol:")
    print(block_mol)

