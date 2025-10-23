import sys, os, torch
from rdkit import RDConfig, Chem
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from tqdm import tqdm
from rdkit.Chem import QED
from collections import Counter


"""
Get all training data
Before this, need to have all molecule sdf file, docking results
"""

charge_dict = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, \
'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, \
'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Br': 35, 'I': 53, \
'As':33, 'Bi':83, 'U':92, \
'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, \
'Ga': 31, 'Ge': 32, 'Se': 34, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, \
'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, \
'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, \
'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, \
'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Po': 84, \
'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'Np': 93, 'Pu': 94, 'Am': 95, \
'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, \
'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, \
'Lv': 116, 'Ts': 117, 'Og': 118
}


def collect(dataName:str,remove_h:bool=False):
    """
    collect structure data and property data and save them together
    """
    
    processed_pdbqt_dir = f'/data/lab_ph/kyle/projects/DrugDesign/vina_gpu_utils/dockingResults/{dataName}_out'
    all_processed_pdbqt = os.listdir(processed_pdbqt_dir)

    original_sdf_dir = f'/data/lab_ph/kyle/projects/DrugDesign/datasets/{dataName}/single_mol_sdf'
    # all_original_sdf = os.listdir(original_sdf_dir)

    molecules = []
    all_atoms = []
    max_num_atoms = -1
    unique_atoms = set()
    for pdbqt in all_processed_pdbqt:
        if pdbqt.endswith('.pdbqt'):
            try:
                # get Vina measured binding affinities
                pdbqt_fullPath = os.path.join(processed_pdbqt_dir, pdbqt)
                with open(pdbqt_fullPath, 'r') as f:
                    _ = f.readline().strip()
                    affinity = float(f.readline().strip().split()[3])

                # load cooresponding sdf file
                sdf_fullPath = os.path.join(original_sdf_dir, pdbqt.replace('.pdbqt', '.sdf'))
                supplier = Chem.SDMolSupplier(sdf_fullPath, removeHs=remove_h)
                mol = next(iter(supplier))
                if mol is None: continue

                # if remove_h:
                #     mol = Chem.RemoveHs(mol)

                # get qed and sas scores
                qed_value = round(QED.qed(mol),1)
                sas_value = round(sascorer.calculateScore(mol),1)

                # get structure information
                num_atoms = mol.GetNumAtoms()
                max_num_atoms = num_atoms if num_atoms > max_num_atoms else max_num_atoms
                atoms = []
                atom_charges = []
                for atom in mol.GetAtoms():
                    symbol = atom.GetSymbol()

                    if remove_h and symbol == 'H':
                        continue

                    atoms.append(symbol)
                    atom_charges.append(charge_dict[symbol])
                unique_atoms.update(atoms)
                all_atoms.append(atoms)
                atom_positions = mol.GetConformers()[0].GetPositions().tolist()

                # Form molecules for training
                molecule = {'num_atoms': num_atoms, 'charges': atom_charges, 'positions': atom_positions,
                            'qed': qed_value, 'sas': sas_value, 'affinity': affinity}

                molecule = {key: torch.tensor(val) for key, val in molecule.items()}
                molecules.append(molecule)
            except:
                print(f'---------- Error: {pdbqt} ----------')
        
        # if len(molecules) == 3:
        #     break
    
    # save the whole dataset
    torch.save(molecules, f'/data/lab_ph/kyle/projects/DrugDesign/datasets/trainData/{dataName}_removeH_{remove_h}.pt')
    print(f'Total molecules: {len(molecules)}')
    print(f'Max num atoms: {max_num_atoms}')
    print(f'Unique atoms: {unique_atoms}')
    
    # configs
    atom_decoder = sorted(unique_atoms, key=lambda x: charge_dict[x])
    atom_encoder = {element: index for index, element in enumerate(atom_decoder)}
    n_nodes = Counter(len(molecule) for molecule in all_atoms)
    flat_atom_list = [atom for sublist in all_atoms for atom in sublist]
    atom_types = Counter(atom_encoder[atom] for atom in flat_atom_list)
    max_n_nodes = max(n_nodes.keys()) if n_nodes else 0

    configs = {
        'name': dataName,
        'atom_encoder': atom_encoder,
        'atom_decoder': atom_decoder,
        'n_nodes': n_nodes,
        'max_n_nodes': max_n_nodes,
        'atom_types': atom_types,
        'with_h': remove_h
    }
    print(configs)

    print(f'Done - {dataName}_removeH_{remove_h}!')


def addSmiles(dataName:str,remove_h:bool=False):
    processed_pdbqt_dir = f'/data/lab_ph/kyle/projects/DrugDesign/vina_gpu_utils/dockingResults/{dataName}_out'
    all_processed_pdbqt = os.listdir(processed_pdbqt_dir)

    original_sdf_dir = f'/data/lab_ph/kyle/projects/DrugDesign/datasets/{dataName}/single_mol_sdf'

    pt_file_path = f'/data/lab_ph/kyle/projects/DrugDesign/datasets/trainData/{dataName}_removeH_{remove_h}.pt'
    molecules = torch.load(pt_file_path)

    assert len(molecules)==len(all_processed_pdbqt), 'Length mismatch'

    for idx, pdbqt in enumerate(tqdm(all_processed_pdbqt)):
        if pdbqt.endswith('.pdbqt'):
            try:
                # load cooresponding sdf file
                sdf_fullPath = os.path.join(original_sdf_dir, pdbqt.replace('.pdbqt', '.sdf'))
                supplier = Chem.SDMolSupplier(sdf_fullPath, removeHs=remove_h)
                mol = next(iter(supplier))
                
                # ------------------------------------------
                # get smiles; dict is ordered so add to original dict
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol) 
                    molecules[idx]['smiles'] = smiles
                else:
                    continue
                # ------------------------------------------
                    
            except:
                print(f'---------- Error: {pdbqt} ----------')

    torch.save(molecules, f'/data/lab_ph/kyle/projects/DrugDesign/datasets/trainData_with_smiles/{dataName}_removeH_{remove_h}.pt')
    print(f'Added SMILES to - {dataName}_removeH_{remove_h}!')


def addMol(dataName:str,remove_h:bool=False):
    processed_pdbqt_dir = f'/data/lab_ph/kyle/projects/DrugDesign/vina_gpu_utils/dockingResults/{dataName}_out'
    all_processed_pdbqt = os.listdir(processed_pdbqt_dir)

    original_sdf_dir = f'/data/lab_ph/kyle/projects/DrugDesign/datasets/{dataName}/single_mol_sdf'

    pt_file_path = f'/data/lab_ph/kyle/projects/DrugDesign/datasets/trainData/{dataName}_removeH_{remove_h}.pt'
    molecules = torch.load(pt_file_path)

    assert len(molecules)==len(all_processed_pdbqt), 'Length mismatch'

    for idx, pdbqt in enumerate(tqdm(all_processed_pdbqt)):
        if pdbqt.endswith('.pdbqt'):
            try:
                # load cooresponding sdf file
                sdf_fullPath = os.path.join(original_sdf_dir, pdbqt.replace('.pdbqt', '.sdf'))
                supplier = Chem.SDMolSupplier(sdf_fullPath, removeHs=remove_h)
                mol = next(iter(supplier))
                
                # ------------------------------------------
                # get smiles; dict is ordered so add to original dict
                if mol is not None:
                    # smiles = Chem.MolToSmiles(mol) 
                    # molecules[idx]['smiles'] = smiles
                    molecules[idx]['mol'] = mol
                else:
                    continue
                # ------------------------------------------
                    
            except:
                print(f'---------- Error: {pdbqt} ----------')

    torch.save(molecules, f'/data/lab_ph/kyle/projects/DrugDesign/datasets/trainData_with_mol/{dataName}_removeH_{remove_h}.pt')
    print(f'Added SMILES to - {dataName}_removeH_{remove_h}!')


if __name__ == "__main__":

    dataNames = ['qm9','zinc15','geom','pubchem'] 

    for dataName in dataNames:
        print(f'------------------- {dataName} removeH {True} -------------------')
        collect(dataName, remove_h=True)
        print(f'------------------- {dataName} removeH {False} -------------------')
        collect(dataName, remove_h=False)

 
    for dataName in dataNames:
        print(f'Start - {dataName} - RemoveH: {False}')
        addSmiles(dataName,remove_h=False)
        print(f'Start - {dataName} - RemoveH: {True}')
        addSmiles(dataName,remove_h=True)

    for dataName in dataNames:
        print(f'Start - {dataName} - RemoveH: {False}')
        addMol(dataName,remove_h=False)
        print(f'Start - {dataName} - RemoveH: {True}')
        addSmiles(dataName,remove_h=True)


    print(f'Done all {dataNames} datasets')



    