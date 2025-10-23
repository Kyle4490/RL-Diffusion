from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm
import os
import tarfile

# logger = RDLogger.logger()
# logger.setLevel(RDLogger.CRITICAL)

# sdf_file = "/data/lab_ph/kyle/projects/DrugDesign/datasets/pubchem/PubChem.sdf"
# sdf_file = "/data/lab_ph/kyle/projects/DrugDesign/datasets/zinc15/zinc15.sdf"


# suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
# output_dir = "/data/lab_ph/kyle/projects/DrugDesign/datasets/zinc15/molecules"

# count = 0
# for idx, mol in tqdm(enumerate(suppl)):
#     if mol is None:
#         continue 
#     output_filename = os.path.join(output_dir, f"molecule_{idx+1}.sdf")
#     w = Chem.SDWriter(output_filename)
#     w.write(mol)
#     w.close()
#     # break
#     count += 1

# print('Total:',count)


# sdf_directory = output_dir


def check_molecule_correctness(sdf_directory):
    """
    sdf_directory: folder storing sdf files, each file contain a molecule
    to see if each molecule is 3D
    """
    valid_3d_count = 0
    invalid_count = 0
    for filename in tqdm(os.listdir(sdf_directory)):
        if filename.endswith(".sdf"):
            filepath = os.path.join(sdf_directory, filename)
            suppl = Chem.SDMolSupplier(filepath, removeHs=False)
            mol = next(suppl)  
            if mol is None:
                invalid_count += 1
                print(f"{filename} is invalid.")
                continue
            try:
                conf = mol.GetConformer()
                z_coords = [conf.GetAtomPosition(atom.GetIdx()).z for atom in mol.GetAtoms()]
                if any(z != 0.0 for z in z_coords):
                    valid_3d_count += 1
                else:
                    invalid_count += 1
                    print(f"{filename} is invalid: all Z coordinates are zero.")
            except ValueError as e:
                invalid_count += 1
                print(f"{filename} skipped: {str(e)}")

    print(f"Total valid 3D molecules: {valid_3d_count}")
    print(f"Total invalid or 2D molecules: {invalid_count}")


def display_3D_coordinates_and_charges(sdf_directory):
    for filename in os.listdir(sdf_directory):
        if filename.endswith(".sdf"):
            filepath = os.path.join(sdf_directory, filename)
            suppl = Chem.SDMolSupplier(filepath, removeHs=False)
            mol = next(suppl)
            if mol is None:
                print(f"{filename} is invalid.")
                continue
            print(f"\nMolecule from file: {filename}")
            try:
                conf = mol.GetConformer()
                print("3D Coordinates:")
                for atom in mol.GetAtoms():
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    print(f"Atom {atom.GetIdx()} ({atom.GetSymbol()}): x={pos.x:.4f}, y={pos.y:.4f}, z={pos.z:.4f}")
                charge_count = sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() != 0)
                print(f"Number of atoms with formal charges: {charge_count}")

            except ValueError as e:
                print(f"{filename} skipped: {str(e)}")

# display_3D_coordinates_and_charges(sdf_directory)


# data = '/data/lab_ph/kyle/projects/DrugDesign/datasets/qm9/dsgdb9nsd.xyz.tar.bz2'
# if tarfile.is_tarfile(data):
#     tardata = tarfile.open(data, 'r')
#     files = tardata.getmembers()

# readfile = lambda data_pt: tardata.extractfile(data_pt)

# molecules = []

# for file in files:
#     print(file)
#     with readfile(file) as openfile:
#         xyz_lines = [line.decode('UTF-8') for line in openfile.readlines()]
#         for i in xyz_lines:
#             print(i)
#         break
#         molecule = process_file_fn(openfile)
#         if molecule is not None:  
#             molecules.append(molecule)

# props = molecules[0].keys()
# assert all(props == mol.keys() for mol in molecules), 'All molecules must have same set of properties/keys!'

# molecules = {prop: [mol[prop] for mol in molecules] for prop in props}

# if stack:
#     molecules = {key: pad_sequence(val, batch_first=True) if val[0].dim() > 0 else torch.stack(val) for key, val in molecules.items()}

# return molecules






def process_xyz_gdb9(datafile):
    """
    Read xyz file and return a molecular dict with number of atoms, energy, forces, coordinates and atom-type for the gdb9 dataset.

    Parameters
    ----------
    datafile : python file object
        File object containing the molecular data in the MD17 dataset.
 
    Returns
    -------
    molecule : dict
        Dictionary containing the molecular properties of the associated file object.

    Notes
    -----
    TODO : Replace breakpoint with a more informative failure?
    """
    xyz_lines = [line.decode('UTF-8') for line in datafile.readlines()]
    
    # '--------------------------------'
    from qm9.custom_condition import get_properties
    from qm9.bond_analyze import get_bond_order

    smiles = xyz2smiles(xyz_lines)

    if smiles is None:
      molecule = None
    else:
      prop = get_properties(smiles)
      prop_list = [prop['MW'], prop['HBD'], prop['HBA'], prop['LogP'], prop['TPSA']]

    # '--------------------------------'

      num_atoms = int(xyz_lines[0])
      mol_props = xyz_lines[1].split()  # get original properties
      mol_props = mol_props + prop_list  # this line is added to merge properties
      mol_xyz = xyz_lines[2:num_atoms+2]
      mol_freq = xyz_lines[num_atoms+2]

      atom_charges, atom_positions = [], []
      for line in mol_xyz:
          atom, posx, posy, posz, _ = line.replace('*^', 'e').split()
          atom_charges.append(charge_dict[atom])
          atom_positions.append([float(posx), float(posy), float(posz)])

      # should be more general 
      prop_strings = ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', \
      'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv', 'MW', 'HBD', 'HBA', 'LogP', 'TPSA']
      prop_strings = prop_strings[1:]
      mol_props = [int(mol_props[1])] + [float(x) for x in mol_props[2:]]
      mol_props = dict(zip(prop_strings, mol_props))
      mol_props['omega1'] = max(float(omega) for omega in mol_freq.split())
      # print(mol_props)
      molecule = {'num_atoms': num_atoms, 'charges': atom_charges, 'positions': atom_positions}
      molecule.update(mol_props)
      molecule = {key: torch.tensor(val) for key, val in molecule.items()}

    '--------------------------------'
    # if invalid:
    #   print(molecule)
    '--------------------------------'

    return molecule

def convert_to_float(s):
    # Replace the '*^' with 'e', which is Python's notation for scientific notation
    s = s.replace('*^', 'e')
    try:
        return float(s)
    except ValueError:
        print(f"Could not convert {s} to float.")
        raise

def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)
















# suppl = Chem.SDMolSupplier(sdf_file)

# count = 0
# for idx, mol in tqdm(enumerate(suppl)):
    # if idx != 23299:
    #     continue

    # if mol is None:
    #     continue  
    
    # all_zero = True
    # conf = mol.GetConformer()
    # for atom in mol.GetAtoms():
    #     if conf.GetAtomPosition(atom.GetIdx()).z != 0.0:
    #         all_zero = False
    #         break
    # if all_zero:
    #     continue

    # try:
    #     if mol.GetNumConformers() > 0:
    #         conf = mol.GetConformer()
    #         z_coords = [conf.GetAtomPosition(atom.GetIdx()).z for atom in mol.GetAtoms()]
    #         if all(z == 0.0 for z in z_coords):
    #             print(f"Molecule {idx+1} skipped: all Z coordinates are zero.")
    #             continue
    # except ValueError as e:
    #     print(f"Molecule {idx+1} skipped: {str(e)}")
    #     continue

    # for atom in mol.GetAtoms():
    #     pos = conf.GetAtomPosition(atom.GetIdx())
    #     print(f"Atom {atom.GetIdx()} ({atom.GetSymbol()}): x={pos.x}, y={pos.y}, z={pos.z}")
    
    # print()

    # count += 1
    

# print(count)








# geom

import os
import numpy as np
import msgpack
from rdkit import Chem
from rdkit.Chem import AllChem

def extract_conformers(args):
    drugs_file = os.path.join(args.data_dir, args.data_file)
    save_file = f"geom_drugs_{'no_h_' if args.remove_h else ''}{args.conformations}"
    smiles_list_file = 'geom_drugs_smiles.txt'
    number_atoms_file = f"geom_drugs_n_{'no_h_' if args.remove_h else ''}{args.conformations}"

    unpacker = msgpack.Unpacker(open(drugs_file, "rb"))

    all_smiles = []
    all_number_atoms = []
    dataset_conformers = []
    mol_id = 0
    sdf_writer = Chem.SDWriter(os.path.join(args.data_dir, f"geom.sdf"))
    
    for i, drugs_1k in enumerate(tqdm(unpacker)):
        print(f"Unpacking file {i}...")
        for smiles, all_info in drugs_1k.items():
            all_smiles.append(smiles)
            conformers = all_info['conformers']

            # Create an RDKit molecule from the SMILES string
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Could not parse SMILES: {smiles}")
                continue

            mol = Chem.AddHs(mol)  # Add hydrogens

            # Get the energy of each conformer. Keep only the lowest values
            all_energies = []
            for conformer in conformers:
                all_energies.append(conformer['totalenergy'])
            all_energies = np.array(all_energies)
            argsort = np.argsort(all_energies)
            lowest_energies = argsort[:args.conformations]

            for id in lowest_energies:
                conformer = conformers[id]
                coords = np.array(conformer['xyz']).astype(float)  # n x 4
                if args.remove_h:
                    mask = coords[:, 0] != 1.0
                    coords = coords[mask]
                n = coords.shape[0]
                all_number_atoms.append(n)

                # Set conformer coordinates to the RDKit molecule
                mol_conf = Chem.Conformer(n)
                for atom_id in range(n):
                    mol_conf.SetAtomPosition(atom_id, coords[atom_id, 1:])
                mol.AddConformer(mol_conf, assignId=True)

                # Write the conformer to the SDF file
                sdf_writer.write(mol)

                mol_id += 1

    sdf_writer.close()
    
    print("Total number of conformers saved", mol_id)
    all_number_atoms = np.array(all_number_atoms)
    # dataset = np.vstack(dataset_conformers)

    # print("Total number of atoms in the dataset", dataset.shape[0])
    # print("Average number of atoms per molecule", dataset.shape[0] / mol_id)

    # Save conformations
    # np.save(os.path.join(args.data_dir, save_file), dataset)
    # Save SMILES
    # with open(os.path.join(args.data_dir, smiles_list_file), 'w') as f:
    #     for s in all_smiles:
    #         f.write(s)
    #         f.write('\n')

    # Save number of atoms per conformation
    # np.save(os.path.join(args.data_dir, number_atoms_file), all_number_atoms)
    print("Dataset processed.")


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conformations", type=int, default=1,
                        help="Max number of conformations kept for each molecule.")
    parser.add_argument("--remove_h", action='store_true', help="Remove hydrogens from the dataset.")
    parser.add_argument("--data_dir", type=str, default='/data/lab_ph/kyle/projects/DrugDesign/datasets/geom/')
    parser.add_argument("--data_file", type=str, default="drugs_crude.msgpack")
    args = parser.parse_args()
    extract_conformers(args)