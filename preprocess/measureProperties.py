import sys, os
from rdkit import RDConfig, Chem, RDLogger
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from rdkit.Chem.QED import qed as qed_
# from vina import Vina
from rdkit.Chem import AllChem
from torch import cuda
from typing import Union
import datamol as dm
from pdbfixer import PDBFixer
from openmm.app import PDBFile
import numpy as np
import MDAnalysis as mda
from visualize import draw_docking_figures
from tqdm import tqdm
import subprocess


logger = RDLogger.logger()
logger.setLevel(RDLogger.CRITICAL)


def measureQED(input_data) -> float:
    """ Input a smile or mol object """
    try:
        if isinstance(input_data, str):
            mol = Chem.MolFromSmiles(input_data)
        elif isinstance(input_data, Chem.rdchem.Mol):
            mol = input_data
        else:
            print('Invalid input type for QED measurement!')
            return None
        
        qed = round(qed_(mol), 4)
        return qed
    except:
        print('Fail to measure QED!')
        return None

def measureSAS(input_data) -> float:
    """ Input a smile or mol object """
    try:
        if isinstance(input_data, str):
            mol = Chem.MolFromSmiles(input_data)
        elif isinstance(input_data, Chem.rdchem.Mol):
            mol = input_data
        else:
            print('Invalid input type for SAS measurement!')
            return None
        
        sascore = sascorer.calculateScore(mol)
        sascore = round(sascore, 4)
        return sascore
    except:
        print('Fail to measure SAS!')
        return None

def measureBindingAffinity(protein_file, molecule_file, center=None, box_size=None):
    """ 
    Both protein_file and molecule_file should be pdbqt files.
    Also need to use absolute paths.
    center and box_size can be determined by other function (calculate_center_and_box_size) automatically
    """
    if cuda.is_available() and False:
        # print("GPU detected. Using GNINA for docking.")
        from gnina import Gnina as DockingTool
        docking_tool_name = 'gnina'
    else:
        # print("No GPU detected. Using standard AutoDock Vina for docking.")
        from vina import Vina as DockingTool
        docking_tool_name = 'vina'
    v = DockingTool(sf_name=docking_tool_name, seed=42)

    v.set_receptor(protein_file)
    v.set_ligand_from_file(molecule_file)
    
    v.compute_vina_maps(center=center, box_size=box_size)
    v.dock(exhaustiveness=8, n_poses=10)
    affinity = v.score()[0]

    return round(affinity, 4), v.poses()


def getMolFromFile(filePath) -> Chem.rdchem.Mol:
    """ process sdf, mol or pdb files """
    try:
        _, file_extension = os.path.splitext(filePath)
        if file_extension.lower() in ['.sdf', '.mol']:
            mol = Chem.MolFromMolFile(filePath)
        elif file_extension.lower() == '.pdb':
            mol = Chem.MolFromPDBFile(filePath)
        else:
            mol = None
            print(f"Unsupported file extension: {file_extension}")
        if mol is None:
            print("Failed to load molecule from file. The file may be corrupted, contain invalid atom types, or be in an invalid format.")
        return mol
    except FileNotFoundError:
        print("File not found. Please check the file path.")
    except ValueError as ve:
        print(f"ValueError encountered: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

class Preprocessor:
    def prepare_receptor(
            self,
            receptor_path: Union[str, os.PathLike],
            vina_receptor_path: Union[str, os.PathLike],
        ):
            """Convert a receptor PDB file to PDBQT for vina docking."""

            convert(
                infile=receptor_path,
                in_format="pdb",
                out_format="pdbqt",
                outfile=vina_receptor_path,
                overwrite=True,
                rigid=True,
                struct_num=0,
                calc_charges=True,
                add_h=True,
            )

    def prepare_ligand(
        self,
        ligand_path: Union[str, os.PathLike],
        vina_ligand_path: Union[str, os.PathLike],
        in_format: str = "sdf",
    ):
        convert(
                infile=ligand_path,
                in_format=in_format,
                out_format="pdbqt",
                outfile=vina_ligand_path,
                overwrite=True,
                struct_num=0,
                calc_charges=True,
                charges_model="gasteiger",
                add_h=True,
                remove_h=False,
            )
        
def convert(
    infile: Union[str, os.PathLike],
    out_format: str,
    outfile: Union[str, os.PathLike] = None,
    in_format: str = None,
    make_3d: bool = False,
    overwrite: bool = False,
    rigid: bool = False,
    struct_num: int = None,
    calc_charges: bool = True,
    charges_model: str = "gasteiger",
    add_h: bool = True,
    remove_h: bool = False,
) -> Union[str, os.PathLike]:
    """Convert an input file to the output_format.
    This is really just a wrapper around pybel
    Args:
        infile: Input file
        out_format: output format
        outfile: Output file. If not provided, we will replace the
        in_format: Input format. If not provided, will be guessed from the extension
        make_3d: Whether to call mol.make3D() on the molecule when a 3D structure is missing
        overwrite: Whether to overwrite file if exists already
        rigid: whether to allow torsion angle in pdbqt writing. For receptors, rigid should be true.
        struct_num: structure number to use when the output is supposed to be pdbqt.
    Returns:
        outfile: path to the converted file
    """
    try:
        from openbabel import pybel
    except ImportError:
        raise ImportError("File convertion requires openbabel >= 3.0.0")
    infile = str(infile)
    outfile = str(outfile)
    if in_format is None:
        in_format = os.path.splitext(infile)[-1].lower().strip(".")
    if out_format not in pybel.outformats:
        raise ValueError(f"Output format {out_format} is not recognized by pybel !")
    if outfile is None:
        outfile = infile.replace(in_format, out_format)
    opt = {"align": None}
    if not rigid:
        opt = {"b": None, "p": None, "h": None}
    else:
        opt = {"r": None, "c": None, "h": None}
    if struct_num is None:
        raise ValueError(
                f"For pdbqt, you need to provide the number `struct_num` of the compound to save"
            )
    mols = _read_ob(infile, in_format)
    if struct_num is not None:
        mols = [mols[struct_num]]
    tmp_file = outfile
    out = pybel.Outputfile(format=out_format, filename=tmp_file, overwrite=overwrite, opt=opt)
    for m in mols:
        if add_h:
            m.addh()
        if remove_h:
            m.removeh()
        if not m.OBMol.HasNonZeroCoords() and make_3d is True:
            m.make3D()
        if calc_charges is True:
            m.calccharges(model=charges_model)
        out.write(m)
    out.close()
    if tmp_file != outfile:
        dm.utils.fs.copy_file(tmp_file, outfile)
        os.unlink(tmp_file)
    return outfile

def _read_ob(infile, in_format):
    """Read a molecule file with open babel
    Args:
        infile (Union[str, os.PathLike]): input file
        in_format (str, optional): input format
    Returns:
        mols (list): list of molecules found in the input file
    """
    try:
        from openbabel import pybel
    except ImportError:
        raise ImportError("Pybel is required for reading openbabel molecules")
    mols = [m for m in pybel.readfile(format=in_format, filename=infile)]
    return mols

def convert_molecule_to_pdbqt(molecule_file, output_file):
    # Convert the SDF file directly to PDBQT using Open Babel
    # not used
    os.system(f"obabel {molecule_file} -O {output_file} --gen3D")

def convert_protein_to_pdbqt(protein_file, output_file):
    # Ensure that the MGLTools are installed and `prepare_receptor4.py` is accessible
    # didn't use in this script
    # os.system(f"prepare_receptor4.py -r {protein_file} -o {output_file} -A checkhydrogens")
    subprocess.run(['prepare_receptor4.py', '-r', protein_file, '-o', output_file], check=True)

def prepare_protein_for_docking(input_pdb, output_pdbqt):
    # Step 1: Remove ligands and water molecules
    fixer = PDBFixer(input_pdb)
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)  # pH 7.4
    temp_pdb = "temp_standard.pdb"
    PDBFile.writeFile(fixer.topology, fixer.positions, open(temp_pdb, 'w'))

    # Step 2: Add Gasteiger charges and convert to PDBQT
    command = f"obabel -ipdb {temp_pdb} -opdbqt -O {output_pdbqt} --partialcharge gasteiger"
    os.system(command)

    # Clean up the temporary file
    os.remove(temp_pdb)

def get_ligand_center_and_box_size(sdf_file, padding=5.0):
    supplier = Chem.SDMolSupplier(sdf_file)
    molecule = supplier[0]  
    conformer = molecule.GetConformer()
    coords = np.array([conformer.GetAtomPosition(i) for i in range(molecule.GetNumAtoms())])
    center = coords.mean(axis=0)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    box_size = max_coords - min_coords + padding
    
    return center, box_size

def calculate_center_and_box_size(pdb_file, binding_site_residues=None):
    u = mda.Universe(pdb_file)
    if binding_site_residues:
        binding_site = u.select_atoms("resid " + " ".join(map(str, binding_site_residues)))
    else:
        binding_site = u.select_atoms("protein")
    center = binding_site.center_of_mass()
    min_coord = binding_site.positions.min(axis=0)
    max_coord = binding_site.positions.max(axis=0)
    box_size = max_coord - min_coord + 10
    return center.tolist(), box_size.tolist()

def sdf2pdbqt(molecule_file, output_file_molecule):
    """
    molecule_file: sdf file
    output_file_molecule: pdbqt file
    protein_file: pdb file
    output_file_protein: prepared pdbqt file path
    """
    Preprocessor().prepare_ligand(molecule_file, output_file_molecule)
    # center, box_size = calculate_center_and_box_size(protein_file)
    return output_file_molecule

def getCenter(protein_file):
    """
    protein_file: pdb file
    """
    center, box_size = calculate_center_and_box_size(protein_file)
    return center, box_size


if __name__ == "__main__":
# def run():
    # print('Pin1 inhibitor')
    # smiles = "CC1=CC2=C(C=C1C)N=C(N2)CSC(=S)N3CCCCC3"
    # print('------------- From SMILE --------------')
    # qed = measureQED(smiles)
    # print("QED:",qed)
    # sas = measureSAS(smiles)
    # print("SAS",sas)

    # print('------------- From SDF --------------')
    # filePath = '/data/lab_ph/kyle/projects/DrugDesign/mockData/Pin1Inhibitor.sdf'
    # mol = getMolFromFile(filePath)
    # qed = measureQED(mol)
    # print("QED:",qed)
    # sas = measureSAS(mol)
    # print("SAS",sas)


    # print('Pin1 modulator')
    # smiles = "C1=CC=C2C(=C1)C=CC=C2/C=C/3\C(=O)N(C(=S)S3)CCCC(=O)O"
    # print('------------- From SMILE --------------')
    # qed = measureQED(smiles)
    # print("QED:",qed)
    # sas = measureSAS(smiles)
    # print("SAS",sas)

    # print('------------- From SDF --------------')
    # filePath = '/data/lab_ph/kyle/projects/DrugDesign/mockData/Pin1Modulator.sdf'
    # mol = getMolFromFile(filePath)
    # qed = measureQED(mol)
    # print("QED:",qed)
    # sas = measureSAS(mol)
    # print("SAS",sas)

    # ------------------------------------------------------
    rootPath = '/data/lab_ph/kyle/projects/DrugDesign/mockData/'
    # candidates = ['Pin1Inhibitor', 'Pin1Inhibitor2',
    #               'Pin1Modulator', '6vhnInhibitor','C2H6O','CH4',
    #               '6vhnInhibitor1','6vhnInhibitor2','6vhnInhibitor3',
    #               '6vhnInhibitor4','6vhnInhibitor5','6vhnInhibitor6',
    #               '6vhnInhibitor7','6vhnInhibitor8','6vhnInhibitor9',
    #               ]
    candidates = ['6vhnInhibitor']
    
    results = {}
    for sdf in candidates:
        molecule_file = rootPath + f"{sdf}.sdf"
        output_file_molecule = rootPath + f"{sdf}.pdbqt"
        Preprocessor().prepare_ligand(molecule_file, output_file_molecule)

        mol = getMolFromFile(molecule_file)
        qed = measureQED(mol)
        print("QED:",qed)
        sas = measureSAS(mol)
        print("SAS",sas)

        protein_name = '6vhn_prepared'
        output_file_protein = rootPath + f"{protein_name}.pdbqt"
        pdb_file = rootPath + f'{protein_name}.pdb'
        center, box_size = calculate_center_and_box_size(pdb_file)
        # print(center, box_size)

        affinity, best_pose = measureBindingAffinity(output_file_protein, output_file_molecule, center, box_size)
        results.update({sdf:affinity})

        # print(type(best_pose))
        best_pose_path = f'/data/lab_ph/kyle/projects/DrugDesign/results/dockingImages/best_pose_{sdf}.pdbqt'
        with open(best_pose_path, "w") as f:
            f.write(best_pose)

        # visualize
        docking_image_path = f'/data/lab_ph/kyle/projects/DrugDesign/results/dockingImages/{sdf}.png'
        # return draw_docking_figures(best_pose, pdb_file, docking_image_path)
        # return best_pose_path, pdb_file, docking_image_path
        print(best_pose_path)
        print(pdb_file)
        print(docking_image_path)
        
    print("Summarize results:")
    for key, value in results.items():
        print(f"{key}: {value}", 'kcal/mol')
    


    # rootPath = '/data/lab_ph/kyle/projects/DrugDesign/mockData/'
    # results = {}
    # sdf_directory = "/data/lab_ph/kyle/projects/DrugDesign/datasets/zinc15/molecules"

    # rpath = '/data/lab_ph/kyle/projects/DrugDesign/results/zincprocessed.txt'
    # with open(rpath, 'w') as f:
    #     for filename in tqdm(os.listdir(sdf_directory)):
    #         if filename.endswith(".sdf"):
    #             # sdf = filename.strip(".sdf")
    #             sdf = 'temp'
    #             molecule_file = os.path.join(sdf_directory, filename)

    #             mol = getMolFromFile(molecule_file)
    #             qed = measureQED(mol)
    #             print("\nQED:",qed)
    #             sas = measureSAS(mol)
    #             print("SAS",sas)

    #             output_file_molecule = rootPath + f"{sdf}.pdbqt"
    #             Preprocessor().prepare_ligand(molecule_file, output_file_molecule)

    #             protein_name = '6vhn_prepared'
    #             output_file_protein = rootPath + f"{protein_name}.pdbqt"
    #             pdb_file = rootPath + f'{protein_name}.pdb'
    #             center, box_size = calculate_center_and_box_size(pdb_file)
    #             # print(center, box_size)

    #             affinity, best_pose = measureBindingAffinity(output_file_protein, output_file_molecule, center, box_size)
    #             # results.update({sdf:affinity})
    #             print("affinity:",affinity)

    #             f.write(f"{filename},{qed:.4f},{sas:.4f},{affinity:.4f}\n")


    
    # print("Summarize results:")
    # for key, value in results.items():
    #     print(f"{key}: {value}", 'kcal/mol')




      