import sys, os
# from rdkit import RDConfig, Chem, RDLogger
# sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# import sascorer
# from rdkit.Chem.QED import qed as qed_
# # from vina import Vina
# from rdkit.Chem import AllChem
# from torch import cuda
from typing import Union
import datamol as dm
# from pdbfixer import PDBFixer
# from openmm.app import PDBFile
# import numpy as np
# import MDAnalysis as mda
# from visualize import draw_docking_figures
# from tqdm import tqdm
# import subprocess


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
