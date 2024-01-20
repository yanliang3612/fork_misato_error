"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""

import os
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from rdkit import Chem
# from train_diffusion import dataset
import torch
# atomic van der waals radii in Angstrom unit
vdw_radii_dict = np.array([0, 1.1, 0, 0, 0, 0, 1.7, 1.55, 1.52, 1.47, 1.54, 2.27, 1.73, 1.84, 2.10, 1.80, 1.80, 1.75], dtype=np.float32)



# prepare input for MSMS surface computation
def convert_to_xyzrn(split,data_file_sdf_list,src_fpath, out_root):
    for i in range(len(data_file_sdf_list)):
        sdf_path = os.path.join(src_fpath,data_file_sdf_list[i])
        assert os.path.exists(sdf_path)
        suppl = Chem.SDMolSupplier(sdf_path)
        mol = suppl[0]

        # num_atoms_no_h = mol.GetNumAtoms()
        #
        # mol = Chem.AddHs(mol)

        assert mol is not None
        # num_atoms
        num_atoms = mol.GetNumAtoms()

        # positions
        positions = np.zeros((num_atoms, 3),dtype=np.float32)
        if mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            for j in range(num_atoms):
                pos = conf.GetAtomPosition(j)
                positions[j] = [pos.x, pos.y, pos.z]

        # charges
        charges = np.zeros(num_atoms, dtype=int)
        if mol is not None:
           for s, atom in enumerate(mol.GetAtoms()):
               charges[s] = atom.GetAtomicNum()
        atom_xyz = positions
        atom_Z = charges
        atom_r = vdw_radii_dict[atom_Z]
        atom_rn = [f'{atom_r[l]} 1 {atom_Z[l]} ' for l in range(num_atoms)]

        if split == "train":
            gdb_id = 'gdb_' + str(i+1).rjust(6, '0')
        elif split == "val":
            gdb_id = 'gdb_' + str(i+1+99990).rjust(6, '0')
        out_dir = os.path.join(out_root, gdb_id)
        os.makedirs(out_dir, exist_ok=False)
        data_file = (data_file_sdf_list[i][:-4]).replace('/', '_')
        xyzrn_path = os.path.join(out_dir, f'{data_file}.xyzrn')
        if os.path.exists(xyzrn_path):
            print(f"Already Have!!!")
            continue
        else:
            with open(xyzrn_path, 'w') as f:
                for a in range(num_atoms):
                    coords = '{:.6f} {:.6f} {:.6f} '.format(*atom_xyz[a])
                    f.write(coords + atom_rn[a] +'\n')

            print(f"finish the sdf {os.path.join(src_fpath, data_file_sdf_list[i])}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qm9-source', type=str, default='/mnt/data/oss_beijing/mesh/targetdiff_mesh/data/crossdocked_pocket10')
    parser.add_argument('--out-root', type=str, default='/mnt/data/oss_beijing/mesh/targetdiff_mesh/data/crossdocked_pocket10_mesh4')
    args = parser.parse_args()
    print(args)
    # specify IO dir
    assert os.path.exists(args.qm9_source)
    # if os.path.exists(args.out_root):
    #     shutil.rmtree(args.out_root)
    # os.makedirs(args.out_root, exist_ok=True)

    for split in ['train']:
        src_fpath = os.path.join("/mnt/workspace/dlly/targetdiff_mesh/data/", f'{split}_data_file_sdf_list.pt')
        assert os.path.isfile(src_fpath)
        data_file_sdf_list = torch.load(src_fpath,map_location="cpu")
        convert_to_xyzrn(split,data_file_sdf_list, args.qm9_source, args.out_root)
