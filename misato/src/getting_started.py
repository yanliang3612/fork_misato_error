# MISATO-Dataset: a tutorial

# In this notebook, we will show how our QM and MD dataset are stored in h5 files. We also show how the data can be loaded so that it can be used by a deep learning model.

# We start by importing the useful packages and set up the paths of the file

import sys
import os
sys.path.insert(0,os.path.join(os.path.abspath('').split('misato-dataset')[0],'misato-dataset/src/data/components/'))

import h5py
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from data.components.datasets import MolDataset, ProtDataset
from data.components.transformQM import GNNTransformQM
from data.components.transformMD import GNNTransformMD
from data.qm_datamodule import QMDataModule
from data.md_datamodule import MDDataModule
from data.processing import preprocessing_db

qmh5_file = "../data/QM/h5_files/tiny_qm.hdf5"
norm_file = "../data/QM/h5_files/qm_norm.hdf5"
norm_txtfile = "../data/QM/splits/train_norm.txt"


# H5 files presentations
# We read the QM H5 file and H5 file used to normalize the target values.
qm_H5File = h5py.File(qmh5_file)
qm_normFile = h5py.File(norm_file)

# The ligands can be accessed using the pdb-id. Bellow we show the first ten molecules of the file.
qm_H5File.keys()

# The following properties are available for each atom:
qm_H5File["10GS"]["atom_properties"]["atom_properties_names"][()]

# You can access the values for each of the properties using the respective index. For example the coordinates are given in the first 3 entries:
xyz = qm_H5File["10GS"]["atom_properties"]["atom_properties_values"][:, 0:3]

# We also provide several molecular properties that can be accessed directly using the respective key.
qm_H5File["10GS"]["mol_properties"].keys()

# Target values can be accessed by specifiying into bracket the molecule name, then mol_properties and finally the name of the target value that we want to access:
qm_H5File["10GS"]["mol_properties"]["Electron_Affinity"][()]

# We can access to the mean and standard-deviation of each target value over all structures by specifiying it into bracket. We first specify the set, then the target value and finally either mean or std.
qm_normFile.keys()

print(qm_normFile["Electron_Affinity"]["mean"][()])
print(qm_normFile["Electron_Affinity"]["std"][()])

# Datasets and dataloaders
# PyTorch
# The QM and MD datasets are warped into a PyTorch Dataset class under the name MolDataset and ProtDataset, respectively. The parameters taken by the two classes as well as their types can be found as follow.

help(MolDataset)
help(ProtDataset)

train = "../data/QM/splits/train_tinyQM.txt"

transform = T.RandomTranslate(0.25)
batch_size = 128
num_workers = 48

data_train = MolDataset(qmh5_file, train, target_norm_file=norm_file, transform=GNNTransformQM(), post_transform=transform)

# Finally, we can load our data using the PyTorch DataLoader.
train_loader = DataLoader(data_train, batch_size, shuffle=True, num_workers=0)

for idx, val in enumerate(train_loader):
    print(val)
    break



### PyTorch lightning
# The QMDataModule is a class inherated from LightningDataModule that instanciate the MolDataset for training, validation and test set and returns a dataloader for each set.
# We start by instanciation of the QMDataModule

files_root =  "../data/QM"
qmh5file = "h5_files/tiny_qm.hdf5"

tr = "splits/train_tinyQM.txt"
v = "splits/val_tinyQM.txt"
te = "splits/test_tinyQM.txt"

qmdata = QMDataModule(files_root, h5file=qmh5file, train=tr, val=v, test=te, num_workers=0)

### Then, we call the setup function to instanciate the MolDataset for training, validation and test set
qmdata.setup()

### Finally, we can return a dataloader for each set.
train_loader = qmdata.train_dataloader()
for idx, val in enumerate(train_loader):
    print(val)
    break


### MD dataset
### # We generated a tiny h5 file that can be inspected right away. We do this for the structure with pdb-id 10GS.

mdh5_file_tiny = '../data/MD/h5_files/tiny_md.hdf5'
md_H5File_tiny = h5py.File(mdh5_file_tiny)

md_H5File_tiny['10GS'].keys()

[(key, np.shape(md_H5File_tiny['10GS'][key])) for key in md_H5File_tiny['10GS'].keys()]

# To run models for the MD dataset you will most likely need to preprocess the h5 file based on your model.
# We provide a preprocessing script (see data/processing/preprocessing_db.py) that can filter out the atom
# types that you are not interested in (e.g. H-atoms) or calculate values of interest based on your models.
# Here, we will show how to use the script to calculate the adaptability values on the dataset and stripping the H-atoms.
# In this notebook we define a new Args class, if you use the script in the terminal just provide these values as input
# parameters in the command line.


class Args:
    # input file
    datasetIn = "../data/MD/h5_files/tiny_md.hdf5"
    # Feature that should be stripped, e.g. atoms_element or atoms_type
    strip_feature = "atoms_element"
    # Value to strip, e.g. if strip_freature= atoms_element; 1 for H.
    strip_value = 1
    # Start index of structures
    begin = 0
    # End index of structures
    end = 20
    # We calculate the adaptability for each atom.
    # Default behaviour will also strip H atoms, if no stripping should be perfomed set strip_value to -1.
    Adaptability = True
    # If set to True this will create a new feature that combines one entry for each protein AA but all ligand entries;
    # e.g. for only ca set strip_feature = atoms_type and strip_value = 14
    Pres_Lat = False
    # We strip the complex by given distance (in Angstrom) from COG of molecule,
    # use e.g. 15.0. If default value is given (0.0) no pocket stripping will be applied.
    Pocket = 0.0
    # output file name and location
    datasetOut = "../data/MD/h5_files/tiny_md_out.hdf5"


args = Args()

preprocessing_db.main(args)

# The same steps used for QM can be used to load the MD dataset. We start by loading the generated h5 file.
files_root =  ""

mdh5_file = '../data/MD/h5_files/tiny_md_out.hdf5'

train_idx = "../data/MD/splits/train_tinyMD.txt"
val_idx = "../data/MD/splits/val_tinyMD.txt"
test_idx = "../data/MD/splits/test_tinyMD.txt"

md_H5File = h5py.File(mdh5_file)

# During preprocessing the H-atoms were stripped (see the change in atoms_ shape) and a new feature, the adaptability was calculated for each atom.
[(key, np.shape(md_H5File['10GS'][key])) for key in md_H5File['10GS'].keys()]

# Atom's coordinates from the first frame
xyz = md_H5File['10GS']['trajectory_coordinates'][0, :, :]

# We can now initiate the dataloader.
train_dataset = ProtDataset(mdh5_file, idx_file=train_idx, transform=GNNTransformMD(), post_transform=T.RandomTranslate(0.05))

train_loader = DataLoader(train_dataset, batch_size=16, num_workers=16)

for idx, val in enumerate(train_loader):
    print(val)
    break

mddata = MDDataModule(files_root, h5file=mdh5_file, train=train_idx, val=val_idx, test=test_idx, num_workers=0)

mddata.setup()

train_loader = mddata.train_dataloader()

for idx, val in enumerate(train_loader):
    print(val)
    break







