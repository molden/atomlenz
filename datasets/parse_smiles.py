import os
import pathlib
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from scipy import ndimage, misc
import skimage.draw as draw
from sklearn.model_selection import train_test_split
from rdkit import Chem


class SMILES_CountAtoms_Dataset(Dataset):

    def __init__(self, datasetfolder):
        self.datasetfolder = datasetfolder

    def __len__(self):
        return len([name for name in os.listdir(f"{self.datasetfolder}/images/")])

    def __getitem__(self, idx):
        #self.dict_atom = {'C': 1, 'H': 2, 'N': 3, 'O': 4, 'S': 5, 'F': 6, 'Cl': 7, 'Br': 8, 'I': 9, 'Se': 10, 'P': 11, 'B': 12, 'Si': 13}
        atomnumber_dict = {6: 1, 1: 2, 7: 3, 8: 4, 16: 5, 9: 6, 17: 7, 35: 8, 53: 9, 34: 10, 15: 11, 5: 12, 14: 13, 52: 15, 50: 16, 33: 17, 13: 18, 32: 19}
        #atomnumber_dict = {6: 1, 1: 2, 7: 3, 8: 4, 16: 5, 9: 6, 17: 7, 35: 8, 53: 9, 34: 10, 15: 11, 5: 12, 14: 13, 19: 14, 33: 14, 36: 14, 51: 14, 74: 14, 32: 14, 50: 14, 80: 14, 52: 14, 92: 14, 49: 14, 75: 14, 81: 14, 40: 14, 13: 14, 23: 14, 82: 14, 42: 14, 20: 14, 46: 14, 24: 14, 26: 14, 83: 14, 78: 14, 41: 14, 45: 14}
        #32  13  33  50  52
        count_atoms = np.zeros(25)
        base_path = pathlib.Path(self.datasetfolder)
        smiles_dir = base_path.joinpath("smiles")
        image_dir  = base_path.joinpath("images")
        smilespath = smiles_dir.joinpath(f"{idx}.txt")
        with open(smilespath) as f:
            true_smiles = f.readline().strip('\n')
        imageindex=idx
        imagename=image_dir.joinpath(f"{idx}.png")
        #import ipdb; ipdb.set_trace()
        img      = Image.open(f"{imagename}").convert("L")
        #img      = Ftrans.to_tensor(img)
        m        = Chem.MolFromSmiles(true_smiles, sanitize=False)
        for atom in m.GetAtoms():
            if atom.GetAtomicNum()==1:
                if atom.GetIsotope()==2:
                   count_atoms[20] += 1
                elif atom.GetIsotope()==3:
                   count_atoms[21] += 1
                else:
                   count_atoms[2] += 1
            else:
                count_atoms[atomnumber_dict.get(atom.GetAtomicNum(),14)] +=1
            #print(atom.GetAtomicNum())
        #values, counts = np.unique(count_atoms, return_counts=True)
        return img, count_atoms, true_smiles

class SMILES_CountCharges_Dataset(Dataset):

    def __init__(self, datasetfolder):
        self.datasetfolder = datasetfolder

    def __len__(self):
        return len([name for name in os.listdir(f"{self.datasetfolder}/images/")])

    def __getitem__(self, idx):
        #self.dict_atom = {'C': 1, 'H': 2, 'N': 3, 'O': 4, 'S': 5, 'F': 6, 'Cl': 7, 'Br': 8, 'I': 9, 'Se': 10, 'P': 11, 'B': 12, 'Si': 13}
        #atomnumber_dict = {6: 1, 1: 2, 7: 3, 8: 4, 16: 5, 9: 6, 17: 7, 35: 8, 53: 9, 34: 10, 15: 11, 5: 12, 14: 13, 52: 15, 50: 16, 33: 17, 13: 18, 32: 19}
        #atomnumber_dict = {6: 1, 1: 2, 7: 3, 8: 4, 16: 5, 9: 6, 17: 7, 35: 8, 53: 9, 34: 10, 15: 11, 5: 12, 14: 13, 19: 14, 33: 14, 36: 14, 51: 14, 74: 14, 32: 14, 50: 14, 80: 14, 52: 14, 92: 14, 49: 14, 75: 14, 81: 14, 40: 14, 13: 14, 23: 14, 82: 14, 42: 14, 20: 14, 46: 14, 24: 14, 26: 14, 83: 14, 78: 14, 41: 14, 45: 14}
        #32  13  33  50  52
        dict_charges = {0: 1, 1: 2, -1: 3, 2: 4, -2: 5, 3: 6, 4: 7, 5: 8, 6: 9}
        count_charges = np.zeros(10)
        base_path = pathlib.Path(self.datasetfolder)
        smiles_dir = base_path.joinpath("smiles")
        image_dir  = base_path.joinpath("images")
        smilespath = smiles_dir.joinpath(f"{idx}.txt")
        with open(smilespath) as f:
            true_smiles = f.readline().strip('\n')
        imageindex=idx
        imagename=image_dir.joinpath(f"{idx}.png")
        #import ipdb; ipdb.set_trace()
        img      = Image.open(f"{imagename}").convert("L")
        #img      = Ftrans.to_tensor(img)
        m        = Chem.MolFromSmiles(true_smiles, sanitize=False)
        for atom in m.GetAtoms():
            count_charges[dict_charges.get(atom.GetFormalCharge(),1)] +=1
            #print(atom.GetAtomicNum())
        #values, counts = np.unique(count_atoms, return_counts=True)
        return img, count_charges, true_smiles

class SMILES_CountStereos_Dataset(Dataset):

    def __init__(self, datasetfolder):
        self.datasetfolder = datasetfolder

    def __len__(self):
        return len([name for name in os.listdir(f"{self.datasetfolder}/images/")])

    def __getitem__(self, idx):
        #self.dict_atom = {'C': 1, 'H': 2, 'N': 3, 'O': 4, 'S': 5, 'F': 6, 'Cl': 7, 'Br': 8, 'I': 9, 'Se': 10, 'P': 11, 'B': 12, 'Si': 13}
        #atomnumber_dict = {6: 1, 1: 2, 7: 3, 8: 4, 16: 5, 9: 6, 17: 7, 35: 8, 53: 9, 34: 10, 15: 11, 5: 12, 14: 13, 52: 15, 50: 16, 33: 17, 13: 18, 32: 19}
        #atomnumber_dict = {6: 1, 1: 2, 7: 3, 8: 4, 16: 5, 9: 6, 17: 7, 35: 8, 53: 9, 34: 10, 15: 11, 5: 12, 14: 13, 19: 14, 33: 14, 36: 14, 51: 14, 74: 14, 32: 14, 50: 14, 80: 14, 52: 14, 92: 14, 49: 14, 75: 14, 81: 14, 40: 14, 13: 14, 23: 14, 82: 14, 42: 14, 20: 14, 46: 14, 24: 14, 26: 14, 83: 14, 78: 14, 41: 14, 45: 14}
        #32  13  33  50  52
        count_stereos=np.zeros(2)
        base_path = pathlib.Path(self.datasetfolder)
        smiles_dir = base_path.joinpath("smiles")
        image_dir  = base_path.joinpath("images")
        smilespath = smiles_dir.joinpath(f"{idx}.txt")
        with open(smilespath) as f:
            true_smiles = f.readline().strip('\n')
        imageindex=idx
        imagename=image_dir.joinpath(f"{idx}.png")
        #import ipdb; ipdb.set_trace()
        img      = Image.open(f"{imagename}").convert("L")
        #img      = Ftrans.to_tensor(img)
        m        = Chem.MolFromSmiles(true_smiles, sanitize=False)
        for atom in m.GetAtoms():
            if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
               count_stereos[1] +=1
            #print(atom.GetAtomicNum())
        #values, counts = np.unique(count_atoms, return_counts=True)
        return img, count_stereos, true_smiles

class SMILES_CountBonds_Dataset(Dataset):

    def __init__(self, datasetfolder):
        self.datasetfolder = datasetfolder

    def __len__(self):
        return len([name for name in os.listdir(f"{self.datasetfolder}/images/")])

    def __getitem__(self, idx):
        #self.dict_atom = {'C': 1, 'H': 2, 'N': 3, 'O': 4, 'S': 5, 'F': 6, 'Cl': 7, 'Br': 8, 'I': 9, 'Se': 10, 'P': 11, 'B': 12, 'Si': 13}
        dict_bond = {'SINGLE': 2, 'DOUBLE': 3, 'TRIPLE': 4}
        #atomnumber_dict = {6: 1, 1: 2, 7: 3, 8: 4, 16: 5, 9: 6, 17: 7, 35: 8, 53: 9, 34: 10, 15: 11, 5: 12, 14: 13, 19: 14, 33: 14, 36: 14, 51: 14, 74: 14, 32: 14, 50: 14, 80: 14, 52: 14, 92: 14, 49: 14, 75: 14, 81: 14, 40: 14, 13: 14, 23: 14, 82: 14, 42: 14, 20: 14, 46: 14, 24: 14, 26: 14, 83: 14, 78: 14, 41: 14, 45: 14}
        count_bonds = np.zeros(5)
        base_path = pathlib.Path(self.datasetfolder)
        smiles_dir = base_path.joinpath("smiles")
        image_dir  = base_path.joinpath("images")
        smilespath = smiles_dir.joinpath(f"{idx}.txt")
        with open(smilespath) as f:
            true_smiles = f.readline().strip('\n')
        imageindex=idx
        imagename=image_dir.joinpath(f"{idx}.png")
        #import ipdb; ipdb.set_trace()
        img      = Image.open(f"{imagename}").convert("L")
        #img      = Ftrans.to_tensor(img)
        m        = Chem.MolFromSmiles(true_smiles, sanitize=False)
        try:
           Chem.Kekulize(m)
        except:
           print(true_smiles)
           #import ipdb; ipdb.set_trace()
        for bond in m.GetBonds():
            count_bonds[dict_bond.get(str(bond.GetBondType()))] +=1
       #     import ipdb; ipdb.set_trace()
            #print(atom.GetAtomicNum())
        #values, counts = np.unique(count_atoms, return_counts=True)
        return img, count_bonds, true_smiles

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Parse the SMILES labels")

    parser.add_argument("--datasetfolder", help="Dataset Folder", type=str, required=True)
    parser.add_argument("--outputfolder", help="Outputfolder where labels will be saved.", type=str, default="outputfolder/")
    parser.add_argument("--parse_bonds", help="Set to 1 to count bonds", type=int, default=0)
    parser.add_argument("--parse_charges", help="Set to 1 to count charges", type=int, default=0)
    parser.add_argument("--parse_stereos", help="Set to 1 to count stereo centers", type=int, default=0)
    args = parser.parse_args()
    if args.parse_bonds == 1:
        dataset = SMILES_CountBonds_Dataset(args.datasetfolder)
    elif args.parse_charges == 1:
        dataset = SMILES_CountCharges_Dataset(args.datasetfolder)
    elif args.parse_stereos == 1:
        dataset = SMILES_CountStereos_Dataset(args.datasetfolder)
    else:
        dataset = SMILES_CountAtoms_Dataset(args.datasetfolder)
    len_data = len(dataset)
    new_i = 0
    pathlib.Path(f"{args.outputfolder}/labels/").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{args.outputfolder}/images/").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{args.outputfolder}/smiles/").mkdir(parents=True, exist_ok=True)
    for i in range(len_data):
        img, counts, smiles = dataset[i]
        with open(f"{args.outputfolder}/labels/{new_i}.txt", "w") as fp:
            fp.write("label,xmin,ymin,xmax,ymax\n")
            for i_atom, label_count in enumerate(counts):
                for i in range(int(label_count)):
                    to_write=f"{i_atom-1},0,0,0,0\n"
                    fp.write(to_write)
        img.save(f"{args.outputfolder}/images/{new_i}.png")
        with open(f"{args.outputfolder}/smiles/{new_i}.txt", 'w') as f:
           f.write(f"{smiles}\n")
        new_i+=1

if __name__ == "__main__":
    main()

