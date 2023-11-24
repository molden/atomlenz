import os
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import shutil
from AtomLenz import *
#from utils_graph import *
#from Object_Smiles import Objects_Smiles 
from skimage import io, transform, color, data
from skimage.transform import resize

#from robust_detection import wandb_config
from robust_detection import utils
from robust_detection.models.rcnn import RCNN
from robust_detection.data_utils.rcnn_data_utils import Objects_RCNN, COCO_RCNN

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from rdkit import Chem


def main(model_cls, data_cls, args, logger = None):
    dataset = data_cls(**vars(args))
    atom_preds, bond_preds, charges_preds, stereo_preds = make_predictions(model_cls,args,args.experiment_path_atoms,args.experiment_path_bonds,args.experiment_path_charges,args.experiment_path_stereo,dataset)
    mol_graphs = create_molecules(atom_preds, bond_preds, charges_preds, stereo_preds)
    predictions=0
    correct=0
    correct_new=0
    correct_new_smiles=[]
    to_label_molecules=[]
    for image_idx,molecule in enumerate(mol_graphs[:]):
        base_path = pathlib.Path(args.data_path)
        image_dir = base_path.joinpath("images")
        smiles_dir = base_path.joinpath("smiles")
        impath = image_dir.joinpath(f"{image_idx}.png")
        smilespath = smiles_dir.joinpath(f"{image_idx}.txt")
        with open(smilespath) as f:
             true_smiles = f.readline().strip('\n')
        m = Chem.MolFromSmiles(true_smiles)
        molecule = save_mol_to_file(molecule,'molfile')
        #import ipdb; ipdb.set_trace()
        mol =  Chem.MolFromMolFile('molfile')
        try:
          pred_smiles = Chem.MolToSmiles(mol)
        except:
          pred_smiles = ""
        try:
          true_smiles = Chem.MolToSmiles(m)
        except:
          true_smiles = true_smiles
        if pred_smiles==true_smiles:
           correct+=1
           to_label_molecules.append([image_idx,molecule,true_smiles])
        else:
           new_molecule = alter_atoms(molecule,true_smiles)
           if new_molecule is not None:
              new_molecule = save_mol_to_file(new_molecule,'molfile_new')
              #import ipdb; ipdb.set_trace()
              mol_new =  Chem.MolFromMolFile('molfile_new') 
           try:
              pred_smiles_new = Chem.MolToSmiles(mol_new)
           except:
              pred_smiles_new = ""

           if pred_smiles_new==true_smiles:
              correct_new+=1
              correct_new_smiles.append(pred_smiles_new)
              to_label_molecules.append([image_idx,new_molecule,true_smiles])
           else:
              new_molecule = alter_bonds(molecule,true_smiles)
              if new_molecule is not None:
                 new_molecule = save_mol_to_file(new_molecule,'molfile_new')
              #import ipdb; ipdb.set_trace()
                 mol_new =  Chem.MolFromMolFile('molfile_new')
              try:
                 pred_smiles_new = Chem.MolToSmiles(mol_new)
              except:
                 pred_smiles_new = ""

              if pred_smiles_new==true_smiles:
                 correct_new+=1
                 correct_new_smiles.append(pred_smiles_new)
                 to_label_molecules.append([image_idx,new_molecule,true_smiles])
              else:
                 new_molecule = alter_charges(molecule,true_smiles)
                 if new_molecule is not None:
                    new_molecule = save_mol_to_file(new_molecule,'molfile_new')
              #import ipdb; ipdb.set_trace()
                    mol_new =  Chem.MolFromMolFile('molfile_new')
                 try:
                    pred_smiles_new = Chem.MolToSmiles(mol_new)
                 except:
                    pred_smiles_new = ""

                 if pred_smiles_new==true_smiles:
                    correct_new+=1
                    correct_new_smiles.append(pred_smiles_new)
                    to_label_molecules.append([image_idx,new_molecule,true_smiles])
                 else:
                    new_molecule = alter_stereos(molecule,true_smiles)
                    if new_molecule is not None:
                       new_molecule = save_mol_to_file(new_molecule,'molfile_new')
              #import ipdb; ipdb.set_trace()
                       mol_new =  Chem.MolFromMolFile('molfile_new')
                    try:
                       pred_smiles_new = Chem.MolToSmiles(mol_new)
                    except:
                       pred_smiles_new = ""

                    if pred_smiles_new==true_smiles:
                       correct_new+=1
                       correct_new_smiles.append(pred_smiles_new)
                       to_label_molecules.append([image_idx,new_molecule,true_smiles])
        predictions+=1
        #import ipdb; ipdb.set_trace()
    print(f"number correct: {correct}")
    print(f"number correct_new: {correct_new}")
    print(correct_new_smiles)
    print(f"total: {predictions}")
    print(f"relabeling ....")
    atom_image_path = pathlib.Path("atom_relabels/images/")
    atom_image_path.mkdir(parents=True, exist_ok=True)
    atom_scaled_image_path = pathlib.Path("atom_relabels/scaled_images/")
    atom_scaled_image_path.mkdir(parents=True, exist_ok=True)
    atom_label_path = pathlib.Path("atom_relabels/labels/")
    atom_label_path.mkdir(parents=True, exist_ok=True)
    atom_smiles_path = pathlib.Path("atom_relabels/smiles/")
    atom_smiles_path.mkdir(parents=True, exist_ok=True)
    bond_image_path = pathlib.Path("bond_relabels/images/")
    bond_image_path.mkdir(parents=True, exist_ok=True)
    bond_label_path = pathlib.Path("bond_relabels/labels/")
    bond_label_path.mkdir(parents=True, exist_ok=True)
    bond_smiles_path = pathlib.Path("bond_relabels/smiles/")
    bond_smiles_path.mkdir(parents=True, exist_ok=True)
    charge_image_path = pathlib.Path("charge_relabels/images/")
    charge_image_path.mkdir(parents=True, exist_ok=True)
    charge_label_path = pathlib.Path("charge_relabels/labels/")
    charge_label_path.mkdir(parents=True, exist_ok=True)
    charge_smiles_path = pathlib.Path("charge_relabels/smiles/")
    charge_smiles_path.mkdir(parents=True, exist_ok=True)
    stereo_image_path = pathlib.Path("stereo_relabels/images/")
    stereo_image_path.mkdir(parents=True, exist_ok=True)
    stereo_label_path = pathlib.Path("stereo_relabels/labels/")
    stereo_label_path.mkdir(parents=True, exist_ok=True)
    stereo_smiles_path = pathlib.Path("stereo_relabels/smiles/")
    stereo_smiles_path.mkdir(parents=True, exist_ok=True)
    source_data_path = pathlib.Path(args.data_path)
    for idx,molecule_tupple in enumerate(to_label_molecules):
        image_source_path      = source_data_path.joinpath(f"images/{molecule_tupple[0]}.png")
        image = io.imread(image_source_path)
        x_scale=1
        y_scale=1
        if image.shape[0] > 1000 or image.shape[1] > 1000:
           y_=image.shape[0]
           x_=image.shape[1]
           x_scale = 300/x_
           y_scale = 300/y_
           image = resize(image, (300,300))
        xpad=int((1000-image.shape[1])/2)
        ypad=int((1000-image.shape[0])/2)
        #import ipdb; ipdb.set_trace()
        image = np.pad(image, ((int((1000-image.shape[0])/2),int((1000-image.shape[0])/2)),(int((1000-image.shape[1])/2),int((1000-image.shape[1])/2))),'constant', constant_values=(1))
        transform_pars = [x_scale,y_scale,xpad,ypad]
        scaled_image_target_atom_path = atom_scaled_image_path.joinpath(f"{idx}.png")
        io.imsave(scaled_image_target_atom_path,image)
        create_atom_bond_labels(molecule_tupple[1],'chemgrapher_labels',idx,transform_pars)
        image_target_atom_path = atom_image_path.joinpath(f"{idx}.png")
        shutil.copyfile(image_source_path, image_target_atom_path)
        label_atom_path        = atom_label_path.joinpath(f"{idx}.txt")
        create_atom_labels(molecule_tupple[1], label_atom_path)
        smiles_atom_path       = atom_smiles_path.joinpath(f"{idx}.txt")
        with open(smiles_atom_path, "w") as fp:
             fp.write(molecule_tupple[2])
        image_target_bond_path = bond_image_path.joinpath(f"{idx}.png")
        image_source_path      = source_data_path.joinpath(f"images/{molecule_tupple[0]}.png")
        shutil.copyfile(image_source_path, image_target_bond_path)
        label_bond_path        = bond_label_path.joinpath(f"{idx}.txt")
        create_bond_labels(molecule_tupple[1], label_bond_path)
        smiles_bond_path       = bond_smiles_path.joinpath(f"{idx}.txt")
        with open(smiles_bond_path, "w") as fp:
             fp.write(molecule_tupple[2])
        image_target_charge_path = charge_image_path.joinpath(f"{idx}.png")
        image_source_path      = source_data_path.joinpath(f"images/{molecule_tupple[0]}.png")
        shutil.copyfile(image_source_path, image_target_charge_path)
        label_charge_path        = charge_label_path.joinpath(f"{idx}.txt")
        create_charge_labels(molecule_tupple[1], label_charge_path)
        smiles_charge_path       = charge_smiles_path.joinpath(f"{idx}.txt")
        with open(smiles_charge_path, "w") as fp:
             fp.write(molecule_tupple[2])
        image_target_stereo_path = stereo_image_path.joinpath(f"{idx}.png")
        image_source_path      = source_data_path.joinpath(f"images/{molecule_tupple[0]}.png")
        shutil.copyfile(image_source_path, image_target_stereo_path)
        label_stereo_path        = stereo_label_path.joinpath(f"{idx}.txt")
        create_stereo_labels(molecule_tupple[1], label_stereo_path)
        smiles_stereo_path       = stereo_smiles_path.joinpath(f"{idx}.txt")
        with open(smiles_stereo_path, "w") as fp:
             fp.write(molecule_tupple[2])
    return

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='SMILES prediction')
    parser.add_argument('--experiment_path_atoms', type=str, help='path where pretrained atom model was logged')
    parser.add_argument('--experiment_path_bonds', type=str, help='path where pretrained bond model was logged')
    parser.add_argument('--experiment_path_stereo', type=str, help='path where pretrained stereo model was logged')
    parser.add_argument('--experiment_path_charges', type=str, help='path where pretrained charge model was logged')
    data_cls = Objects_Smiles
    parser = data_cls.add_dataset_specific_args(parser)
    model_cls = RCNN
    parser = model_cls.add_dataset_specific_args(parser)
    args = parser.parse_args()
    main(model_cls,data_cls, args)
