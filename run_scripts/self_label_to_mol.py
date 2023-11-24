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
    for image_idx,molecule in enumerate(mol_graphs):
        base_path = pathlib.Path(args.data_path)
        image_dir = base_path.joinpath("images")
        smiles_dir = base_path.joinpath("smiles")
        impath = image_dir.joinpath(f"{image_idx}.png")
        import cv2
        im = cv2.imread(str(impath))
        height, width, channels = im.shape
        #import ipdb; ipdb.set_trace()
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
           to_label_molecules.append([image_idx,molecule,true_smiles,height,width])
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
              to_label_molecules.append([image_idx,new_molecule,true_smiles,height,width])
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
                 to_label_molecules.append([image_idx,new_molecule,true_smiles,height,width])
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
                    to_label_molecules.append([image_idx,new_molecule,true_smiles,height,width])
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
                       to_label_molecules.append([image_idx,new_molecule,true_smiles,height,width])
        predictions+=1
        #import ipdb; ipdb.set_trace()
    print(f"number correct: {correct}")
    print(f"number correct_new: {correct_new}")
    print(correct_new_smiles)
    print(f"total: {predictions}")
    print(f"relabeling to mols ....")
    mol_image_path = pathlib.Path("mols_relabels/images/")
    mol_image_path.mkdir(parents=True, exist_ok=True)
    mol_label_path = pathlib.Path("mols_relabels/labels/")
    mol_label_path.mkdir(parents=True, exist_ok=True)
    mol_smiles_path = pathlib.Path("mols_relabels/smiles/")
    mol_smiles_path.mkdir(parents=True, exist_ok=True)
    mol_info_path = pathlib.Path("mols_relabels/mol_info.csv")
    mol_info_handle = open(mol_info_path, "a") 
    mol_info_handle.write("file_path,mol_path,raw_SMILES,SMILES,node_coords,edges\n")
    source_data_path = pathlib.Path(args.data_path)
    for idx,molecule_tupple in enumerate(to_label_molecules):
        image_target_mol_path = mol_image_path.joinpath(f"{idx}.png")
        image_source_path      = source_data_path.joinpath(f"images/{molecule_tupple[0]}.png")
        shutil.copyfile(image_source_path, image_target_mol_path)
        label_mol_path        = mol_label_path.joinpath(f"{idx}.txt")
        mol_atom_list,mol_bond_list = save_mol_to_file(molecule_tupple[1],label_mol_path,return_mol_info=1,height=molecule_tupple[3],width=molecule_tupple[4])
        #create_atom_labels(molecule_tupple[1], label_atom_path)
        smiles_mol_path       = mol_smiles_path.joinpath(f"{idx}.txt")
        with open(smiles_mol_path, "w") as fp:
             fp.write(molecule_tupple[2])
        #file_path,mol_path,raw_SMILES,SMILES,node_coords,edges
        mol_extra_info=f"{image_target_mol_path},{label_mol_path},{molecule_tupple[2]},{molecule_tupple[2]},\"{mol_atom_list}\",\"{mol_bond_list}\"\n"
        mol_info_handle.write(mol_extra_info)
    mol_info_handle.close()
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
