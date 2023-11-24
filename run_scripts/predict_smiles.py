import os
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import pathlib
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
from rdkit.Chem import AllChem
from rdkit import DataStructs

def tanimoto_calc(smi1, smi2):
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 3, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 3, nBits=2048)
    s = round(DataStructs.TanimotoSimilarity(fp1,fp2),3)
    return s

def main(model_cls, data_cls, args, logger = None):
    atomnumber_dict = {6: 1, 1: 2, 7: 3, 8: 4, 16: 5, 9: 6, 17: 7, 35: 8, 53: 9, 34: 10, 15: 11, 5: 12, 14: 13, 19: 14, 33: 14, 36: 14, 51: 14, 74: 14, 32: 14, 50: 14, 80: 14, 52: 14, 92: 14, 49: 14, 75: 14, 81: 14, 40: 14, 13: 14, 23: 14, 82: 14, 42: 14, 20: 14, 46: 14, 24: 14, 26: 14, 83: 14, 78: 14, 41: 14, 45: 14}
    dict_bond = {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3}
    experiment_path_atoms = args.experiment_path_atoms
    dir_list = os.listdir(experiment_path_atoms)
    dir_list = [os.path.join(experiment_path_atoms,f) for f in dir_list]
    dir_list.sort(key=os.path.getctime, reverse=True)
    checkpoint_file_atoms = [f for f in dir_list if "ckpt" in f][0]
    dataset = data_cls(**vars(args))
    dataset.prepare_data()
    model_atom = model_cls.load_from_checkpoint(checkpoint_file_atoms)
    model_atom.model.roi_heads.score_thresh = args.score_thresh
    trainer = pl.Trainer(logger=False, gpus=1)
    atom_preds = trainer.predict(model_atom, dataset.test_dataloader())
    experiment_path_bonds = args.experiment_path_bonds
    dir_list = os.listdir(experiment_path_bonds)
    dir_list = [os.path.join(experiment_path_bonds,f) for f in dir_list]
    dir_list.sort(key=os.path.getctime, reverse=True)
    checkpoint_file_bonds = [f for f in dir_list if "ckpt" in f][0]
    model_bond = model_cls.load_from_checkpoint(checkpoint_file_bonds)
    bond_preds = trainer.predict(model_bond, dataset.test_dataloader())
    experiment_path_stereo = args.experiment_path_stereo
    dir_list = os.listdir(experiment_path_stereo)
    dir_list = [os.path.join(experiment_path_stereo,f) for f in dir_list]
    dir_list.sort(key=os.path.getctime, reverse=True)
    checkpoint_file_stereo = [f for f in dir_list if "ckpt" in f][0]
    model_stereo = model_cls.load_from_checkpoint(checkpoint_file_stereo)
    stereo_preds = trainer.predict(model_stereo, dataset.test_dataloader())
    experiment_path_charges = args.experiment_path_charges
    dir_list = os.listdir(experiment_path_charges)
    dir_list = [os.path.join(experiment_path_charges,f) for f in dir_list]
    dir_list.sort(key=os.path.getctime, reverse=True)
    checkpoint_file_charges = [f for f in dir_list if "ckpt" in f][0]
    model_charges = model_cls.load_from_checkpoint(checkpoint_file_charges)
    charges_preds = trainer.predict(model_charges, dataset.test_dataloader())
    mol_graphs = []
    count_bonds_preds = np.zeros(4)
    count_atoms_preds = np.zeros(15)
    correct=0
    correct_objects=0
    correct_both=0
    predictions=0
    tanimoto_dists=[]
    predictions_list = []
    for image_idx, bonds in enumerate(bond_preds):
        count_bonds_preds = np.zeros(8)
        count_atoms_preds = np.zeros(18)
        atom_boxes = atom_preds[image_idx]['boxes'][0]
        atom_labels = atom_preds[image_idx]['preds'][0]
        atom_scores = atom_preds[image_idx]['scores'][0]
        charge_boxes = charges_preds[image_idx]['boxes'][0]
        charge_labels = charges_preds[image_idx]['preds'][0]
        charge_mask=torch.where(charge_labels>1)
        filtered_ch_labels=charge_labels[charge_mask]
        filtered_ch_boxes=charge_boxes[charge_mask]
        #import ipdb; ipdb.set_trace()
        filtered_bboxes, filtered_labels = iou_filter_bboxes(atom_boxes, atom_labels, atom_scores)
        #for atom_label in filtered_labels:
        #    count_atoms_preds[atom_label] += 1
        #import ipdb; ipdb.set_trace()
        mol_graph = np.zeros((len(filtered_bboxes),len(filtered_bboxes)))
        stereo_atoms = np.zeros(len(filtered_bboxes))
        charge_atoms = np.ones(len(filtered_bboxes))
        for index,box_atom in enumerate(filtered_bboxes):
            for box_charge,label_charge in zip(filtered_ch_boxes,filtered_ch_labels):
                if bb_box_intersects(box_atom,box_charge) == 1:
                    charge_atoms[index]=label_charge
            
        for bond_idx, bond_box in enumerate(bonds['boxes'][0]):
            label_bond = bonds['preds'][0][bond_idx]
            if label_bond > 1:
              try:
                 count_bonds_preds[label_bond] += 1
              except:
                 count_bonds_preds=count_bonds_preds 
               #import ipdb; ipdb.set_trace()
              result = []
              limit = 0
            #TODO: values of 50 and 5 should be made dependent of mean size of atom_boxes
              while result.count(1) < 2 and limit < 80:
                 result=[]
                 bigger_bond_box = [bond_box[0]-limit,bond_box[1]-limit,bond_box[2]+limit,bond_box[3]+limit]
                 for atom_box in filtered_bboxes:
                     result.append(bb_box_intersects(atom_box,bigger_bond_box))
                 limit+=5
              indices = [i for i, x in enumerate(result) if x == 1]
              if len(indices) == 2:
               #import ipdb; ipdb.set_trace()
                 mol_graph[indices[0],indices[1]]=label_bond
                 mol_graph[indices[1],indices[0]]=label_bond
              if len(indices) > 2:
                #we have more then two canidate atoms for one bond, we filter ...
                  cand_bboxes = filtered_bboxes[indices,:]
                  cand_indices = dist_filter_bboxes(cand_bboxes)
                #import ipdb; ipdb.set_trace()
                  mol_graph[indices[cand_indices[0]],indices[cand_indices[1]]]=label_bond
                  mol_graph[indices[cand_indices[1]],indices[cand_indices[0]]]=label_bond
                  #print("more than 2 indices")
              #if len(indices) < 2:
              #    print("less than 2 indices")
                #import ipdb; ipdb.set_trace()
 #           else:
 #             result=[]
 #             for atom_box in filtered_bboxes:
 #                 result.append(bb_box_intersects(atom_box,bond_box))
 #             indices = [i for i, x in enumerate(result) if x == 1]
 #             if len(indices) == 1:
 #                stereo_atoms[indices[0]]=label_bond
        stereo_bonds = np.where(mol_graph>4, True, False)
        if np.any(stereo_bonds):
           stereo_boxes = stereo_preds[image_idx]['boxes'][0]
           stereo_labels= stereo_preds[image_idx]['preds'][0]
           for stereo_box in stereo_boxes:
               result=[]
               for atom_box in filtered_bboxes:
                   result.append(bb_box_intersects(atom_box,stereo_box))
               indices = [i for i, x in enumerate(result) if x == 1]
               if len(indices) == 1:
                   stereo_atoms[indices[0]]=1
               
        molecule = dict()
        molecule['graph'] = mol_graph
        #molecule['atom_labels'] = atom_preds[image_idx]['preds'][0]
        molecule['atom_labels'] = filtered_labels
        molecule['atom_boxes'] = filtered_bboxes
        molecule['stereo_atoms'] = stereo_atoms
        molecule['charge_atoms'] = charge_atoms
        mol_graphs.append(molecule)
        base_path = pathlib.Path(args.data_path)
        image_dir = base_path.joinpath("images")
        smiles_dir = base_path.joinpath("smiles")
        impath = image_dir.joinpath(f"{image_idx}.png")
        smilespath = smiles_dir.joinpath(f"{image_idx}.txt")
       # visualize_atom_bonds(impath, filtered_bboxes, bonds['boxes'][0])
       # import ipdb; ipdb.set_trace()
        with open(smilespath) as f:
             true_smiles = f.readline().strip('\n')
        m = Chem.MolFromSmiles(true_smiles)
        #try:
        #  m=neutralize_atoms(m)
        #  true_smiles = Chem.MolToSmiles(m,isomericSmiles=False)
        #except:
        #    print(f"number of predictions:{predictions}")
        #    print(f"number correct:{correct}")
        #count_atoms = np.zeros(15)
        #count_bonds = np.zeros(4)
        #m        = Chem.MolFromSmiles(true_smiles)
        #try:
        #  Chem.Kekulize(m)
        #  for bond in m.GetBonds():
        #      count_bonds[dict_bond.get(str(bond.GetBondType()))] +=1
        #  for atom in m.GetAtoms():
        #      count_atoms[atomnumber_dict.get(atom.GetAtomicNum(),14)] +=1
        #except:
        #  print(f"number of predictions:{predictions}")
        #  print(f"number correct:{correct}")
       # import ipdb; ipdb.set_trace()
        save_mol_to_file(molecule,'molfile')
        mol =  Chem.MolFromMolFile('molfile',sanitize=False)
        problematic = 0
        try:
          problems = Chem.DetectChemistryProblems(mol)
          if len(problems) > 0:
             mol = solve_mol_problems(mol,problems)
             problematic = 1
           #import ipdb; ipdb.set_trace()
          try:
            Chem.SanitizeMol(mol) 
          except:
            problems = Chem.DetectChemistryProblems(mol)
            if len(problems) > 0:
              mol = solve_mol_problems(mol,problems)
            try:
              Chem.SanitizeMol(mol)
            except:
              pass
        except:
          problematic = 1
        try:
          pred_smiles = Chem.MolToSmiles(mol)
        except:
          pred_smiles = ""
        try:
          true_smiles = Chem.MolToSmiles(m)
        except:
          true_smiles = true_smiles
        #import ipdb; ipdb.set_trace()
      #  pred = generateSmiles(molecule)
      #  m_pred=Chem.MolFromSmiles(pred)
      #  try:
      #     pred=Chem.MolToSmiles(m_pred,isomericSmiles=False)
      #  except:
      #     pred=pred
      #  if np.array_equal(count_bonds_preds, count_bonds):
      #      if np.array_equal(count_atoms_preds, count_atoms):
      #          correct_objects += 1
      #          if pred==true_smiles:
      #              correct_both +=1
      #          else:
      #              visualize_atom_bonds(impath, filtered_bboxes, bonds['boxes'][0])
      #              import ipdb; ipdb.set_trace()
        match_smiles=0
        if pred_smiles==true_smiles:
           correct+=1
           match_smiles=1
    #    if pred_smiles == "":
    #       tanimoto_dists.append(0)
    #    else:
        tanimoto_dist = 0
        if Chem.MolFromSmiles(pred_smiles) is not None:
           if Chem.MolFromSmiles(true_smiles) is not None:
              tanimoto_dist = tanimoto_calc(pred_smiles,true_smiles)
        tanimoto_dists.append(tanimoto_dist)
       # else:
       #    tanimoto_dists.append(0)
        print(f"true {true_smiles}")
        print(f"pred {pred_smiles}")
        predictions+=1
        predictions_list.append([image_idx,pred_smiles,match_smiles,problematic])
                #import ipdb; ipdb.set_trace()
    file_preds = open('preds_handdrawn_train_wo_dangling_chemgrapher2','w')
    for pred,tanimoto in zip(predictions_list,tanimoto_dists):
        file_preds.write(f"{pred[0]},{pred[1]},{tanimoto},{pred[2]},{pred[3]}\n")
        #import ipdb; ipdb.set_trace()
    print(f"number correct: {correct}")
   # print(f"number correct objects: {correct_objects}")
   # print(f"number correct both: {correct_both}")
    print(f"total: {predictions}")
    print(f"average tanimoto dist:{sum(tanimoto_dists)/len(tanimoto_dists)}")
    print(f"sum: {sum(tanimoto_dists)}")
    print(f"len: {len(tanimoto_dists)}")
    #import ipdb; ipdb.set_trace()
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
