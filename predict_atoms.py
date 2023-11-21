import os
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from utils_graph import *
from Object_Smiles import Objects_Smiles 
#from torchmetrics.detection.map import MeanAveragePrecision
from torchmetrics.detection.mean_ap import MeanAveragePrecision

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
        
    Y = []
    Y_hat = []
    map_metric = MeanAveragePrecision()
    map_metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.05,0.1,0.15,0.2,0.25,0.3,0.35])
    for pred in atom_preds:
        Y += pred["targets"]
        Y_hat += pred["preds"]
        pred_map = [dict(boxes=pred["boxes"][i],scores=pred["scores"][i],labels=pred["preds"][i]) for i in range(len(pred["targets"]))]
        target_map = [dict(boxes=pred["boxes_true"][i],labels=pred["targets"][i]) for i in range(len(pred["targets"]))]
        import ipdb; ipdb.set_trace()
        map_metric.update(pred_map,target_map)
        #import ipdb; ipdb.set_trace()
    mAP = map_metric.compute()
    accuracy = np.array([torch.equal(Y[i].sort()[0],Y_hat[i].sort()[0]) for i in range(len(Y))]).mean()
    print(accuracy)
    print(mAP)
    #atom_labels = atom_preds[image_idx]['preds'][0]
    #atom_scores = atom_preds[image_idx]['scores'][0]
    #import ipdb; ipdb.set_trace()
    return

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='SMILES prediction')
    parser.add_argument('--experiment_path_atoms', type=str, help='path where pretrained atom model was logged')
    #data_cls = Objects_Smiles
    data_cls = Objects_RCNN
    parser = data_cls.add_dataset_specific_args(parser)
    model_cls = RCNN
    parser = model_cls.add_dataset_specific_args(parser)
    args = parser.parse_args()
    main(model_cls,data_cls, args)
