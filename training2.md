# Train AtomLenz on target domain

This step assumes a pretrained AtomLenz model (on synthetic data) and will adapt to a target domain where only SMILES labels are available. 
Pretrained AtomLenz can be obtained here: [pretrained atomlenz](./training.md)

# Prepare dataset

AtomLenz can be trained on images with only SMILES labels. One such a dataset is available here: https://dx.doi.org/10.6084/m9.figshare.24599412 . 
This is a dataset with hand drawn images labeled with the corresponding SMILES.

Once downloaded this dataset should be prepared for use by ProbKT. This can be done with  [parse smiles](./datasets/parse_smiles.py) script.

```
python parse_smiles.py --datasetfolder hand_drawn_train/train/ --outputfolder atoms_dataset/
```

This should be repeated for bonds, charges and stereocenters:

```
python parse_smiles.py --datasetfolder hand_drawn_train/train/ --outputfolder bonds_dataset/ --parse_bonds 1
```

```
python parse_smiles.py --datasetfolder hand_drawn_train/train/ --outputfolder charges_dataset/ --parse_charges 1
```

```
python parse_smiles.py --datasetfolder hand_drawn_train/train/ --outputfolder stereos_dataset/ --parse_stereos 1
```
