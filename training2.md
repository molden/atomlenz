# Train AtomLenz on target domain

This step assumes a pretrained AtomLenz model (on synthetic data) and will adapt to a target domain where only SMILES labels are available. 
Pretrained AtomLenz can be obtained here: [pretrained atomlenz](./training.md)

trained models using below steps can be found here:

* Atom model: https://huggingface.co/spaces/moldenhof/atomlenz/resolve/main/models/atoms_model/real.ckpt
* Bond model: https://huggingface.co/spaces/moldenhof/atomlenz/resolve/main/models/bonds_model/real.ckpt
* Charge model: https://huggingface.co/spaces/moldenhof/atomlenz/resolve/main/models/charges_model/real.ckpt
* Stereo center model: https://huggingface.co/spaces/moldenhof/atomlenz/resolve/main/models/stereos_model/real.ckpt

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

# Train (with ProbKT) AtomLenz on each prepared dataset

This step assumes a pretrained AtomLenz model and some ProbKT prepared dataset (previous step). For each prepared dataset from target domain now the pretrained AtomLenz model can be finetuned.

```
python robust_detection/train/train_fine_tune.py --og_data_path path_to_synthetic_dataset --target_data_path atoms_dataset/ --fold 0 --experiment_path locations_of_pretrained_atom_model
```

The locations of the datasets and models needs to be adapted to each type (atoms/bonds/charges/stereos).
For the ``path_to_synthetic_dataset`` we recommend to create a random sample so that the number of samples is in the same order as the number of samples from the dataset in target domain.

# Train (with self correcting mechanism) AtomLenz jointly on target domain

```
python self_label.py --experiment_path_bonds ../ProbKT/robust_detection/train/logger/path_to_bonds_network  --experiment_path_charges ../ProbKT/robust_detection/train/logger/path_to_charge_network --experiment_path_stereo ../ProbKT/robust_detection/train/logger/path_to_stereos_network --experiment_path_atoms ../ProbKT/robust_detection/train/logger/path_to_atom_network  --data_path ../ProbKT/generate_data/path_to_dataset_target_domain --score_thresh 0.65
```
