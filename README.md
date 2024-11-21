# AtomLenz (ATOM-Level ENtity localiZation)


AtomLenz is a chemical structure recognition tool providing atom-level localization, and can therefore segment the image into the different atoms and bonds. Our method operates by using a self-labeling strategy to generate atom-level annotations, enhancing the dataset’s value, and using this enriched representation to fine-tune the model to a new domain.

## Reference

If you like this work, consider citing our related paper accepted in ***CVPR 2024***:

```
@article{oldenhof2024atom,
  title={Atom-Level Optical Chemical Structure Recognition with Limited Supervision},
  author={Oldenhof, Martijn and De Brouwer, Edward and Arany, Adam and Moreau, Yves},
  journal={arXiv preprint arXiv:2404.01743},
  year={2024}
}
```
## Huggingface Space

Please check out our huggingface space if you want to quickly test AtomLenz:

[AtomLenz Space](https://huggingface.co/spaces/moldenhof/atomlenz)
## Prerequisites

install [ProbKT](https://github.com/molden/ProbKT)

## Install

install AtomLenz:

``
pip install -e .
``


## Getting Started

download datasets in [datasets folder](./datasets/README.md)

download models in [models folder](./models/README.md)

## Predict SMILES

```
python run_scripts/predict_only_smiles.py --experiment_path_atoms models/atoms_model --experiment_path_bonds models/bonds_model --experiment_path_stereo models/stereos_model --experiment_path_charges models/charges_model --data_path ../datasets/hand_drawn_dataset/test/ --score_thresh 0.65
```

predictions are stored in ``preds_atomlenz`` file.

In case true SMILES are available for dataset also performance metrics can be reported:

```
python run_scripts/predict_smiles.py --experiment_path_atoms models/atoms_model --experiment_path_bonds models/bonds_model --experiment_path_stereo models/stereos_model --experiment_path_charges models/charges_model --data_path <absolute_path>/datasets/hand_drawn_dataset/test/ --score_thresh 0.65
```

predictions are stored in ``preds_atomlenz_long`` file.

## Pretraining AtomLenz

The procedure to pretrain AtomLenz is described here: [pretrained atomlenz](./training.md)

## Train AtomLenz on target domain

The procedure to train AtomLenz on target domain is described here: [train atomlenz](./training2.md)
