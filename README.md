# AtomLenz (ATOM-Level ENtity localiZation): Atom-Level Optical Chemical Structure Recogition with Limited Supervision (accepted in CVPR2024)

AtomLenz is a chemical structure recognition tool providing atom-level localization, and can therefore segment the image into the different atoms and bonds. Our method operates by using a self-labeling strategy to generate atom-level annotations, enhancing the dataset’s value, and using this enriched representation to fine-tune the model to a new domain.

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

### Predict SMILES

```
python run_scripts/predict_only_smiles.py --experiment_path_atoms models/atoms_model --experiment_path_bonds models/bonds_model --experiment_path_stereo models/stereos_model --experiment_path_charges models/charges_model --data_path ../datasets/hand_drawn_dataset/test/ --score_thresh 0.65
```

predictions are stored in ``preds_atomlenz`` file.

In case true SMILES are available for dataset also performance metrics can be reported:

```
python run_scripts/predict_smiles.py --experiment_path_atoms models/atoms_model --experiment_path_bonds models/bonds_model --experiment_path_stereo models/stereos_model --experiment_path_charges models/charges_model --data_path <absolute_path>/datasets/hand_drawn_dataset/test/ --score_thresh 0.65
```

predictions are stored in ``preds_atomlenz_long`` file.
