# Advancing Chemical Structure Recognition in Hand-Drawn Images by AtomLenz (ATOM-Level ENtity localiZation)

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

### Predict images to SMILES

```
python run_scripts/predict_only_smiles.py --experiment_path_atoms models/atoms_model --experiment_path_bonds models/bonds_model --experiment_path_stereo models/stereos_model --experiment_path_charges models/charges_model --data_path datasets/hand_drawn_dataset/test/ --score_thresh 0.65
```
