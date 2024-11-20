# Train AtomLenz on target domain

This step assumes a pretrained AtomLenz model (on synthetic data) and will adapt to a target domain where only SMILES labels are available. Pretrained AtomLenz can be obtained here: [pretrained atomlenz](./training.md)

# Prepare dataset

AtomLenz can be trained on images with only SMILES labels. One such a dataset is available here: https://dx.doi.org/10.6084/m9.figshare.24599412 . This is a dataset with hand drawn images labeled with the corresponding SMILES.
