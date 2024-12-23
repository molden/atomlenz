# (Pre)Train AtomLenz

This is the pretraining step of AtomLenz on synthetically generated data with localization labels.
The pretrained models (following these steps) are made available here:
* Atom model: https://huggingface.co/spaces/moldenhof/atomlenz/resolve/main/models/atoms_model/synthetic.ckpt
* Bond model: https://huggingface.co/spaces/moldenhof/atomlenz/resolve/main/models/bonds_model/synthetic.ckpt
* Charge model: https://huggingface.co/spaces/moldenhof/atomlenz/resolve/main/models/charges_model/synthetic.ckpt
* Stereo center model: https://huggingface.co/spaces/moldenhof/atomlenz/resolve/main/models/stereos_model/synthetic.ckpt

## Download synthetically generated datasets

In order to pretrain the different (atom-level) object detection backbones of AtomLenz, datasets were synthetically generated and are made available here:

* part 1 atom and bond entity annotated images: https://zenodo.org/records/10185264
* part 2 charge and stereocenter entity annotated images: https://zenodo.org/records/10200185

## Install ProbKT

As (part of) the training of AtomLenz is based on ProbKT, we refer you to the installation instructions of ProbKT here: https://github.com/molden/ProbKT .

## Configuration

Once ProbKT is installed, the downloaded datasets need to be copied (and extracted) to the ``generate_data`` folder of ProbKT.

## Run first training round (pretraining)

For the pretraining of the RCNN object detection models of AtomLenz the following command needs to be executed for each dataset (atoms/bonds/charges/stereo-centers):

```
python robust_detection/train/train_rcnn.py --data_path path_to_datasetfolder_inside_generated_data
```

## Run second training round

The second traing round is descriped here: [train atomlenz](./training2.md)
