from torch.utils.data import Dataset, DataLoader, Subset
from robust_detection.data_utils.rcnn_data_utils import *
import pytorch_lightning as pl
import robust_detection.transforms as T

DATA_FOLDER = os.path.join(os.path.dirname(__file__))
def get_transform():
        transforms = []
        transforms.append(T.ToTensor())
        return T.Compose(transforms)

class Objects_Smiles(pl.LightningDataModule):
    def __init__(self, data_path, **kwargs):
                super().__init__()
                self.batch_size = 1
                self.num_workers = 4
                self.data_path = data_path
                self.transforms = get_transform()
                self.base_class = Objects_Detection_Predictor_Dataset
    def prepare_data(self):
                dataset = self.base_class(os.path.join(DATA_FOLDER, self.data_path), self.transforms)
                self.train = dataset
                self.test  = dataset
                self.val   = dataset
                
                self.test_ood = dataset

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_tuple
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_tuple
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_tuple
        )

    def test_ood_dataloader(self, shuffle=False):
        return DataLoader(
            self.test_ood,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_tuple
        )
   
    @classmethod
    def add_dataset_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--data_path', type=str,
                            default="mnist/alldigits/")
        return parser


class Objects_fold_Smiles(pl.LightningDataModule):
    def __init__(self, data_path, fold, **kwargs):
                super().__init__()
                self.batch_size = 1
                self.num_workers = 4
                self.data_path = data_path
                self.fold = fold
                self.transforms = get_transform()
               # self.base_class = Objects_Detection_Predictor_Dataset
                self.base_class = Objects_Detection_Dataset
    def prepare_data(self):
                dataset = self.base_class(os.path.join(DATA_FOLDER, self.data_path), self.transforms)
                if self.fold > -1:
                   train_idx = np.load(os.path.join(DATA_FOLDER, f"{self.data_path}", "../folds", str(self.fold), "train_idx.npy"))
                   self.train = Subset(dataset, train_idx)
                   val_idx = np.load(os.path.join(DATA_FOLDER, f"{self.data_path}", "../folds", str(self.fold), "val_idx.npy"))

                   self.val = Subset(dataset, val_idx)
                else:
                   self.train = dataset
                   self.val   = dataset
                self.test = self.val
                self.test_ood = self.test

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_tuple
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_tuple
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_tuple
        )
    
    def test_ood_dataloader(self, shuffle=False):
        return DataLoader(
            self.test_ood,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_tuple
        )

    @classmethod
    def add_dataset_specific_args(cls, parent):
        import argparse
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
        parser.add_argument('--data_path', type=str,
                            default="mnist/alldigits/")
        parser.add_argument('--fold', type=int,
                            default=0)
        return parser
