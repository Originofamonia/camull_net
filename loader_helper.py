"""The following module deals with creating the loader he"""
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np

from data_declaration import MRIDataset, Task
from data_declaration import FcmNormalize


class LoaderHelper:
    """An abstract class for assisting with dataset creation."""

    def __init__(self, task: Task = Task.CN_v_AD):

        self.task = task
        self.labels = []

        if task == Task.CN_v_AD:
            self.labels = ["CN", "AD"]
        else:
            self.labels = ["sMCI", "pMCI"]

        self.dataset = MRIDataset(
            data_path="/home/qiyuan/2021fall/camull_net/data",
            labels=self.labels,
            transform=transforms.Compose([
                                      FcmNormalize(),
                                  ]))

        self.indices = []
        self.set_indices()

    def get_task(self):
        """gets task"""
        return self.task

    def get_task_string(self):
        """Gets task string"""
        if self.task == Task.CN_v_AD:
            return "NC_v_AD"
        else:
            return "sMCI_v_pMCI"

    def change_ds_labels(self, labels_in):
        """Function to change the labels of the dataset obj."""
        self.dataset = MRIDataset(data_path="../data/",
                                  labels=labels_in,
                                  transform=transforms.Compose([
                                      FcmNormalize()])
                                  )

    def change_task(self, task: Task):
        """Function to change task of the Datasets"""
        self.task = task

        if task == Task.CN_v_AD:
            self.labels = ["NC", "AD"]
        else:
            self.labels = ["sMCI", "pMCI"]

        self.dataset = MRIDataset(data_path="../data/",
                                  labels=self.labels,
                                  transform=transforms.Compose([
                                      FcmNormalize()])
                                  )

        self.set_indices()

    def set_indices(self, total_folds=5):
        """Abstract function to set indices"""
        test_split = .2
        shuffle_dataset = True
        random_seed = 42

        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(test_split * dataset_size))

        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        fold_indices = []
        lb_split = 0
        ub_split = split

        for _ in range(total_folds):
            train_indices = indices[:lb_split] + indices[ub_split:]
            test_indices = indices[lb_split:ub_split]
            lb_split = split
            ub_split = 2 * split  # only works if kfold is 5 so be carefull
            fold_indices.append((train_indices, test_indices))

        self.indices = fold_indices

    # def make_loaders(self, shuffle=True):
    #     """Makes the loaders"""
    #     fold_indices = self.indices
    #
    #     for k in range(5):
    #         train_ds = Subset(self.dataset, fold_indices[k][0])
    #         test_ds = Subset(self.dataset, fold_indices[k][1])
    #
    #         train_dl = DataLoader(train_ds, batch_size=4, shuffle=shuffle,
    #                               num_workers=4, drop_last=True)
    #         test_dl = DataLoader(test_ds, batch_size=4, shuffle=shuffle,
    #                              num_workers=4, drop_last=True)
    #
    #     print(len(test_ds))
    #
    #     return train_dl, test_dl

    def get_train_dl(self, batch_size=2, shuffle=True):

        # train_ds = Subset(self.dataset, self.indices[fold_ind][0])
        train_dl = DataLoader(self.dataset, batch_size=batch_size,
                              shuffle=shuffle, num_workers=2, drop_last=True)

        return train_dl

    def get_test_dl(self, batch_size=1, shuffle=False):

        # test_ds = Subset(self.dataset, self.indices[fold_ind][1])
        test_dl = DataLoader(self.dataset, batch_size=batch_size,
                             shuffle=shuffle, num_workers=2, drop_last=True)

        return test_dl
