"""The following module declares the Dataset objects required by torch to
iterate over the data. """
from enum import Enum
import glob
import os

import numpy as np
import nibabel as nib
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader


class Task(Enum):
    """
        Enum class for the two classification tasks
    """
    CN_v_AD = 1
    sMCI_v_pMCI = 2


def get_im_id(path):
    """Gets the image id from the file path string"""
    fname = path.stem
    im_id_str = ""
    # the I that comes before the id needs to be removed hence [1:]
    im_id_str = fname.split("_")[-1][1:]
    return int(im_id_str)


def get_ptid(path):
    """Gets the ptid from the file path string"""
    fname = path.stem
    ptid_str = ""
    count = 0
    for char in fname:
        if count == 4:
            break
        if 0 < count < 4:
            ptid_str += char

        if char == '_':
            count += 1

    return ptid_str[:-1]


def get_acq_year(im_data_id, im_df):
    """Gets the acquisition year from a pandas dataframe by searching the
    image id """
    acq_date = im_df[im_df['Image Data ID'] == im_data_id]["Acq Date"].iloc[0]
    acq_year_str = ""

    slash_count = 0
    for char in acq_date:
        if char == "/":
            slash_count += 1

        if slash_count == 2:
            acq_year_str += char

    return acq_year_str[1:]


def get_label(path, labels):
    """Gets label from the path"""
    label_str = path.parent.stem
    label = None

    if label_str == labels[0]:
        label = np.array([0], dtype=np.double)
    elif label_str == labels[1]:
        label = np.array([1], dtype=np.double)
    return label


def get_mri(path):
    """Gets a numpy array representing the mri object from a file path"""
    mri = nib.load(str(path)).get_fdata()
    mri.resize(1, 110, 110, 110)
    mri = np.asarray(mri)

    return mri


def get_clinical(im_id, clin_df):
    """Gets clinical features vector by searching dataframe for image id"""
    clinical = np.zeros(21)

    row = clin_df.loc[clin_df["Image Data ID"] == im_id]

    for k in range(1, 22):
        clinical[k - 1] = row.iloc[0][k]

    return clinical


class MRIDataset(Dataset):
    """Provides an object for the MRI data that can be iterated."""

    def __init__(self, data_path, labels, transform=None):

        self.data_path = data_path
        self.transform = transform
        self.img_names = os.listdir(data_path)
        self.labels = labels

        self.len = len(self.img_names)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img = nib.load(os.path.join(
            self.data_path, img_name)).get_fdata()
        img = img.astype(float)
        label = np.array(self.labels.index(img_name[:2]))
        label = np.expand_dims(label, axis=0)

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.from_numpy(label).type(torch.FloatTensor)


class FcmNormalize:
    """
    add normalize here
    https://github.com/jcreinhold/intensity-normalization
    try FCM-based WM-based normalization
    (assuming you have access to a T1-w image for all the time-points)
    """

    def __call__(self, image):
        mri_t = torch.from_numpy(image)
        mri_t = mri_t.unsqueeze(0)
        return mri_t
