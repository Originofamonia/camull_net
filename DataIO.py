# this file transform the nii into numpy array and store them in direcotry Data
import nibabel as nib
import os
from nibabel.viewers import OrthoSlicer3D
import numpy as np

data_path = './10AD'  # the directory that stores the data

img_names = os.listdir(data_path)

for img_name in img_names:
    print('Image name is', img_name)
    if img_name[-2:] == 'gz':
        img = nib.load(
            data_path + '/' + img_name).get_fdata()  # load the MRI file
        OrthoSlicer3D(img).show()  # uncommnet if u want to visualize the MRI image
        print('img dimension is ', img.shape)
        np.save(img_name[0:-7], img)  # save as numpy format
        img_numpy = np.load(img_name[0:-7] + '.npy')  # load the npy file
    else:
        continue

# 166 256 256
