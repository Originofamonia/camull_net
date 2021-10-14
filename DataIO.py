# this file transform the nii into numpy array and store them in direcotry Data
import nibabel as nib
import os
from nibabel.viewers import OrthoSlicer3D
import numpy as np
from intensity_normalization.normalize.fcm import FCMNormalize


def main():
    data_path = '/home/qiyuan/2021fall/camull_net/data'

    img_names = os.listdir(data_path)
    fcm_norm = FCMNormalize(tissue_type="wm")
    mus = []
    stds = []
    for img_name in img_names:
        print('Image name is', img_name)
        if img_name[-2:] == 'gz':
            img = nib.load(os.path.join(
                data_path, img_name)).get_fdata()  # load the MRI file
            normed_img = fcm_norm(img, modality="t1")
            mus.append(img.mean(axis=(0, 1, 2)))
            stds.append(img.std(axis=(0, 1, 2)))
            OrthoSlicer3D(img).show()  # uncommnet if u want to visualize the MRI image
            print('img dimension is ', img.shape)
            # np.save(img_name[0:-7], img)  # save as numpy format
            # img_numpy = np.load(img_name[0:-7] + '.npy')  # load the npy file
        else:
            continue

    mu = np.asarray(mus).mean(axis=0)
    std = np.asarray(stds).mean(axis=0)
    print(f"mu: {mu}, std: {std}")


# 166 256 256
if __name__ == '__main__':
    main()
