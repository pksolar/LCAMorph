import pickle
import random

import numpy as np
import nibabel as nib
import torch

from augmentation.utils import align_img
from augmentation.transformations import AffineTransform


def load_vol(data_name, types="vol"):
    vol_data = None
    if data_name.endswith((".nii.gz", ".nii")):
        vol_data = nib.load(data_name).get_fdata()
    elif data_name.endswith(".pkl"):
        f = open(data_name, "rb")
        if types == "vol":
            vol_data = pickle.load(f)[0]
        else:
            vol_data = pickle.load(f)[1]
    elif data_name.endswith((".npz", ".npy")):
        vol_data = np.load(data_name)["vol_data"]
    else:
        assert "unkown files"

    return vol_data


def save_vol(data, data_name, affine=None):
    if affine is None:
        affine = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)

    nib.Nifti1Image(data, affine).to_filename(data_name)


def data_generator_brain(data, vol_shape=(160, 192, 224), batch_size=1):
    vol_len = len(data)
    ndims = len(vol_shape)

    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    while True:
        idx1 = random.randint(0, vol_len - 1)
        idx2 = random.randint(0, vol_len - 1)
        if idx1 == idx2:
            idx2 = (idx2 + 5) % vol_len

        moving_name = data[idx1]
        fixed_name = data[idx2]
        moving_image, fixed_image = load_vol(moving_name), load_vol(fixed_name)

        inputs = [moving_image.squeeze(), fixed_image.squeeze()]
        outputs = [fixed_image, zero_phi]

        yield inputs, outputs


def data_generator_brain_with_seg(data, data_seg, vol_shape=(160, 192, 224), batch_size=1):
    vol_len = len(data)
    ndims = len(vol_shape)

    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    while True:
        idx1 = random.randint(0, vol_len - 1)

        vol_name = data[idx1]
        seg_name = data_seg[idx1]
        image, segmentation = load_vol(vol_name), load_vol(seg_name)

        inputs = [image.squeeze(), segmentation.squeeze()]
        outputs = [image, zero_phi]

        yield inputs, outputs


def data_generator_lung(data, vol_shape=(160, 192, 224), batch_size=1):
    ndims = len(vol_shape)

    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    vols_list_all = []
    for idx in range(0, 10):
        _ = [name for name in data if "Case" + str(idx) in name]
        if len(_) > 0:
            vols_list_all.append(_)
    for vols_list in vols_list_all:
        print(vols_list)

    while True:
        idx0 = random.randint(0, len(vols_list_all) - 1)
        vols_list = vols_list_all[idx0]

        idx1 = random.randint(0, len(vols_list) - 1)
        idx2 = random.randint(0, len(vols_list) - 1)
        if idx1 == idx2:
            # idx2 = (idx2 + 5) % len(vols_list)
            continue

        moving_name = vols_list[idx1]
        fixed_name = vols_list[idx2]

        moving_image, fixed_image = load_vol(moving_name), load_vol(fixed_name)
        inputs = [moving_image.squeeze(), fixed_image.squeeze()]
        outputs = [fixed_image, zero_phi]

        yield inputs, outputs

