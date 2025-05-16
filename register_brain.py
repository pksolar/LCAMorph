import os
import time
import glob
import argparse

from scipy.ndimage import zoom
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib

from dataset import load_vol, save_vol
from lcamorph.lcamorph import SpatialTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--load_model", type=str, default="./save_model_pt/oasis/lcamorph499.pt")
parser.add_argument("--moving_name", type=str,
                    default="/public/home/gyf/registration/data/OASIS/Test/Vols/brain_395.nii.gz")
parser.add_argument("--fixed_name", type=str,
                    default="/public/home/gyf/registration/data/OASIS/Test/Vols/brain_396.nii.gz")
parser.add_argument("--model_name", type=str, default="lcamorph")
parser.add_argument("--gpu_id", type=str, default="0")

args = parser.parse_args()
load_model = args.load_model
model_name = args.model_name
moving_name = args.moving_name
fixed_name = args.fixed_name
gpu_id = args.gpu_id


def main():
    model_stn = SpatialTransformer((160, 192, 224), "nearest")
    model = torch.load(load_model)
    device = torch.device("cuda:" + gpu_id)

    model.to(device)
    model_stn.to(device)

    moving_image, fixed_image = load_vol(moving_name), load_vol(fixed_name)

    moving = torch.from_numpy(moving_image).to(device)[None, None, ...].float()
    fixed = torch.from_numpy(fixed_image).to(device)[None, None, ...].float()

    model_in = torch.cat((moving, fixed), dim=1)

    warped, flow = model(model_in)

    moving = moving.detach().cpu().numpy().squeeze()
    fixed = fixed.detach().cpu().numpy().squeeze()
    warped = warped.detach().cpu().numpy().squeeze()
    flow = flow.squeeze().permute(1, 2, 3, 0).cuda().data.cpu().numpy()

    save_vol(moving, "./save_image/" + model_name + "_moving.nii.gz")
    save_vol(fixed, "./save_image/" + model_name + "_fixed.nii.gz")
    save_vol(warped, "./save_image/" + model_name + "_warped.nii.gz")
    save_vol(flow, "./save_image/" + model_name + "_flow.nii.gz")


if __name__ == "__main__":
    with torch.no_grad():
        main()
