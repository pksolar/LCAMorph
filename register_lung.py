import os
import time
import glob
import argparse

import nibabel as nib
from scipy.ndimage import zoom
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import point_spatial_transformer
from dataset import save_vol

parser = argparse.ArgumentParser()
parser.add_argument("--load_model", type=str, default="./save_model_pt/dirlab/lcamorph100.pt")
parser.add_argument("--case_id", type=str, default="8")
parser.add_argument("--model_name", type=str, default="lcamorph")
parser.add_argument("--test_data", type=str, default="/public/home/root/registration/data/DIRLAB/Case")
parser.add_argument("--gpu_id", type=str, default="0")

args = parser.parse_args()
load_model = args.load_model
case_id = args.case_id
model_name = args.model_name
test_data = args.test_data
gpu_id = args.gpu_id


def main():
    device = "cuda:" + gpu_id
    model = torch.load(load_model, map_location=device)
    with torch.no_grad():
        mov_name = test_data + case_id + "/new_vols/case" + case_id + "_T00.nii.gz"
        fix_name = test_data + case_id + "/new_vols/case" + case_id + "_T50.nii.gz"
        mov_p_name = test_data + case_id + "/new_landmarks/case" + case_id + "_T00.txt"
        fix_p_name = test_data + case_id + "/new_landmarks/case" + case_id + "_T50.txt"

        moving = nib.load(mov_name).get_fdata()
        fixed = nib.load(fix_name).get_fdata()
        moving_image_landmarks = np.loadtxt(mov_p_name)
        fixed_image_landmarks = np.loadtxt(fix_p_name)

        moving = torch.from_numpy(moving).to(device)
        fixed = torch.from_numpy(fixed).to(device)

        moving = moving[None, None, ...].float()
        fixed = fixed[None, None, ...].float()

        model_in = torch.cat((moving, fixed), dim=1)
        warped, flow = model(model_in)

        fixed_image_landmarks = fixed_image_landmarks[None, ...]
        moving_image_landmarks = moving_image_landmarks
        flow = flow.squeeze().permute(1, 2, 3, 0)
        dvf_Data = flow.cuda().data.cpu().numpy().squeeze()
        data = [torch.from_numpy(t).cuda() for t in [fixed_image_landmarks, dvf_Data]]
        moved_image_landmarks = point_spatial_transformer(data)

        moved_image_landmarks = moved_image_landmarks.cuda().data.cpu().numpy().squeeze()
        # moving_image_landmarks = moving_image_landmarks.cuda().data.cpu().numpy().squeeze()
        # fixed_image_landmarks = fixed_image_landmarks.cuda().data.cpu().numpy().squeeze()
        np.savetxt('./save_txt/moved_' + model_name + "_" + case_id + '.txt', moved_image_landmarks, delimiter=" ",
                   fmt='%.3f')
        np.savetxt('./save_txt/moving_' + model_name + "_" + case_id + '.txt', moving_image_landmarks, delimiter=" ",
                   fmt='%.3f')

        moving = moving.detach().cpu().numpy().squeeze()
        fixed = fixed.detach().cpu().numpy().squeeze()
        warped = warped.detach().cpu().numpy().squeeze()
        flow = flow.squeeze().permute(1, 2, 3, 0).cuda().data.cpu().numpy()

        save_vol(moving, "./save_image/" + model_name + "_moving.nii.gz")
        save_vol(fixed, "./save_image/" + model_name + "_fixed.nii.gz")
        save_vol(warped, "./save_image/" + model_name + "_warped.nii.gz")
        save_vol(flow, "./save_image/" + model_name + "_flow.nii.gz")


if __name__ == "__main__":
    main()
