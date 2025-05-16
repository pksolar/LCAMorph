import os
import time
import copy
import glob
import argparse

import torch
from torch import optim
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import losses
import utils
from dataset import data_generator_brain_with_seg, load_vol, random_affine_augment

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="lcamorph")
parser.add_argument("--train_data", type=str, default="/public/home/root/registration/data/SRReg/Train/")
parser.add_argument("--max_epoches", type=int, default=500)
parser.add_argument("--base_lr", type=float, default=0.0001)
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--aug", type=bool, default=True)
parser.add_argument("--adj", type=bool, default=True)

args = parser.parse_args()
model_name = args.model_name
lr = args.base_lr
train_data = args.train_data
max_epoches = args.max_epoches
gpu_id = args.gpu_id
aug = args.aug
adj = args.adj


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def train():
    train_database_ct = glob.glob(train_data + "CT/Vols/" + "*.nii.gz")
    train_database_ct_seg = glob.glob(train_data + "CT/Segs/" + "*.nii.gz")
    train_database_mr = glob.glob(train_data + "MR/Vols/" + "*.nii.gz")
    train_database_mr_seg = glob.glob(train_data + "MR/Segs/" + "*.nii.gz")
    steps_per_epoch = 100
    device = torch.device("cuda:" + gpu_id)
    criterions = [losses.Dice_mambamorph().loss, losses.Grad3d(penalty='l2')]
    weights = [1.0, 0.1]
    vol_shape = (160, 192, 224)

    if model_name == "lcamorph":
        from lcamorph.lcamorph import LCAMorph
        model = LCAMorph(vol_shape)
    from lcamorph.lcamorph import SpatialTransformer
    model_stn = SpatialTransformer((160, 192, 224), "bilinear")

    model_stn.to(device)
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    train_gen_ct = data_generator_brain_with_seg(train_database_ct, train_database_ct_seg)
    train_gen_mr = data_generator_brain_with_seg(train_database_mr, train_database_mr_seg)

    good_labels = [0., 2., 3., 4., 5., 7., 8., 10., 11., 12., 13., 14., 15., 16., 17., 18., 24., 26., 28.]
    for epoch in range(0, max_epoches):
        start_time = time.time()
        if adj:
            adjust_learning_rate(optimizer, epoch, max_epoches, lr)
        for idx in range(steps_per_epoch):
            data, _ = next(train_gen_ct)
            moving_vol = data[0]
            moving_seg = data[1]
            data, _ = next(train_gen_mr)
            fixed_vol = data[0]
            fixed_seg = data[1]

            moving_vol, moving_seg = moving_vol[None, None, ...], moving_seg[None, None, ...]
            fixed_vol, fixed_seg = fixed_vol[None, None, ...], fixed_seg[None, None, ...]
            moving_seg = utils.split_seg_global(moving_seg, good_labels)
            fixed_seg = utils.split_seg_global(fixed_seg, good_labels)

            moving_vol = torch.from_numpy(moving_vol).to(device).float()
            fixed_vol = torch.from_numpy(fixed_vol).to(device).float()
            moving_seg = torch.from_numpy(moving_seg).to(device).float().permute(0, 4, 1, 2, 3)
            fixed_seg = torch.from_numpy(fixed_seg).to(device).float().permute(0, 4, 1, 2, 3)

            if aug:
                max_random_params = (0.2, 0.05, 3.1416 / 18, 0.2)
                moving_vol, moving_seg = random_affine_augment(moving_vol, moving_seg,
                                                               max_random_params=max_random_params, scale_params=1,
                                                               return_affine_matrix=False, )
                fixed_vol, fixed_seg = random_affine_augment(fixed_vol, fixed_seg, max_random_params=max_random_params,
                                                             scale_params=1,
                                                             return_affine_matrix=False, )

            model_in = torch.cat((moving_vol, fixed_vol), dim=1)
            _, flow = model(model_in)
            warped_seg = model_stn(moving_seg, flow)
            loss = 0.0
            curr_loss = [0, 0]
            curr_loss[0] += criterions[0](warped_seg, fixed_seg) * weights[0]
            curr_loss[1] += criterions[1](flow, flow) * weights[1]
            loss += curr_loss[0] + curr_loss[1]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model_in = torch.cat((fixed_vol, moving_vol), dim=1)
            _, flow = model(model_in)
            warped_seg = model_stn(fixed_seg, flow)
            loss = 0.0
            curr_loss = [0, 0]
            curr_loss[0] += criterions[0](warped_seg, moving_seg) * weights[0]
            curr_loss[1] += criterions[1](flow, flow) * weights[1]
            loss += curr_loss[0] + curr_loss[1]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            end_time = time.time()
        print(
            "{} of Epoch {}, Loss: {:.4f}, Sim: {:.4f}, Reg: {:.4f}. Cost Time: {:.2f}".format(epoch, max_epoches,
                                                                                               loss.item(),
                                                                                               curr_loss[0].item(),
                                                                                               curr_loss[1].item(),
                                                                                               end_time - start_time))

        if epoch % 1 == 0:
            torch.save(model, os.path.join("./save_model_pt/srreg", model_name + str(epoch) + ".pt"))


if __name__ == "__main__":
    train()
