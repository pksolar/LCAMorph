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
from dataset import data_generator_lung, load_vol, random_affine_augment

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="lcamorph")
parser.add_argument("--train_data", type=str, default="/public/home/root/registration/data/DIRLAB/")
parser.add_argument("--max_epoches", type=int, default=100)
parser.add_argument("--base_lr", type=float, default=0.0001)
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--case_id_test1", type=str, default="0")
parser.add_argument("--case_id_test2", type=str, default="9")
parser.add_argument("--case_id_val", type=str, default="1")
parser.add_argument("--aug", type=bool, default=True)
parser.add_argument("--adj", type=bool, default=True)
args = parser.parse_args()
model_name = args.model_name
lr = args.base_lr
train_data = args.train_data
max_epoches = args.max_epoches
gpu_id = args.gpu_id
case_id_test1 = args.case_id_test1
case_id_test2 = args.case_id_test2
case_id_val = args.case_id_val
aug = args.aug
adj = args.adj

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def train():
    case_ids = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    train_dataset = []

    if case_id_test1 in case_ids:
        case_ids.remove(case_id_test1)
    if case_id_test2 in case_ids:
        case_ids.remove(case_id_test2)
    if case_id_val in case_ids:
        case_ids.remove(case_id_val)

    train_dataset += glob.glob(train_data + "Case" + str(case_ids) + "/new_vols/*.nii.gz")
    train_dataset.sort()

    for data in train_dataset:
        print(data)

    steps_per_epoch = 100
    device = torch.device("cuda:" + gpu_id)
    # criterions = [losses.NCC(device=device), losses.Grad3d(penalty='l2')]
    criterions = [losses.NCC(device=device), losses.DisplacementRegularizer("gradient-l2")]
    weights = [1.0, 1.0]
    vol_shape = (192, 192, 224)

    if model_name == "lcamorph":
        from lcamorph.lcamorph import LCAMorph
        model = LCAMorph(vol_shape)
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    train_gen = data_generator_lung(train_dataset)

    for epoch in range(0, max_epoches):
        start_time = time.time()
        if adj:
            adjust_learning_rate(optimizer, epoch, max_epoches, lr)
        for idx in range(steps_per_epoch):
            data, _ = next(train_gen)
            data = [torch.from_numpy(t).to(device) for t in data]
            moving = data[0]
            fixed = data[1]

            moving = moving[None, None, ...].float()
            fixed = fixed[None, None, ...].float()
            max_random_params = (0.15, 0.05, 3.1416 / 18, 0.15)
            if aug:
                moving = random_affine_augment(moving, max_random_params=max_random_params, scale_params=1,
                                               return_affine_matrix=False, )
                fixed = random_affine_augment(fixed, max_random_params=max_random_params, scale_params=1,
                                              return_affine_matrix=False, )

            model_in = torch.cat((moving, fixed), dim=1)
            warped, flow = model(model_in)
            loss = 0.0
            curr_loss = [0, 0]
            curr_loss[0] += criterions[0](warped, fixed) * weights[0]
            curr_loss[1] += criterions[1](flow, flow) * weights[1]
            loss += curr_loss[0] + curr_loss[1]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del model_in
            del warped, flow

            model_in = torch.cat((fixed, moving), dim=1)
            warped, flow = model(model_in)

            loss = 0.0
            curr_loss = [0, 0]
            curr_loss[0] += criterions[0](warped, moving) * weights[0]
            curr_loss[1] += criterions[1](flow, flow) * weights[1]
            loss += curr_loss[0] + curr_loss[1]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            end_time = time.time()
            print("{} of Epoch {}, Loss: {:.4f}, Sim: {:.4f}, Reg: {:.4f}. Cost Time: {:.2f}".format(epoch, max_epoches,
                                                                                                     loss.item(),
                                                                                                 curr_loss[0].item(),
                                                                                                 curr_loss[1].item(),
                                                                                                 end_time - start_time))

        if epoch % 1 == 0:
            torch.save(model, os.path.join("./save_model_pt/dirlab", model_name + str(epoch) + ".pt"))

    torch.save(model, os.path.join("./save_model_pt/dirlab", model_name + str(max_epoches) + ".pt"))

if __name__ == '__main__':
    train()
