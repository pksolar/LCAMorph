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

from dataset import load_vol
from utils import point_spatial_transformer

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--case_id_val", type=str, default="1")
parser.add_argument("--model_name", type=str, default="lcamorph")
parser.add_argument("--test_data", type=str, default="/public/home/root/registration/data/DIRLAB/Case")

args = parser.parse_args()
gpu_id = args.gpu_id
case_id_val = args.case_id_val
model_name = args.model_name
test_data = args.test_data


def validation(case_id, epoch):
    # alpha_list = [0.97, 0.97, 1.16, 1.15, 1.13, 1.10, 0.97, 0.97, 0.97, 0.97]
    # alpha = alpha_list[int(case_id)]
    case_id = str(case_id)
    # beta = 0.625
    # alpha /= beta
    alpha = 1.0
    
    mov_name = test_data + case_id + "/new_vols/case" + case_id + "_T00.nii.gz"
    fix_name = test_data + case_id + "/new_vols/case" + case_id + "_T50.nii.gz"
    mov_p_name = test_data + case_id + "/new_landmarks/case" + case_id + "_T00.txt"
    fix_p_name = test_data + case_id + "/new_landmarks/case" + case_id + "_T50.txt"

    load_model = "./save_model_pt/dirlab/" + model_name + str(epoch) + ".pt"
    if not os.path.exists(load_model):
        return 100
    model = torch.load(load_model)
    device = torch.device("cuda:" + gpu_id)

    model.to(device)
    #model.eval()
    with torch.no_grad():

        mov_img = nib.load(mov_name).get_fdata()
        fix_img = nib.load(fix_name).get_fdata()
        moving_image_landmarks = np.loadtxt(mov_p_name)
        fixed_image_landmarks = np.loadtxt(fix_p_name)
        init_tre_list = []
        before_distance = fixed_image_landmarks - moving_image_landmarks
        for i in range(300):
            init_tre_list += [np.linalg.norm(before_distance[i])]
        # print(np.mean(init_tre_list) * alpha, np.std(init_tre_list) * alpha)

        mov_img = torch.from_numpy(mov_img).to(device)
        fix_img = torch.from_numpy(fix_img).to(device)

        mov_img = mov_img[None, None, ...].float()
        fix_img = fix_img[None, None, ...].float()

        model_in = torch.cat((mov_img, fix_img), dim=1)
        warped, flow = model(model_in)

        fixed_image_landmarks = fixed_image_landmarks[None, ...]
        moving_image_landmarks = moving_image_landmarks[None, ...]
        flow = flow.squeeze().permute(1, 2, 3, 0)
        dvf_Data = flow.cuda().data.cpu().numpy().squeeze()
        # 这并没有写错，对图像来说，移动图像配准到固定图像。但对标记点来说，固定图像的标记点对齐到移动图像的标记点。这虽然有点反直觉。
        data = [torch.from_numpy(t).cuda() for t in [fixed_image_landmarks, dvf_Data]]
        moved_image_landmarks = point_spatial_transformer(data)

        moving_image_landmarks = torch.from_numpy(moving_image_landmarks).cuda()
        after_distance = moved_image_landmarks - moving_image_landmarks
        after_distance = after_distance.cuda().data.cpu().numpy().squeeze()

        reg_tre_list = []
        for i in range(300):
            reg_tre_list += [np.linalg.norm(after_distance[i])]

        tre_value = np.mean(reg_tre_list) * alpha

        #######################################
        #######################################
        #######################################
        mov_img = nib.load(fix_name).get_fdata()
        fix_img = nib.load(mov_name).get_fdata()
        moving_image_landmarks = np.loadtxt(fix_p_name)
        fixed_image_landmarks = np.loadtxt(mov_p_name)
        init_tre_list = []
        before_distance = fixed_image_landmarks - moving_image_landmarks
        for i in range(len(before_distance)):
            init_tre_list += [np.linalg.norm(before_distance[i])]
        # print(np.mean(init_tre_list) * alpha, np.std(init_tre_list) * alpha)

        mov_img = torch.from_numpy(mov_img).to(device)
        fix_img = torch.from_numpy(fix_img).to(device)

        mov_img = mov_img[None, None, ...].float()
        fix_img = fix_img[None, None, ...].float()
        model_in = torch.cat((mov_img, fix_img), dim=1)
        warped, flow = model(model_in)

        fixed_image_landmarks = fixed_image_landmarks[np.newaxis, ...]
        moving_image_landmarks = moving_image_landmarks[np.newaxis, ...]
        flow = flow.squeeze().permute(1, 2, 3, 0)
        dvf_Data = flow.cuda().data.cpu().numpy().squeeze()
        data = [torch.from_numpy(t).cuda() for t in [fixed_image_landmarks, dvf_Data]]
        moved_image_landmarks = point_spatial_transformer(data)

        moving_image_landmarks = torch.from_numpy(moving_image_landmarks).cuda()
        after_distance = moved_image_landmarks - moving_image_landmarks
        after_distance = after_distance.cuda().data.cpu().numpy().squeeze()

        reg_tre_list = []
        for i in range(300):
            reg_tre_list += [np.linalg.norm(after_distance[i])]
        tre_value += np.mean(reg_tre_list) * alpha

        return tre_value / 2


def main(epoch, model_performance):
    if not os.path.exists("./save_model_pt/dirlab/" + model_name + str(epoch) + ".pt"):
        return
    tre_value = 0
    tre_value += validation(case_id_val, epoch)

    # 仅保存在验证集最好的几个pt文件，其他的为了不占存储空间，全部删掉
    # 为了避免保存的pt文件把你的硬盘撑爆，我们建议你在python train_lung.py的时候 这个val_lung 一起执行
    with open("./save_txt/" + model_name + "_dirlab.txt", "a") as f:
        # f.write(str(epoch) + " " + str(tre_value) + "\n")
        if len(model_performance) <= 2:
            model_performance[epoch] = tre_value
        else:
            min_key = max(model_performance, key=model_performance.get)
            if model_performance[min_key] > tre_value:
                del model_performance[min_key]
                os.remove("./save_model_pt/dirlab/" + model_name + str(min_key) + ".pt")
                model_performance[epoch] = tre_value
            else:
                pass
                print(epoch, ": ", tre_value)
                os.remove("./save_model_pt/dirlab/" + model_name + str(epoch) + ".pt")

    # print("验证的tre性能", tre_value)
    print(model_performance)

    # 等待
    time.sleep(260)


if __name__ == "__main__":
    model_performance = {}
    with open("./save_txt/" + model_name + "_dirlab.txt", "a") as f:
        # f.write("\n" + model_name + "\n")
        # f.write("val " + case_id_val + " " + "\n")
        pass
    for epoch in range(100):
        main(epoch, model_performance)
