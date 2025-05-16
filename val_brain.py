import os
import glob
import argparse

import numpy as np
import torch

from lcamorph.lcamorph import SpatialTransformer
from dataset import load_vol
from utils import dice, jacobian_determinant

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="lcamorph")
parser.add_argument("--test_data", type=str, default="/public/home/root/registration/data/LPBA/Val/")
parser.add_argument("--gpu_id", type=str, default="1")

args = parser.parse_args()
model_name = args.model_name
test_data = args.test_data
gpu_id = args.gpu_id


def main(epoch, model_performance):
    val_vol_dataset = glob.glob(test_data + "/Vols/" + "*.nii.gz")
    val_seg_dataset = glob.glob(test_data + "/Segs/" + "*.nii.gz")
    val_vol_dataset.sort()
    val_seg_dataset.sort()

    load_model = "./save_model_pt/oasis/" + model_name + str(epoch) + ".pt"
    if not os.path.exists(load_model):
        return

    model_stn = SpatialTransformer((160, 192, 224), "nearest")

    model = torch.load(load_model)
    device = torch.device("cuda:" + gpu_id)

    model.to(device)
    model_stn.to(device)

    mean_dice_list = []

    with torch.no_grad():
        for moving_vol_name, moving_seg_name in zip(val_vol_dataset, val_seg_dataset):
            for fixed_vol_name, fixed_seg_name in zip(val_vol_dataset, val_seg_dataset):
                if moving_vol_name == fixed_vol_name:
                    continue
                moving_vol, fixed_vol = load_vol(moving_vol_name), load_vol(fixed_vol_name)
                moving_seg, fixed_seg = load_vol(moving_seg_name), load_vol(fixed_seg_name)

                good_labels = np.intersect1d(moving_seg, fixed_seg)[1:]

                moving_vol = torch.from_numpy(moving_vol).to(device)[None, None, ...].float()
                moving_seg = torch.from_numpy(moving_seg).to(device)[None, None, ...].float()
                fixed_vol = torch.from_numpy(fixed_vol).to(device)[None, None, ...].float()
                fixed_seg = torch.from_numpy(fixed_seg).to(device)[None, None, ...].float()

                model_in = torch.cat((moving_vol, fixed_vol), dim=1)
                _, flow = model(model_in)
                warp_seg = model_stn(moving_seg, flow)

                fixed_seg = fixed_seg.detach().cpu().numpy().squeeze()
                warp_seg = warp_seg.detach().cpu().numpy().squeeze()
                dice_val = dice(fixed_seg, warp_seg, good_labels)
                mean_dice_list.append(np.mean(dice_val))

        mean_dice = np.mean(mean_dice_list)
        # 仅保存在验证集最好的几个pt文件，其他的为了不占存储空间，全部删掉
        # 为了避免保存的pt文件把你的硬盘撑爆，我们建议你在python train_brain.py的时候 这个val_brain 一起执行
        with open("./save_txt/" + model_name + "_oasis.txt", "a") as f:
            f.write(str(epoch) + ": " + str(mean_dice) + "\n")
        if len(model_performance) <= 4:
            model_performance[epoch] = mean_dice
        else:
            min_key = min(model_performance, key=model_performance.get)
            if model_performance[min_key] < mean_dice:
                del model_performance[min_key]
                os.remove("./save_model_pt/oasis/" + model_name + str(min_key) + ".pt")
                model_performance[epoch] = mean_dice
            else:
                os.remove("./save_model_pt/oasis/" + model_name + str(epoch) + ".pt")

    print("Performance: ", model_performance)

    # 等待
    time.sleep(300)


if __name__ == "__main__":
    model_performance = {}
    import time

    with open("./save_txt/" + model_name + "_oasis.txt", "a") as f:
        f.write("\n" + model_name + "\n")
    for epoch in range(0, 500):
        main(epoch, model_performance)
