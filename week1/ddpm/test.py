import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from model import UNet
import wandb
import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
from logzero import logger

from train import Diffusion

if __name__ == "__main__":
    device = "cuda"
    model = UNet().to(device)
    ckpt = torch.load("./models/DDPM_Uncondtional/ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = Diffusion(img_size=64, device=device)
    x = diffusion.sample(model, 64)
    print(x.shape)
    plt.figure(figsize=(32, 32))
    
    plt.imshow(torch.cat([
        torch.cat([i for i in x.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()
    plt.savefig("./sample.png")
