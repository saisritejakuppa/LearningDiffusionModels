# download the standford cars dataset from torch

# import torch
# import torchvision
# dataset = torchvision.datasets.Food101('./dataset', download = True)
# print("done downloading")


import os
import shutil
os.makedirs('./curated_dataset', exist_ok=True)
for food in ['carrot_cake', 'chocolate_cake', 'cup_cakes', 'donuts', 'french_fries', 'hamburger', 'hot_dog', 'ice_cream', 'pizza', 'waffles']:
    shutil.copytree(f'./dataset/food-101/images/{food}', f'./curated_dataset/{food}')
