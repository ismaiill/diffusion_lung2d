import os

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from img_unet2d_ms import aera1d_conv
from PIL import Image

output_directory_path = '/home/peter/data/2dmap'
output = os.listdir(output_directory_path)

data = [os.path.join(output_directory_path, file) for file in output]

# directory = '/home/ismail/diffusion_lung_2d_1chanel/datasets/lung_8_256/'
directory = '/home/shining/images' # NOTE: please change this to the image directory
if not os.path.exists(directory):
    os.mkdir(directory)

for path in data:
    target = torch.load(path)[:32, :]
    target = aera1d_conv(target, 4, stride=4)
    target = target.unsqueeze(-1)
    uni_chanel_tensor = target.permute(2, 0, 1)
    to_pil = transforms.ToPILImage()
    rgb_image = to_pil(uni_chanel_tensor)
    name = path.split('/')[-1].split('.')[0]
    rgb_image.save(os.path.join(directory, f'{name}.png'))
    
