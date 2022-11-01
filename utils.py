import os
import time
import cv2
import numpy as np
import torch
from torchvision.transforms import *
def get_jit_model(weight_path:str):
    try:
        net = torch.jit.load(weight_path)
    except:
        raise RuntimeError(f"weight_path({weight_path}) is wrong")
    return net

class cv2tensor(object):
    def __init__(self, size=(240, 426)) -> None:
        #  Change code to cv2 compatible
        self.size = size
        self.min_hw = min(size[0], size[1])
        self.pp = Compose([
            ToTensor()
        ])
    def __call__(self, x:np.ndarray):
        h, w, c = x.shape
        if self.min_hw > min(h, w):
            raise ValueError(f"Size is too small..")
        crop_width, crop_height = self.size[1], self.size[0]
        mid_x, mid_y = w//2, h//2
        offset_x, offset_y = crop_width//2, crop_height//2
        crop_img = x[mid_y - offset_y:mid_y + offset_y, mid_x - offset_x:mid_x + offset_x]
        return self.pp(crop_img).unsqueeze(0)
        

class tensor2cv(object):
    def __call__(self, x:torch.Tensor):
        x = x.detach().cpu().numpy()
        x = np.transpose(x, (1, 2, 0))
        x = x * 255.0
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = x.astype(np.uint8).copy()
        return x
