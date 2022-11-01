import os, cv2
import torch
import numpy as np
from utils import *

torch.backends.quantized.engine = 'qnnpack'
net = get_jit_model("edgeSR_int8_qnnpack_jit.pth")
preprocess = cv2tensor()
postprocess = tensor2cv()

with torch.no_grad():
    img = cv2.imread('sample_img.jpeg', cv2.IMREAD_COLOR)
    started = time.time()
    lr_tensor = preprocess(img)
    sr_tensor = net(lr_tensor).squeeze(0)
    sr_ndarray = postprocess(sr_tensor)
    time_elapsed = time.time() - started

print(f"# BENCHMARK (SINGLE IMG) [{time_elapsed * 1000:.3f}ms]")

# SHOW IMG
cv2.imwrite("output.jpg", sr_ndarray)
