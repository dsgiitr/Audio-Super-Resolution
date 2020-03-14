import numpy as np
import torch
import torch.nn as nn


def SubPixel1d(tensor, r): #(b,r,w)
    ps = nn.PixelShuffle(r)
    tensor = torch.unsqueeze(tensor, -1) #(b,r,w,1)
    tensor = ps(tensor)
    #print(tensor.shape) #(b,1,w*r,r)
    tensor = torch.mean(tensor, -1)
    #print(tensor.shape) #(b,1,w*r)
    return tensor