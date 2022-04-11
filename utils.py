import math
import torch

def preprocess_gradients(x):
    """附录中的梯度预处理
    :param x:输入的梯度
    :return:2D tensor，即文中的坐标对"""
    p = 10
    eps = 1e-6
    indicator = (x.abs() > math.exp(-p)).float()
    x1 = (x.abs() + eps).log() / p * indicator - (1 - indicator)
    x2 = x.sign() * indicator + math.exp(p) * x * (1 - indicator)

    return torch.cat((x1, x2), 1)
