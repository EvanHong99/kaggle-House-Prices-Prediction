# -*- coding=utf-8 -*-
# @File     : mytools.py
# @Time     : 2022/7/14 14:44
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : house-prices-advanced-regression-techniques
# @Description:

import torch
from myconfigs import *

def get_device():
    device='cuda' if (torch.cuda.is_available() and USE_CUDA) else 'cpu'
    print(f"using device {device}")
    return device

def concat_path(str1,str2):
    if str1[-1]!='/':
        str1=str1+'/'

    if str2[0]=='/':
        str2=str2[1:]
    return str1+str2