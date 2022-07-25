# -*- coding=utf-8 -*-
# @File     : test.py
# @Time     : 2022/7/15 0:59
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : house-prices-advanced-regression-techniques
# @Description:
import numpy as np
import pandas as pd
import torch

if __name__ == '__main__':
    print(torch.cuda.is_available())
    a=torch.ones((10,1)).to('cuda')