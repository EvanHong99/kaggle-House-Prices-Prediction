# -*- coding=utf-8 -*-
# @File     : mymodels.py
# @Time     : 2022/7/14 14:26
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : house-prices-advanced-regression-techniques
# @Description:

import torch
from torch import nn

class MLP(nn.Module):
    """
      Multilayer Perceptron for regression.
    """

    def __init__(self, in_features=319):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(in_features=in_features,out_features=64),
            nn.LeakyReLU(),
            nn.Linear(64,32),
            nn.LeakyReLU(),
            nn.Linear(32,1)
        )
        # self.layers=nn.Sequential(
        #     nn.Linear(in_features=in_features,out_features=64),
        #     nn.LeakyReLU(),
        #     nn.Linear(64,1)
        # )

    def forward(self,x):
        return self.layers(x)