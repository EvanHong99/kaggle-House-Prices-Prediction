# -*- coding=utf-8 -*-
# @File     : run.py
# @Time     : 2022/7/14 15:16
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : house-prices-advanced-regression-techniques
# @Description:

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import myconfigs
from mytools import *
from mydataset import MyDataSet
from mymodels import MLP

from abc import abstractmethod
import pandas as pd
import numpy as np
import time


class BaseRunner(object):

    def __init__(self):
        self.device = None
        self.epochs = None
        self.model = None
        self.loss_fn = None
        self.optimizer = None

    @abstractmethod
    def load_data(self):
        raise NotImplementedError

    @abstractmethod
    def train_one_epoch(self):
        raise NotImplementedError

    @abstractmethod
    def train_epochs(self):
        raise NotImplementedError

    @abstractmethod
    def save_model(self):
        raise  NotImplementedError


class Runner(BaseRunner):

    def __init__(self):
        super().__init__()
        self.device = get_device()
        self.epochs = myconfigs.TRAIN_EPOCHS
        self.lr = myconfigs.LR

        self.model = MLP().to(device=self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

        self.writer=SummaryWriter(log_dir="../logs/mlplogs/",comment=f'MLP')

    def load_data(self, train_path=myconfigs.TRAIN_PATH, valid_path=myconfigs.VALID_PATH,
                  test_path=myconfigs.VALID_PATH):

        def normalization(dataset:pd.DataFrame,smax,smin):
            return (dataset.iloc[:,:-1]-smin)/(smax-smin)

        trainset = MyDataSet(train_path)
        validset = MyDataSet(valid_path)
        testset = MyDataSet(test_path)

        # normalization
        alldata=trainset.data.append(validset.data).append(testset.data)
        scalar_max=alldata.max().values[:-1]
        scalar_min=alldata.min().values[:-1]
        trainset.data.iloc[:,:-1]=normalization(trainset.data,scalar_max,scalar_min)
        validset.data.iloc[:,:-1]=normalization(validset.data,scalar_max,scalar_min)
        testset.data.iloc[:,:-1]=normalization(testset.data,scalar_max,scalar_min)
        self.trainloader = DataLoader(trainset, batch_size=myconfigs.BATCH_SIZE, shuffle=True, drop_last=False)
        self.validloader = DataLoader(validset, batch_size=myconfigs.BATCH_SIZE, shuffle=False, drop_last=False)
        self.testloader = DataLoader(testset, batch_size=myconfigs.BATCH_SIZE, shuffle=False, drop_last=False)

    # train
    def train_one_epoch(self, epoch_idx):
        print(f"training epoch {epoch_idx+1}")
        self.model.train()
        start_time=time.time()

        counter = 0
        total_loss = 0
        total_batchs = len(self.trainloader)
        for batch_idx, features in enumerate(self.trainloader):
            # counter+=len(features)
            y = features[:, -1].to(self.device).float()
            x = features[:, :-1].to(self.device).float()

            pred = self.model(x)
            mseloss = self.loss_fn(torch.log(pred), torch.log(y))  # 注意加入log
            total_loss += mseloss.cpu().detach().numpy()

            self.optimizer.zero_grad()
            mseloss.backward()
            self.optimizer.step()


            if batch_idx % myconfigs.LOG_PRINT_BATCHS == 0:
                print("epoch:[%d|%d] [%d|%d] avg rmse loss:%f" % (
                    epoch_idx + 1, self.epochs, batch_idx + 1, total_batchs, np.sqrt(total_loss / (batch_idx + 1))))

        print(f'finish epoch {epoch_idx+1} ' + "time:%.3f" % (time.time() - start_time))

        return np.sqrt(total_loss / (batch_idx + 1))

    def train_epochs(self, epochs):
        print("="*10+"start training"+"="*10)
        loss=None
        for epoch in range(epochs):
            loss=self.train_one_epoch(epoch)
            if epoch%myconfigs.LOG_TENSORBOARD_EPOCHS==0:
                self.writer.add_scalar(tag='train_loss', scalar_value=loss, global_step=epoch)

        self.save_model(concat_path(myconfigs.MODEL_SAVE_PATH,f"mlp_{str(time.strftime('%Y%m%d-%H%M%S'))}_{str(np.round(loss,6))}.pt"))


        print("="*10+"finish training"+"="*10)


    def save_model(self,path):
        torch.save(self.model,path)


if __name__ == '__main__':
    runner=Runner()
    runner.load_data()
    runner.train_epochs(myconfigs.TRAIN_EPOCHS)