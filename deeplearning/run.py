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
from sklearn.decomposition import PCA

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

    def __init__(self,pretrained_model=None,use_pca=False):
        super().__init__()
        self.device = get_device()
        self.epochs = myconfigs.TRAIN_EPOCHS
        self.lr = myconfigs.LR

        self.features=0 # 特征数量
        self.load_data(use_pca=use_pca)
        self.model = MLP(in_features=self.features).to(device=self.device)
        if pretrained_model:# 用于加载checkpoint
            self.model=torch.load(pretrained_model)
        self.loss_fn = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.scheduler=ReduceLROnPlateau(self.optimizer,'min',patience=2,verbose=True)

        self.writer=SummaryWriter(log_dir="../logs/mlplogs/",comment=f'MLP')


    def load_model(self,model_path):
        self.model = torch.load(model_path)

    def normalization(self,dataset:pd.DataFrame,smax,smin):
        process_col=np.where((smax - smin) != 0)[0]
        dataset.iloc[:,process_col]=(dataset.iloc[:,process_col]-smin[process_col])/(smax[process_col]-smin[process_col])
        return dataset

    def load_data(self, train_path=myconfigs.TRAIN_PATH, valid_path=myconfigs.VALID_PATH,
                  test_path=myconfigs.TEST_PATH,use_pca=False):

        self.trainset = MyDataSet(train_path)
        self.validset = MyDataSet(valid_path)
        self.testset = MyDataSet(test_path)
        self.features=len(self.validset.data.columns.values)-1

        # normalization
        alldata=self.trainset.data.append(self.validset.data).append(self.testset.data)
        scalar_max=alldata.max().values[:-1]
        scalar_min=alldata.min().values[:-1]
        self.trainset.data.iloc[:,:-1]=self.normalization(self.trainset.data.iloc[:,:-1],scalar_max,scalar_min)
        self.validset.data.iloc[:,:-1]=self.normalization(self.validset.data.iloc[:,:-1],scalar_max,scalar_min)
        self.testset.data.iloc[:,:-1]=self.normalization(self.testset.data.iloc[:,:-1],scalar_max,scalar_min)

        if use_pca:
            print("pca processing")
            alldata = self.trainset.data.append(self.validset.data).append(self.testset.data)
            y=alldata["SalePrice"]
            pca = PCA(n_components='mle')
            tempdata=pca.fit_transform(alldata.iloc[:,:-1])
            # print(tempdata.shape)
            idx1=len(self.trainset.data)
            idx2=idx1+len(self.validset.data)
            self.trainset.data=pd.DataFrame(index=alldata.index[:idx1],columns=range(tempdata.shape[1]),data=tempdata[:idx1])
            self.validset.data =pd.DataFrame(index=alldata.index[idx1:idx2],columns=range(tempdata.shape[1]),data=tempdata[idx1:idx2])
            self.testset.data =pd.DataFrame(index=alldata.index[idx2:],columns=range(tempdata.shape[1]),data=tempdata[idx2:])
            self.trainset.data['SalePrice'] =y[:idx1]
            self.validset.data['SalePrice'] =y[idx1:idx2]
            self.testset.data['SalePrice'] =y[idx2:]

            self.features=len(self.validset.data.columns.values)-1

        self.trainloader = DataLoader(self.trainset, batch_size=myconfigs.BATCH_SIZE, shuffle=True, drop_last=False)
        self.validloader = DataLoader(self.validset, batch_size=myconfigs.BATCH_SIZE, shuffle=False, drop_last=False)
        self.testloader = DataLoader(self.testset, batch_size=myconfigs.BATCH_SIZE, shuffle=False, drop_last=False)

        # print(len(self.trainset),len(self.validset),len(self.testset))
        # print((self.trainset.data),(self.validset.data),(self.testset.data))

    def save_data(self,prefix):
        self.trainset.data.to_csv(f"../data/{prefix}_train.csv")
        self.validset.data.to_csv(f"../data/{prefix}_valid.csv")
        self.testset.data.to_csv(f"../data/{prefix}_test.csv")


    def save_model(self,path):
        torch.save(self.model,path)


    # train
    def train_one_epoch(self, epoch_idx):
        print(f"training epoch {epoch_idx+1}")
        self.model.train()
        start_time=time.time()

        batch_idx = 0
        total_loss = 0
        counter=0
        total_batchs = len(self.trainloader)
        for batch_idx, features in enumerate(self.trainloader):
            # counter+=len(features)
            y = features[:, -1].to(self.device).float()
            x = features[:, :-1].to(self.device).float()
            # print("input",x,x.shape)
            pred = self.model(x)
            # print(pred,y)
            mseloss = self.loss_fn(pred.flatten(), y)  # 注意加入log
            total_loss += mseloss.cpu().detach().numpy()
            # print("mseloss.cpu().detach().numpy(),total_loss",mseloss.cpu().detach().numpy(),total_loss)

            self.optimizer.zero_grad()
            mseloss.backward()
            self.optimizer.step()

            if batch_idx % myconfigs.LOG_PRINT_BATCHS == 0:
                print("epoch:[%d|%d] [%d|%d] avg rmse loss:%f" % (
                    epoch_idx + 1, self.epochs, batch_idx + 1, total_batchs, np.sqrt(total_loss / (batch_idx + 1))))

        print(f'finish epoch {epoch_idx+1} ' + "time:%.3f" % (time.time() - start_time))

        return np.sqrt(total_loss / (batch_idx + 1))

    def valid_one_epoch(self,epoch):
        print("-"*5,f"valid {epoch}","-"*5)
        self.model.eval()
        total_loss=0
        counter=0
        for batch_idx,data in enumerate(self.validloader):
            x=data[:,:-1].to(self.device).float()
            y=data[:,-1].to(self.device).float()
            pred=self.model(x).flatten()
            # 注意使用log
            valid_loss=self.loss_fn(torch.log(pred),torch.log(y)).cpu().detach().numpy()*len(y)
            total_loss+=valid_loss
            counter+=len(y)
        print("-"*5,f"finish valid {np.sqrt(total_loss/counter)}","-"*5)
        return np.sqrt(total_loss/counter)

    def train_epochs(self, epochs):
        print("="*10+"start training"+"="*10)
        train_loss=0
        valid_loss=0
        for epoch in range(epochs):
            train_loss=self.train_one_epoch(epoch)
            valid_loss=self.valid_one_epoch(epoch)
            if epoch%myconfigs.LOG_TENSORBOARD_EPOCHS==0:
                self.writer.add_scalar(tag='train_loss', scalar_value=train_loss, global_step=epoch)
                self.writer.add_scalar(tag='valid_loss', scalar_value=valid_loss, global_step=epoch)
            self.scheduler.step(valid_loss)
            for param_group in self.optimizer.param_groups:
                print("learning rate is ",param_group['lr'])
        self.save_model(concat_path(myconfigs.MODEL_SAVE_PATH,f"mlp_{str(time.strftime('%Y%m%d-%H%M%S'))}_{str(np.round(valid_loss,6))}.pt"))
        print("="*10+"finish training"+"="*10)

    def gen_submission(self,output_file):
        print("-"*5,f"gen_submission","-"*5)
        self.model.eval()
        res=[]
        for batch_idx,data in enumerate(self.testloader):
            # print(batch_idx,len(data))
            x=data[:,:-1].to(self.device).float()
            pred=self.model(x).flatten().cpu().detach().numpy().tolist()
            res.extend(pred)

        print("-"*5,f"finish generation len {len(res)}","-"*5)
        self.testset.data["SalePrice"]=np.array(res)
        self.testset.data["SalePrice"].to_csv(output_file)

if __name__ == '__main__':
    # runner=Runner(use_pca=True)
    runner=Runner("../trained_models/mlp/mlp_20220715-180912_0.167102.pt",use_pca=False)

    # save
    # runner.save_data("temp")

    # train
    # runner.train_epochs(myconfigs.TRAIN_EPOCHS)

    # gen submission
    runner.gen_submission("../my_submission.csv")

