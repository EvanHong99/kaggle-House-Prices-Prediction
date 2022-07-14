# -*- coding=utf-8 -*-
# @File     : mydataset.py
# @Time     : 2022/7/14 15:28
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : house-prices-advanced-regression-techniques
# @Description:


from torch.utils.data import Dataset

import pandas as pd
import numpy as np

import myconfigs

# class RNNDataLoader(Dataset):
#     """
#     gru data loader
#
#     需要先将数据存储成需要的格式，再由该类读取
#
#     """
#     import myconfigs as myconfig
#     def __init__(self, file_path,y_type:str='tag',n_steps_ahead=myconfig.N_STEPS_AHEAD,sequence_length=myconfig.SEQUENCE_LENGTH,sentiment=myconfig.USE_SENTIMENT,use_cols=myconfig.USE_COLS,norm_cols=myconfig.NORM_COLS):
#         """
#         ,open,close,high,low,volume,money,predicted,open_pct,close_pct,high_pct,low_pct
#         2016-01-05,9.85,9.97,10.55,9.7,41932750.0,422708448.0,1.2121212121212122,-0.15883859948761747,-0.07513914656771792,-0.1081994928148774,-0.1001855287569573
#
#
#         :param file_path:
#         :param y_type: list['tag','logits']
#             加载情感分类数据的类型，只完成了tag，todo logits
#         """
#         stockdata = pd.read_csv(file_path, header=0,index_col=0)
#         alldata = pd.read_csv(myconfig.GRU_ALLDATA_PATH, header=0,index_col=0)
#         stockdata=stockdata.iloc[n_steps_ahead:]#需要iloc抛掉前面pct change的nan，不能用dropna，因为valid和test前几行没有nan但也是需要去掉以保证仅在某个数据及上进行验证测试而不需要其他数据集的补充
#         # 这里不能转换成datetime，因为dataloader不允许
#         # stockdata.index=pd.to_datetime(stockdata.index)
#         shift=n_steps_ahead+sequence_length-1
#         self.y_list=stockdata[myconfig.GRU_PRED_TARGET].iloc[shift:].values.tolist()
#         self.pred_dates=stockdata.iloc[shift:].index.tolist()
#         # print(self.y_list)
#         # t=alldata[norm_cols]
#         # stockdata[norm_cols]=(stockdata[norm_cols]-t.min())/(t.max()-t.min()+1e-8)
#         t=stockdata[norm_cols]
#         stockdata[norm_cols]=(t-t.min())/(t.max()-t.min()+1e-8)
#         self.norm_cols=norm_cols
#
#         '''
#         self.x_list shape: dates*features
#         array([[9.85000000e+00, 9.97000000e+00, 1.05500000e+01, ...,
#         4.19327500e+07, 4.22708448e+08, 1.21212121e+00],
#        [1.01500000e+01, 1.02000000e+01, 1.03100000e+01, ...,
#         2.89728590e+07, 2.92845632e+08, 1.50000000e+00],
#         注意这里n steps ahead
#         '''
#
#         self.use_cols=use_cols
#         shift=n_steps_ahead
#         if myconfig.TEST_MODE:
#             t=deepcopy(stockdata[use_cols].iloc[:-shift])
#             t.loc[:,'close_change']=self.y_list
#             self.x_list =t.values.tolist()
#         else:
#             self.x_list =stockdata[use_cols].iloc[:-shift].values.tolist()
#         # print(len(self.x_list),len(self.y_list))
#         assert len(self.x_list)==len(self.y_list)+sequence_length-1
#
#         # self.x_list =[]
#         # t=stockdata[use_cols].iloc[:-shift].values
#         # for i in range(len(t)-sequence_length+1):
#         #     self.x_list.append(t[i:i+sequence_length].flatten().tolist())
#         # assert len(self.x_list)==len(self.y_list)
#
#
#         # print(stockdata.iloc[:-shift].index)
#         # self.x_list =torch.tensor(stockdata[['open_pct','close_pct','high_pct','low_pct','volume','money','predicted']].iloc[:-n_steps_ahead].values,dtype=torch.float32)
#         self.sequence_length=sequence_length
#         print(f'len(self.y_list) {len(self.y_list)}')
#
#     def __getitem__(self, index):
#         # self.dates可能是str类型
#         # return self.x_list[index:index+self.sequence_length], self.y_list[index:index+self.sequence_length],self.pred_dates[index:index+self.sequence_length]
#         return self.x_list[index:index+self.sequence_length], self.y_list[index],self.pred_dates[index]
#
#
#     def __len__(self):
#         return len(self.y_list)


class MyDataSet(Dataset):

    def __init__(self,file_path):
        self.data=pd.read_csv(file_path,index_col=0,header=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.values[index]
