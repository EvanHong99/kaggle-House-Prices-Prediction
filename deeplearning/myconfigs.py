# -*- coding=utf-8 -*-
# @File     : myconfigs.py
# @Time     : 2022/7/14 14:45
# @Author   : EvanHong
# @Email    : 939778128@qq.com
# @Project  : house-prices-advanced-regression-techniques
# @Description:

# data
TRAIN_PATH=u"../data/preprocessed_dummy_train.csv"
VALID_PATH=u"../data/preprocessed_dummy_valid.csv"
TEST_PATH=u"../data/preprocessed_dummy_test.csv"

# train
USE_CUDA=True
TRAIN_EPOCHS=10
BATCH_SIZE=32
LR=1e-1

MODEL_SAVE_PATH="../trained_models/mlp/"

# log
LOG_TENSORBOARD_EPOCHS=1
LOG_PRINT_BATCHS=5 #5 batch print
