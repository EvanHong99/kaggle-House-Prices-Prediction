# kaggle-House-Prices-Prediction

## 数据预处理

1. 先使用map将文本列的文本转化为数字，从而可以调用knn的包
2. 利用knn填补数值列缺失值
3. 利用`pandas.get_dummies()`构建文本列onehot表示，并删除原文本列
4. 从train中分割出valid集（0.1）用于验证模型表现


## DeepLearning

- 简单三层Linear网络
- adam

### todo

- optuna
