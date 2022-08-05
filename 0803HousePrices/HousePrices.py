#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : HousePrices.py
# @Author: LauTrueYes
# @Date  : 2022/8/3 16:32
import os
import random
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.optimizer import SGD

class Regressor(nn.Layer):
    def __init__(self):
        super(Regressor, self).__init__()
        self.fc = nn.Linear(in_features=13, out_features=1)

    def forward(self, inputs):
        x = self.fc(inputs)
        return x

def load_data(datafile):
    data = np.fromfile(datafile, sep=' ', dtype=np.float32)

    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    feature_num = len(feature_names)

    data = data.reshape([data.shape[0]//feature_num, feature_num])  # 将原始数据进行Reshape，变成[N, 14]

    ratio = 0.8
    offset = int(data.shape[0]*ratio)
    training_data = data[:offset]    # 将原数据集拆分成训练集和测试集,这里使用80%的数据做训练，20%的数据做测试， 测试集和训练集必须是没有交集的

    maximums = training_data.max(axis=0)
    minimums = training_data.min(axis=0)
    avgs = training_data.sum(axis=0) / training_data.shape[0]

    # 记录数据的归一化参数，在预测时对数据做归一化

    for i in range(feature_num):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data, maximums, minimums, avgs

def load_one_example():
    idx = np.random.randint(0, test_data.shape[0])
    idx = -10
    one_data, label = test_data[idx, :-1], test_data[idx, -1]
    one_data = one_data.reshape([1, -1])
    return one_data, label


if __name__ == '__main__':
    datafile = './work/housing.data'
    train_data, test_data, maximums, minimums, avgs = load_data(datafile)

    print(train_data.shape)
    print(train_data[1,:])

    model = Regressor()
    model.train()

    optimizer = SGD(learning_rate=0.01, parameters=model.parameters())  # 定义优化算法，使用随机梯度下降SGD 学习率设置为0.01

    epoch_num = 10
    batch_size = 10

    for epoch_id in range(epoch_num):
        np.random.shuffle(train_data)
        mini_batches = [train_data[k: k+batch_size] for k in range(0, len(train_data), batch_size)]
        for iter_id, mini_batch in enumerate(mini_batches):
            x = np.array(mini_batch[:, :-1])
            y = np.array(mini_batch[:, -1:])

            house_features = paddle.to_tensor(x)
            prices = paddle.to_tensor(y)

            predicts = model(house_features)

            loss = F.square_error_cost(predicts, label=prices)
            avg_loss = paddle.mean(loss)

            if iter_id % 20 == 0:
                print("epoch: {}, iter: {}, loss: {}".format(epoch_id, iter_id, avg_loss.numpy()))

            avg_loss.backward()
            optimizer.step()
            optimizer.clear_grad()
    paddle.save(model.state_dict(), './LR_model.pdparams')
    print('模型保存成功，模型参数保存在LR_model.pdparams中')

    model_dict = paddle.load('./LR_model.pdparams')
    model.load_dict(model_dict)
    model.eval()

    one_data, label = load_one_example()
    one_data = paddle.to_tensor(one_data)
    predict = model(one_data)

    predict = predict * (maximums[-1] - minimums[-1]) + avgs[-1]
    label = label * (maximums[-1] - minimums[-1]) + avgs[-1]
    print("Inference result is {}, the corresponding label is {}".format(predict.numpy(), label))
