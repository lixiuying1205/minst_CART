# -*- coding: utf-8 -*-
# 使用LR进行MNIST手写数字分类
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import time

# 加载数据
time_1 = time.time()
digits = load_digits()
data = digits.data
# 数据探索
# print(data.shape)
# 查看第一幅图像
# print(digits.images[0])
# 第一幅图像代表的数字含义
# print(digits.target[0])
# 将第一幅图像显示出来
# plt.gray()
# plt.imshow(digits.images[0])
# plt.show()
time_2 = time.time()
print("数据加载时间为：",time_2-time_1)
# 分割数据，将30%的数据作为测试集，其余作为训练集
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.3, random_state=33)

# 采用Z-Score规范化
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)
time_3 = time.time()
print("数据预处理时间为：",time_3-time_2)
# 创建CART分类器
clf = DecisionTreeClassifier(criterion='gini')
clf.fit(train_ss_x, train_y)
predict_y = clf.predict(test_ss_x)
time_4 = time.time()
print("模型训练预测时间为：",time_4-time_3)
print('CART准确率: %0.4lf' % accuracy_score(predict_y, test_y))
