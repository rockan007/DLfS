import numpy as np
from sigmoid_function import sigmoid 

X = np.array([1.0,0.5])
W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])

A1 = np.dot(X,W1)+B1
print(A1)
# 执行激活函数
Z1 = sigmoid(A1)
print(Z1)

W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2 = np.array([[0.1,0.2]])
A2 = np.dot(Z1,W2)+B2
print(A2)
# 执行激活函数
Z2 = sigmoid(A2)
print(Z2)
# 输出层的激活函数 sigma()|σ()
# 将输入原样输出的函数称为恒定函数
def identity_function(x):
    return x

W3 = np.array([[0.1,0.3],[0.2,0.4]])
B3 = np.array([0.1,0.2])

A3 = np.dot(Z2,W3)+B3
Y = identity_function(A3)