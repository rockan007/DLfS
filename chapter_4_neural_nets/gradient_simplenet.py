
from gradient import numerical_gradient
from loss_function import cross_entropy_error
import numpy as np
import sys,os
sys.path.append(os.pardir)
from chapter_3_neural_networks.softmax_function import softmax

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 用高斯分布进行初始化

    def predict(self,x):
        return np.dot(x,self.W)
    
    def loss(self,x,t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y,t)
        return loss

net = simpleNet()
x = np.array([0.6,0.9])
p = net.predict(x)


np.argmax(p)
t = np.array([0,0,1])
net.loss(x,t)
# 求梯度
def f(W):
    return net.loss(x,t)
dW = numerical_gradient(f,net.W)
print(dW)