import numpy as np

y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
t = [0,0,1,0,0,0,0,0,0,0]

# 均方误差
def mean_squared_error(y,t):
    return 0.5*np.sum((y-t)**2)

mean_squared_error(np.array(y), np.array(t))

# 交叉熵误差
def cross_entropy_error(y,t):
    # 添加微小值 防止y为0时，np.log(0)为-inf
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))

cross_entropy_error(np.array(y),np.array(t))