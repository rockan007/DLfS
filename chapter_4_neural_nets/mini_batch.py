import numpy as np

# one-hot交叉熵误差
def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+1e-7))/batch_size

def cross_entropy_error_no_one_hot(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    
    batch_size = y.shape[0]
    # 获取神经网络在正确解上处的输出,(mnist数据)
    yr = y[np.arange(batch_size),t]
    return -np.sum(np.log(yr+1e-7))/batch_size
'''
不能将识别精度作为指标，
因为如果以识别精度为指标，则参数的导数在绝大多数地方都会变为0

识别精度对微小的参数变化基本没什么反应,即便有反应，也是不连续的、突然的变化

如果使用阶跃函数作为激活函数，神经网络的学习将无法进行

如果使用了阶跃函数,即便引入损失函数作为指标，
参数的微小变化也会被阶跃函数抹杀，
导致随时函数的值不会发生任何变化
'''
