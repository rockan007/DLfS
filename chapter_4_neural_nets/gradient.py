# 梯度
'''
全部变量的偏导数汇总而成的向量成为梯度(gradient)
'''
import numpy as np

def function_2(x):
    return np.sum(x**2)

def numerical_gradient(f,x):
    h = 1e-4
    # 生成和x形状相同的数组
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)的计算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] =  (fxh1 - fxh2)/(2*h)
        x[idx] = tmp_val

    return grad

print(numerical_gradient(function_2, np.array([3.0,4.0])))
print(numerical_gradient(function_2, np.array([0.0,2.0])))
print(numerical_gradient(function_2, np.array([3.0,0.0])))

'''
梯度指示的方向时各点处的函数值减小最多的方向
'''