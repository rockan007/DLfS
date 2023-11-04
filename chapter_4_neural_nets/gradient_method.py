'''
梯度标识的是各点处的函数值减小最多的方向。
无法保证梯度所指的方向就是函数的最小值或者真正应该前进的方向    
在复杂的函数中，梯度只是的方向基本上不是函数值的最小处
'''
'''
函数的极小值、最小值以及被称为鞍点(saddle point)的地方，梯度为零

极小值是局部的最小值
鞍点是从某个方向看是极大值，从另一个方向上看则是极小值的点
'''
'''
虽然梯度的方向不一定指向最小值，但沿着它的方向能最大限度的减少(损失)函数的值
'''
# gradient decent method 
from gradient import numerical_gradient,function_2
import numpy as np

def gradient_descent(f,init_x,lr=0.01,step_num=100):
    x =init_x
    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr*grad

    return x

init_x = np.array([-3.0,4.0])
print(gradient_descent(function_2, init_x=init_x,lr=0.1,step_num=100)) 
init_x = np.array([-3.0,4.0])
print(gradient_descent(function_2, init_x=init_x,lr=10.0,step_num=100)) 
init_x = np.array([-3.0,4.0])
print(gradient_descent(function_2, init_x=init_x,lr=1e-10,step_num=100)) 