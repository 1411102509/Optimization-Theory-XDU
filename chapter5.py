import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#函数
def f(X):
    x1 = X[0][0]    
    x2 = X[1][0]
    y = x1**2+x2**2-x1*x2-10*x1-4*x2+60
    return y
A = np.mat([[2,-2],[0,2]])  #需要手动给出

#求梯度矩阵
def g(X):
    x1_value = np.array(X)[0][0]
    x2_value = np.array(X)[1][0]

    x1 = sp.Symbol('x1')
    x2 = sp.Symbol('x2')
    
    #求偏导
    f_x1 = sp.diff(f([[x1],[x2]]),x1)
    f_x2 = sp.diff(f([[x1],[x2]]),x2)

    return np.mat([f_x1.evalf(subs={x1:x1_value,x2:x2_value}),
                   f_x2.evalf(subs={x1:x1_value,x2:x2_value})]).T

#求Hesse矩阵
def G(X):
    x1_value = np.array(X)[0][0]
    x2_value = np.array(X)[1][0]

    x1 = sp.Symbol('x1')
    x2 = sp.Symbol('x2')
    
    #求偏导
    f_x1 = sp.diff(f([[x1],[x2]]),x1)
    f_x2 = sp.diff(f([[x1],[x2]]),x2)

    f_x1_x1 = sp.diff(f_x1,x1)
    f_x1_x2 = sp.diff(f_x1,x2)
    f_x2_x1 = sp.diff(f_x2,x1)
    f_x2_x2 = sp.diff(f_x2,x2)

    return np.mat([[float(f_x1_x1.evalf(subs={x1:x1_value,x2:x2_value})),
                   float(f_x1_x2.evalf(subs={x1:x1_value,x2:x2_value}))],
                   [float(f_x2_x1.evalf(subs={x1:x1_value,x2:x2_value})),
                   float(f_x2_x2.evalf(subs={x1:x1_value,x2:x2_value}))]])

#牛顿法
def Newton(X_0,stop_value = 0.01):
    Iteration_times = 0
    X_0 = np.mat(X_0)

    f_0 = f(X_0)
    g_0 = g(X_0)

    while 1:
        Iteration_times += 1
        print("********************* ", Iteration_times , "************************")
        #计算K+1个点的位置
        X_1 = X_0 + G(X_0).I * (-1*g_0)   #t=1
        f_1 = f(X_1)
        g_1 = g(X_1)

        print("X_0 = \n",np.array(X_0),'\nf_0 = \n',f_0,"\n")
        print("X_1 = \n",np.array(X_1),'\nf_1 = \n',f_1,"\n")
        #终止条件
        if abs(f_1 - f_0) < stop_value:
            break

        #继续迭代，更新点位置
        X_0 = X_1
        f_0 = f_1
        g_0 = g_1

    return X_1,f_1,Iteration_times-1

#最速下降法
def Steepest_Descent(X_0,stop_value = 0.01):
    Iteration_times = 0
    X_0 = np.mat(X_0)

    f_0 = f(X_0)
    g_0 = g(X_0)

    while 1:
        Iteration_times += 1
        print("********************* ", Iteration_times , "************************")
        #计算K+1个点的位置
        X_1 = X_0 - float( (g_0.T*g_0) / (g_0.T*A*g_0) )*g_0
        f_1 = f(X_1)
        g_1 = g(X_1)

        print("X_0 = \n",np.array(X_0),'\nf_0 = \n',f_0,'\ng_0 = \n',g_0,"\n")
        print("X_1 = \n",np.array(X_1),'\nf_1 = \n',f_1,'\ng_1 = \n',g_1,"\n")
        #终止条件
        if abs(f_1 - f_0) < stop_value:
            break
        
        #继续迭代，更新点位置
        X_0 = X_1
        f_0 = f_1
        g_0 = g_1

    return X_1,f_1,Iteration_times-1


if __name__ == "__main__":
    plt.figure()  #定义新的三维坐标轴
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    #定义三维数据
    x1 = np.arange(-10,20,0.5)
    x2 = np.arange(-10,20,0.5)
    X1, X2 = np.meshgrid(x1, x2)
    Y = f([[X1],[X2]])
    #作图
    ax.plot_surface(X1,X2,Y,rstride = 1, cstride = 1,cmap='rainbow')
    ax.contour(X1, X2, Y, offset=0, cmap='rainbow')   #等高线图，要设置offset，为Z的最小值

    
    print("******************************* 最速下降法 ****************************")
    X_out,f_out,times = Steepest_Descent([[0],[0]],stop_value = 0.01)
    print("起始点：[0,0]，经过",times,"次迭代，找出最优解：\n当X = (",float(X_out[0][0]),","
          ,float(X_out[1][0]),") 时，minf(X) = ",float(f_out))
    
    ax.scatter(float(X_out[0][0]),float(X_out[1][0]),float(f_out),c = 'r')
    plt.show()

    print("******************************* 牛顿法 ****************************")
    X_out,f_out,times = Newton([[0],[0]],stop_value = 0.01)
    print("起始点：[0,0]，经过",times,"次迭代，找出最优解：\n当X = (",float(X_out[0][0]),","
          ,float(X_out[1][0]),") 时，minf(X) = ",float(f_out))