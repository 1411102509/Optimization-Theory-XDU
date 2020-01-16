import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def f(x):
    return 3*x**4-4*x**3-12*x**2

# def f_diff_1(x):
#     return 12*x**3-12*x**2-24*x

# def f_diff_2(x):
#     return 36*x**2-24*x-24

def  f_diff_1(x_value):
    x = sp.Symbol('x')
    fx = sp.diff(f(x),x)
    return fx.evalf(subs = {x:x_value})

def  f_diff_2(x_value):
    x = sp.Symbol('x')
    fx = sp.diff(f(x),x)
    fx = sp.diff(fx,x)
    return fx.evalf(subs = {x:x_value})

# 黄金分割法
def Golden_Section(a,b,stopValue = 0.01):
    t_1 = a
    t_2 = b
    beta = 1-(np.sqrt(5)-1)/2   #t1取在0.618位置，#t2取在0.382处

    while abs(t_1-t_2) >= stopValue:
        t_2 = a + beta*(b-a)
        value_2 = f(t_2)

        t_1 = a + b - t_2
        value_1 = f(t_1)
        # print("t1 : f(%0.3f) = %0.6f" % (t_1,value_1))
        # print("t2 : f(%0.3f) = %0.6f" % (t_2,value_2))

        if value_1 < value_2:
            a = t_2
            # print("丢弃a -- t2   [a,b]:[%0.3f,%0.3f]\n" % (a,b))
        else:
            b = t_1
            # print("丢弃t1 -- b   [a,b]:[%0.3f,%0.3f]\n" % (a,b))

    t_out = (t_1+t_2)/2
    f_out = f(t_out)
    return t_out,f_out 

# 牛顿切线法
def Newton_Tangent(a,b,t_0,stopValue = 0.0001):    #要求a、b两点处的一阶导数值相反
    t = t_0+1   #随便取值

    while abs(t-t_0) >= stopValue:
        # print("t_0:",t_0)
        t_0 = t
        t = t_0 - f_diff_1(t_0)/f_diff_2(t_0)
    # print("t_0:",t_0)
    t_out = t_0
    f_out = f(t_0)
    return t_out,f_out 

# 二次插值法
def Quadratic_Interpolation(t_0,t_1,t_2,stopValue = 0.0001):   #要求f(t_1)>f(t_0),f(t_2)>f(t_0)
    t_match = 0.5 \
        * ( (pow(t_0,2)-pow(t_2,2))*f(t_1) + (pow(t_2,2)-pow(t_1,2))*f(t_0) + (pow(t_1,2)-pow(t_0,2))*f(t_2) ) \
        / ( (t_0-t_2)*f(t_1) + (t_2-t_1)*f(t_0) + (t_1-t_0)*f(t_2) )

    while abs(t_match-t_0) >= stopValue:
        print("t_match:",t_match)
        if t_match > t_0:
            if f(t_match) <= f(t_0):
                t_2 = t_0
                t_0 = t_match
            else:
                t_1 = t_match
        else:
            if f(t_match) <= f(t_0):
                t_1 = t_0
                t_0 = t_match
            else:
                t_2 = t_match

        t_match = 0.5 \
            * ( (pow(t_0,2)-pow(t_2,2))*f(t_1) + (pow(t_2,2)-pow(t_1,2))*f(t_0) + (pow(t_1,2)-pow(t_0,2))*f(t_2) ) \
            / ( (t_0-t_2)*f(t_1) + (t_2-t_1)*f(t_0) + (t_1-t_0)*f(t_2) )
    
    t_out = t_match
    f_out = f(t_match)
    return t_out,f_out
                


if __name__ == "__main__":
    x = np.linspace(-5,5)
    y1 = f(x)
    plt.plot(x,y1)
    # plt.plot(x,y2)
    plt.show()

    print("******************************* 黄金分割法 ****************************")
    t_out,f_out = Golden_Section(-2,0)
    print("起始区间：[-2,0]， minf(x) = f(%f) = %f" % (t_out,f_out),'\n')
    t_out,f_out = Golden_Section(0,3)
    print("起始区间：[0，3]， minf(x) = f(%f) = %f\n" % (t_out,f_out))

    print("******************************* 牛顿切线法 ****************************")
    t_out,f_out = Newton_Tangent(-5,5,-1.2)
    print("起始点：-1.2， minf(x) = f(%f) = %f" % (t_out,f_out),'\n')
    t_out,f_out = Newton_Tangent(-5,5,2.5)
    print("起始点：2.5， minf(x) = f(%f) = %f\n" % (t_out,f_out))

    print("******************************* 二次插值法 ****************************")
    t_out,f_out = Quadratic_Interpolation(-1.2,-1.1,-0.8)
    print("起始点：[-1.2,-1.1,-0.8]， minf(x) = f(%f) = %f" % (t_out,f_out),"\n")
    t_out,f_out = Quadratic_Interpolation(1.5,1.7,2.5)
    print("起始点：[1.5, 1.7, 2.5]， minf(x) = f(%f) = %f\n" % (t_out,f_out))