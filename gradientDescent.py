#!/usr/bin/env python3

from matplotlib.pyplot import *
import numpy as np
from sympy import *
import random

ERR = 0.01
alpha = 0.2
beta = 0.7

#x,y = symbols('x y', real = True)
#fx = x**2 + 2*y**2

# https://github.com/YEN-GitHub/NumericalOptimization_BasicAlgorithm/blob/master/LinearSearchMethods/SteepestDescent/GradientDescentMethod.py

# Data will be a list of tuples in the form of (xi, yi, wi)
def fxy(x,y,data):
    s = 0
    for xi,yi,wi in data: 
        s += wi * ( sqrt( (xi - x)**2  + (yi - y)**2 ) )
    return s

def f_grad_x(x,y,data):
    s = 0
    for xi,yi,wi in data: 
        s += (-wi *(xi - x )) / ( sqrt( (xi - x)**2 + (yi - y)**2 ) )
    return s

def f_grad_y(x,y,data):
    s = 0
    for xi, yi, wi in data:
        s += (-wi *(yi - y )) / ( sqrt( (xi - x)**2 + (yi - y)**2 ) ) 
    return s


# search step size
# x0,y0: start point
def BacktrackingLineSearch(x0,y0, data):
    t = 1
    x = x0
    y = y0
    beta = 0.6
    alpha = 0.3
    c = 1e-4
    backtrackingIteration = 1
    while fxy( x + t * (-f_grad_x(x,y,data)) , y + t * (-f_grad_y(x,y,data)) , data ) >\
         (fxy(x,y,data)) + (alpha * t  * ( -f_grad_x(x,y,data)*f_grad_x(x,y,data) + -f_grad_y(x,y,data) * f_grad_y(x,y,data))):
        #print("new t: {}".format(t))
        print("[{}] Backtracking Iteration, new t: {}".format(backtrackingIteration, t), end = "\r" )
        backtrackingIteration += 1
        t *= beta
    return t

def initialPoint(data):
    xdenom = 0
    xnumer = 0
    ydenom = 0
    ynumer = 0
    for x,y,w in data:
        xnumer += (w*x)
        xdenom += (w)
        ynumer += (w*y)
        ydenom += (w)
    x = xnumer / xdenom
    y = ynumer / ydenom
    return (x,y)



# Gradient Descent method
def GradientDescent(n):
    x0 = 0
    error = 10
    #curve_y = [f(x0)]
    #curve_x = [x0]
    initialData = []

    # Generate data
    for k in range(n):
        print("[{}/{}] Generating list...".format(str(k+1), str(n)) ,end="\r")
        #time.sleep(0.001)
        xn = int(random.uniform(0,n))
        yn = int(random.uniform(0,n))
        weight = random.uniform(n/10,(n/10)*5)
        initialData.append((xn,yn,weight))
    data = initialData
    p0 = initialPoint(initialData)
    #p0 = (10,10)
    print()
    print("The initial point is: {}".format(p0))
    x0 = p0[0]
    y0 = p0[1]
    startTime = time.time()
    iteram = 0
    while error > 0.004:

        stepSize = BacktrackingLineSearch(x0,y0,data) # Armijo condition
        #stepSize = -0.7
        anterior = fxy(x0, y0, data)
        y0 = y0 + stepSize * (-f_grad_y(x0,y0,data))
        x0 = x0 + stepSize * (-f_grad_x(x0,y0,data))

        actual = fxy(x0, y0, data)
        error = abs(actual - anterior)
        print("New point is: {}, {}\nError: {}\n".format(x0, y0, error))
        iteram += 1
    print("\nFinal point is {},{} ".format(x0,y0))
    return startTime, iteram
        #curve_x.append(x0)
        #curve_y.append(y1)

    #plot(curve_y, 'g*-')
    #plot(curve_x, 'r+-')
    #xlabel('iterations')
    #ylabel('objective function value')
    #legend(['backtracking line search algorithm'])
    #show()

if __name__ == "__main__":
    n = input("Which test run: \n[1] 100 iterations\n[2] 1000 iterations\n[3] 10000 iterations\n> ")
    try:
        n = int(n)
    except:
        print("Error")
        exit(1)
    if n == 1:
        n = 100
    elif n == 2:
        n = 1000
    elif n == 3:
        n = 10000
    times = list()
    iteram = list()
    for k in range(10):
        start, iteramount = GradientDescent(n)
        times.append(time.time() - start)
        iteram.append(iteramount)

        print("Time of the {} iteration: {}, number of iterations to solve: {}".format(k+1, times[k], iteramount))
    
    average = sum(times)/10 
    iteraverage = sum(iteram) / 10
    print("\nCPU average time of 10 iterations: {}[s]\nNumber of average iterations: {}\n".format(average, iteraverage))