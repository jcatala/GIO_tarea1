#!/usr/bin/env python3

import math
import numpy as np
import random
import matplotlib.pyplot as plt
import time

def calculateDistance(x1,x2,y1,y2):
    xVal = pow(x1-x2, 2)
    yVal = pow(y1-y2, 2)
    return math.sqrt(xVal + yVal)




def newPoints(data):
    xdenom = 0
    xnumer = 0
    ydenom = 0
    ynumer = 0
    for x,y,w,d2 in data:
        xnumer += (w*x)/d2
        xdenom += (w)/d2
        ynumer += (w*y)/d2
        ydenom += (w)/d2
    x = xnumer / xdenom
    y = ynumer / ydenom
    return (x,y)


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


def solveN(n):
    maxK = n*100
    ERR = 0.001

    data = []
    datai= []
    initialData = []
    # Generamos la lista de puntos y pesos respectivos
    for k in range(n):
        print("[{}/{}] Generating list...".format(str(k+1), str(n)) ,end="\r")
        time.sleep(0.001)
        xn = int(random.uniform(0,n))
        yn = int(random.uniform(0,n))
        weight = random.uniform(n/10,(n/10)*5)
        initialData.append((xn,yn,weight))

    # Computamos el punto inicial
    p0 = initialPoint(initialData)
    #p0 = (0,0)

    print("The initial point is: {}".format(p0))

    # Creamos la primera iteración, y poblamos la lista de datos.
    startTime = time.time()
    for xn,yn,weight in initialData:
        d2p0 = calculateDistance(xn, p0[0], yn, p0[1]) # Calculate distance to the p0
        datai.append(  (xn,yn, weight, d2p0) ) # List with all the points, weights and distance to p0
    data.append(datai) 
    print()
    #print (data)
    print ("List generated,")
    print ("Starting to solve...")
    zlist = []
    zlist.append(0) # zlist will be the list with all the Z's, the difference will be between the last 2 added.
    for k in data[0]:
        zlist[0] += k[-1]* k[-2]
    #print("Initial z0: {}".format(str(zlist[0]) ) )
    k = 0
    while k <= maxK:
        # iteramos
        print("[{}/{}] Iteration".format(str(k), str(n)), end="\r")
        time.sleep(0.1)
        # Nuevo punto de prueba
        newPoint = newPoints(data[k])
        print("New point is: {}".format(newPoint))
        # Start the iteration
        xn = newPoint[0]
        yn = newPoint[1]
        datai = []
        zn = 0
        for pointVal in data[k]:
            # Distancia a cada punto
            d2p0 = calculateDistance(pointVal[0],newPoint[0], pointVal[1], newPoint[1])
            weight = pointVal[2]
            # Poblamos data de la nueva iteración
            datai.append( (pointVal[0] , pointVal[1] , weight, d2p0) )
            zn += (weight * d2p0)
        data.append(datai)
        zlist.append(zn)
        # Dif de Z
        zdif = abs(zlist[-1] - zlist[-2])
    
        if zdif < ERR :
            print("converge. stopping")
            break
        k = k + 1
    return startTime, k
        



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
        start,iteramount = solveN(n)
        times.append(time.time() - start)
        iteram.append(iteramount)
        
        print("Time of the {} iteration: {}, number of iterations to solve: {}".format(k+1, times[k], iteramount))
    
    average = sum(times)/10 
    iteraverage = sum(iteram) / 10
    print("\nCPU average time of 10 iterations: {}[s]\nNumber of average iterations: {}\n".format(average, iteraverage))