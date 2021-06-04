import numpy as np
import math


# Name: Tien Li Shen
# Homework 1 Q1

def pt1n2(m, n):
    pt1 = np.zeros((m,n))
    pt2 = np.random.rand(m,n)
    print(pt1)
    print(pt2)
    return (pt1, pt2)

def pt3(v): # compute norm
    nor = sum(sum(v**2))**(1/2) # element-wise square -> sum -> sum -> sqrt
    print(nor)
    return nor

def pt4(u ,v):
    dot = np.dot(u[0], v[0])
    angle = np.arccos(dot/(pt3(u)*pt3(v)))
    print(angle)
    euc_dist = (sum(sum((u-v)**2)))**(1/2)
    print(euc_dist)

def pt5(a):
    mn = a.shape[0] * a.shape[1]
    b = np.reshape(a, (mn, 1))
    print(b)


if __name__ == '__main__':
    m = 5
    n = 4
    v = np.array([[1, 2, 3, 4]])
    u = np.array([[4, 3, 2, 1]])
    
    print("pt 1 and 2")
    pt1n2(m, n)
    print("pt 3")
    pt3(v)
    print("pt 4")
    pt4(u,v)
    a = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
    print("pt 5")
    pt5(a)