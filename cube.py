# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 15:23:33 2021

@author: sj
"""
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from itertools import cycle, islice
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

from kmodes.kprototypes import KPrototypes


def row_echelon(A):
    r, c = A.shape
    if r == 0 or c == 0:
        return A
    for i in range(len(A)):
        if A[i,0] != 0:
            break
    else:
        B = row_echelon(A[:,1:])
        return np.hstack([A[:,:1], B])

    if i > 0:
        ith_row = A[i].copy()
        A[i] = A[0]
        A[0] = ith_row
    A[0] = A[0] / A[0,0]
    return A
def fastflightcube(p,X):
    nc=len(X[0])
    nr=len(X)
    u=[0]*nc
    uset=[0]*nc
    la1=la2=10**200
    la=eps=10**-9
    v=free=-1
    X=np.array(X)
    X=row_echelon(X)
    for i in range(nr-1,-1,-1):
        lead=0
        for j in range(0,nc):
            if X[i][j]==0:
                lead+=1
            else:
                break
    if lead<nc:
        v=0.0
        for j in range(lead+1,nc):
            if uset[j]==0:
                uset[j]=1
                free*=-1
                u[j]=free
            v-=u[j]*X[i][j]
        u[lead]=v/X[i][lead]
        uset[lead]=1
    for i in range(0,nc):
        if uset[i]==0:
            free*=-1
            u[i]=free
        else:
            break
    for i in range(0,nc):
        if u[i]>0:
            la1=min(la1,(1-p[i])/u[i])
            la2=min(la2,p[i]/u[i])
        if u[i]<0:
            la1=min(la1,-p[i]/u[i])
            la2=min(la2,(p[i]-1)/u[i])
    if la2/(la1+la2)>random.uniform(0, 1):
        la=la1
    else:
        la=-la2
    for i in range(0,nc):
        p[i]+=la*u[i]
        if p[i]<eps:
            p[i]=0
        if p[i]>1-eps:
            p[i]=1
    return p
def flightphase(X,P):
    n=len(P)
    NN=len(X[0])
    index=[]
    p=P.copy()
    for i in range(0,n):
        index.append(i)
    e=10**(-12)
    d=0
    random.shuffle(index)
    for i in range(d,n):
        if p[index[i]]<e or p[index[i]]>1-e:
            index[d],index[i]=index[i],index[d]
            d+=1
    while d<n:
        hm=min(NN+1,n-d)
        if hm<=NN:
            d=n
            break
        if hm>1:
            p_s=[0]*hm
            i_s=[0]*hm
            B=[[0 for i in range(hm)] for j in range(hm-1)]
            for i in range(0,hm):
                i_s[i]=index[d+i]
                for j in range(0,hm-1):
                    B[j][i]=X[i_s[i]][j]/P[i_s[i]]
                p_s[i]=p[i_s[i]]
            p_s=fastflightcube(p_s,B)
            for i in range(0,hm):
                p[i_s[i]]=p_s[i]
            hl=d+hm
            for i in range(d,hl):
                if p[index[i]]<e or p[index[i]]>1-e:
                    index[d],index[i]=index[i],index[d]
                    d+=1
        else:
            if p[index[d]]>random.uniform(0, 1):
                p[index[d]]=1
            else:
                p[index[d]]=0
            d=n
    for i in range(0,n):
        if p[index[i]]>1-e:
            p[index[i]]=1
        if p[index[i]]<e:
            p[index[i]]=0
    return p

def cube(X,P):
    n=len(P)
    NN=len(X[0])
    index=[]
    p=P.copy()
    for i in range(0,n):
        index.append(i)
    e=10**(-12)
    d=0
    random.shuffle(index)
    for i in range(d,n):
        if p[index[i]]<e or p[index[i]]>1-e:
            index[d],index[i]=index[i],index[d]
            d+=1
    while d<n:
        hm=min(NN+1,n-d)
        if hm>1:
            p_s=[0]*hm
            i_s=[0]*hm
            B=[[0 for i in range(hm)] for j in range(hm-1)]
            for i in range(0,hm):
                i_s[i]=index[d+i]
                for j in range(0,hm-1):
                    B[j][i]=X[i_s[i]][j]/P[i_s[i]]
                p_s[i]=p[i_s[i]]
            p_s=fastflightcube(p_s,B)
            for i in range(0,hm):
                p[i_s[i]]=p_s[i]
            hl=d+hm
            for i in range(d,hl):
                if p[index[i]]<e or p[index[i]]>1-e:
                    index[d],index[i]=index[i],index[d]
                    d+=1
        else:
            if p[index[d]]>random.uniform(0, 1):
                p[index[d]]=1
            else:
                p[index[d]]=0
            d=n
    n1=round(sum(p))
    s=[0]*n1
    count=0
    for i in range(0,n):
        if p[index[i]]>1-e:
            s[count]=index[i]+1
            count+=1
    return s

def eg():
    import csv 
    filename = "upd111.csv"
    fields = [] 
    rows = [] 
    with open(filename, 'r') as csvfile: 
        csvreader = csv.reader(csvfile) 
        fields = next(csvreader) 
        for row in csvreader: 
            rows.append(row) 
    X1=[]
    for i in range(0,len(rows)):
        xy=[]
        for j in rows[i]:
            xy.append(float(j))
        X1.append(xy.copy())
    n=300
    N=len(X1)
    p=[n/N]*N
    pi=cube(X1,p)
    print(len(pi))
    print(pi)
    
#eg()