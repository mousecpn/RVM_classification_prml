# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 09:20:19 2019

@author: Edsong
"""

import numpy as np
from keras.utils import np_utils
from keras.datasets import mnist

def get_data():
    (x_train, y_train), (x_validation, y_validation) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_validation = x_validation.astype('float32')
    x_train = x_train/255.0
    x_validation = x_validation/255.0
    
    # 进行one-hot编码
    y_train = np_utils.to_categorical(y_train)
    y_validation = np_utils.to_categorical(y_validation)
#    num_classes = y_train.shape[1]
    return x_train,y_train,x_validation,y_validation

def Gaussian(x,mean,var):
    var_inv = np.linalg.pinv(var)
    n = x.shape[0]
    b = 1/((2*np.pi)**(n/2)*np.linalg.det(var))
    p = b*np.exp(-1/2*np.dot(np.dot((x-mean).T,var_inv),(x-mean)))
    return p

def Gaussian_kernel(x1,x2,variance=1):
    k = np.exp(-np.sum(np.square(np.abs(x1-x2)))/(2*variance**2))
    return k

def Gaussian_kernel_mat(x):
    n = x.shape[0]
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if j > i:
                K[i,j] = K[j,i]
            else:
                K[i,j] = Gaussian_kernel(x[i,:],x[j,:],variance = 0.5)
    return K

def sigmoid(x):
    return 1/(1+np.exp(-x))

def rvm(K,weight):
    y_pred = sigmoid(np.dot(K,weight))
    return y_pred
def diag_rev(A):
    a=np.dot(A,np.ones((A.shape[0],1)))
    return a
    
if __name__=="__main__":
    x_train,y_train,x_validation,y_validation = get_data()
    n = 1000
    
    x_train = x_train[:n,:].reshape((n,-1))
    y_train = y_train[:n,3]
#    x_train = np.concatenate((x_train,np.ones((n,1))),axis=1)
    x_validation = x_validation.reshape((x_validation.shape[0],-1))
#    x_validation = np.concatenate((x_validation,np.ones((n,1))),axis=1)
    
    K_train = Gaussian_kernel_mat(x_train)
#    K_train = np.concatenate((K_train,np.ones((n,1))),axis=1)
    A = np.eye(n)
    iteration = 5
    weight = np.ones((n,1))
    train_acc = []
    I =+ 0.001*np.eye(n)
    
    y_train = y_train.reshape((-1,1))
    
    for i in range(iteration):
        print("epoch:" , i+1)
        y_prob = rvm(K_train,weight)
        weight = np.dot(np.dot(np.linalg.pinv(A+I),K_train.T),y_train.reshape((-1,1))-y_prob)
        B = y_prob*(1-y_prob)
        B = np.dot(B,np.ones((1,A.shape[0])))
        var_posterior = np.linalg.inv(np.dot(np.dot(K_train.T,B.T),K_train) + A + I)
        
        gamma = np.eye(A.shape[0]) - np.multiply(A,var_posterior)
        w_sq_diag = np.diag(np.square(weight))
        A = gamma/w_sq_diag
        

        
        y_pred = np.zeros((A.shape[0],1))
        y_pred[y_prob>0.5] = 1
    
        train_acc.append(np.sum(np.abs(y_train.reshape((-1,1))-y_prob))/n)
        
        if np.max(A) > 1e+50:
            break
        
        # actificial pruning-out
#        a = diag_rev(A)
#        delpos = np.where(a>1e+80)[0]
#        A = np.delete(A,delpos , axis=0)
#        A = np.delete(A,delpos , axis=1)
#        K_train = np.delete(K_train,delpos , axis=0)
#        K_train = np.delete(K_train,delpos , axis=1)
#        weight = np.delete(weight,delpos , axis=0)
#        y_train = np.delete(y_train,delpos , axis=0)
#        
#        I = np.delete(I,delpos , axis=0)
#        I = np.delete(I,delpos , axis=1)
#        print("acc:",train_acc)
    
    
        
    
    