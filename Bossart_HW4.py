import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.linalg import inv
import pandas as pd

data = pd.read_csv("oilspills.dat", delimiter = " ")

year = data['year'].to_numpy().reshape((26,1))
N = data['spills'].to_numpy().reshape((26,1))
b = data[['importexport', 'domestic']].to_numpy().reshape((26,2))
b1 = data[['importexport']].to_numpy().reshape((26,1))
b2 = data[['domestic']].to_numpy().reshape((26,1))
n = len(year)

def loglike(a1, a2):
    summation = []
    alpha = np.asarray((a1,a2)).reshape(2,1)
    for i in range(n):
        b_i = b[i].reshape(2,1).T
        summation.append(N[i]*np.log(b_i.dot(alpha))-np.log(np.math.factorial(N[i]))-(b_i.dot(alpha)))
    
    summation=np.asarray(summation)
    summation=summation.sum(axis=0)
    return summation

def llprime(a1, a2):
    summation = []
    alpha = np.asarray((a1,a2)).reshape((2,1))
    for i in range(n):
        Ni = N[i]
        bi = np.asarray(b[i]).reshape((2,1))
        dotprod = np.dot(alpha.T, bi)
        value = (Ni * bi / dotprod) - bi
        summation.append(value)
    summation=np.asarray(summation)
    summation=summation.sum(axis=0)
    return summation


#def llprime2
def llprime2(a1, a2):
    summation = []
    alpha = np.asarray((a1,a2)).reshape((2,1))
    for i in range(n):
        Ni = N[i]
        bi = np.asarray(b[i]).reshape((2,1))
        dotprod = np.dot(alpha.T, bi)**2
        value = (Ni/dotprod)*np.dot(bi, bi.T)
        summation.append(value)
    summation=np.asarray(summation)
    summation=summation.sum(axis=0)
    return -summation


newtons_hess=[]
def newt_step(a1, a2):
    ie = -np.linalg.inv(llprime2(a1,a2))
    newtons_hess.append(ie)
    jj = llprime(a1,a2)
    return np.dot(ie, jj)

def newton_method(a1, a2, max_iterations, print_option):
    num_iterations = 0
    alpha_ones = []
    alpha_twos = []
    alpha_iter = np.asarray((a1,a2)).reshape((2,1))
    
    while num_iterations < max_iterations:
        alpha_ones.append(alpha_iter[0])
        alpha_twos.append(alpha_iter[1])
        alpha_iter = alpha_iter + newt_step(alpha_iter[0],alpha_iter[1]) 
        num_iterations += 1

    if print_option == 0:
        alpha_ones = np.asarray(alpha_ones).reshape(num_iterations, 1)
        alpha_twos = np.asarray(alpha_twos).reshape(num_iterations, 1)
        alphas = np.hstack((alpha_ones, alpha_twos))
        return alphas
    
    if print_option == 1:
        print('Using Newtons method')
        print('Starting values: ', alpha_ones[0], alpha_twos[0])
        print('Approximation values: ', alpha_ones[-1], alpha_twos[-1])
        print('Number of iterations: ', num_iterations)
        
fishers_hess = []      


def fisher(a1, a2):
    summation = []
    alpha = np.asarray((a1,a2)).reshape((2,1))
    for i in range(n):
        b_i = b[i].reshape((2,1))
        summation.append((1/(alpha.T.dot(b_i)))*(b_i.dot(b_i.T)))
    summation=np.asarray(summation)
    summation=summation.sum(axis=0)
    return summation
        
fishers_hess = []
def fisher_method(a1, a2, max_iterations, print_option):
    num_iterations = 0
    alpha_ones = []
    alpha_twos = []
    alpha_iter = np.asarray((a1,a2)).reshape((2,1))
    
    while num_iterations < max_iterations:
        alpha_ones.append(alpha_iter[0])
        alpha_twos.append(alpha_iter[1])
        inv_fish = inv(fisher(alpha_iter[0], alpha_iter[1]))
        fishers_hess.append(inv_fish)
        alpha_iter = alpha_iter + np.dot(inv_fish, llprime(alpha_iter[0], alpha_iter[1]))
        num_iterations += 1

    if print_option == 0:
        alpha_ones = np.asarray(alpha_ones).reshape(num_iterations, 1)
        alpha_twos = np.asarray(alpha_twos).reshape(num_iterations, 1)
        alphas = np.hstack((alpha_ones, alpha_twos))
        return alphas
    
    if print_option == 1:
        print('\nUsing Fishers method')
        print('Starting values: ', alpha_ones[0], alpha_twos[0])
        print('Approximation values: ', alpha_ones[-1], alpha_twos[-1])
        print('Number of iterations: ', num_iterations)
        print()
    
    
    

a1 = 0.5
a2 = 0.5
newts = newton_method(a1, a2, 8, 0)
newtons_hess = np.asarray(newtons_hess)
fish = fisher_method(a1, a2, 10, 0)
fishers_hess = np.asarray(fishers_hess)

alpha1_se = []
alpha2_se = []
def mle_se():
    
    for i in range(4,10):
        alpha1_se.append(np.sqrt(fishers_hess[i, 0, 0]))
        alpha2_se.append(np.sqrt(fishers_hess[i, 1, 1]))
    print(alpha1_se)
    print(alpha2_se)

mle_se()



def steep_ascent(a1, a2, scale, max_iterations, print_option):
    identity = np.identity(2)
    alpha = np.asarray((a1, a2)).reshape((2,1))
    alpha_ones = [a1]
    alpha_twos = [a2]

    for i in range(max_iterations):
        new = alpha + scale*identity.dot(llprime(alpha[0], alpha[1])) 

        if loglike(new[0], new[1]) > loglike(alpha[0], alpha[1]):
            alpha = new
            alpha_ones.append(alpha[0])
            alpha_twos.append(alpha[1])
            
        else:
            scale = scale / 2.0
            
    if print_option == 0:
        alpha_ones = np.asarray(alpha_ones).reshape(max_iterations, 1)
        alpha_twos = np.asarray(alpha_twos).reshape(max_iterations, 1)
        alphas = np.hstack((alpha_ones, alpha_twos))
        return alphas
    
    if print_option == 1:
        print('\nUsing steepest ascent method')
        print('Starting values: ', alpha_ones[0], alpha_twos[0])
        print('Approximation values: ', alpha_ones[-1], alpha_twos[-1])
        print()
    
    
steep_ascent(a1, a2, 1, 100, 1)
m = -fisher(0.5, 0.5)


def update_hessian(alpha0, alpha1, m, max_iterations):
    num_iter = 0
    alph_list = [alpha0, alpha1]
    
    while num_iter < max_iterations:
        z = alpha1 - alpha0
        y = llprime(alpha1[0], alpha1[1]) - llprime(alpha0[0], alpha0[1])
        v = y - np.dot(m, z)
        denom = np.dot(v.T, z)  
        
        if ((denom>0) | (denom == 0)):
            m = m

        else:
            c = 1/denom
            m = m + c*np.dot(v, v.T)
               
        temp = alpha1
        alpha1 = alpha0 - np.dot(inv(m), llprime(alpha0[0], alpha0[1]))
        alph_list.append(alpha1)
        alpha0 = temp
        num_iter += 1
        
    print(alpha1)
    return

startalpha = np.asarray([0.5, 0.5]).reshape((2,1))
update_hessian(startalpha,startalpha, m, 100)