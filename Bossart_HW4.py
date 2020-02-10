import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

data = pd.read_csv("oilspills.dat", delimiter = " ")

year = data['year'].to_numpy().reshape((26,1))
N = data['spills'].to_numpy().reshape((26,1))
b = data[['importexport', 'domestic']].to_numpy().reshape((26,2))
n = len(year)

def llprime(a1, a2):
    summation = []
    alpha = np.asarray((a1,a2)).T
    for i in range(n):
        summation.append(N[i]*b[i]/(b[i]@alpha)-b[i])
    summation=np.asarray(summation)
    summation=summation.sum(axis=0)
    return summation

#def llprime2
def llprime2(a1, a2):
    summation = []
    alpha = np.asarray((a1,a2)).T
    for i in range(n):
        summation.append((N[i]/(b[i]@alpha)**2)*b[i].reshape(-1,1)@b[i].reshape(1,2))
    summation=np.asarray(summation)
    summation=summation.sum(axis=0)
    return -summation

newtons_hess=[]
def newt_step(a1, a2):
    ie = -np.linalg.inv(llprime2(a1,a2))
    newtons_hess.append(ie)
    jj = llprime(a1,a2)
    return ie@jj

def newton_method(a1, a2, max_iterations, print_option):
    num_iterations = 0
    alpha_ones = []
    alpha_twos = []
    alpha_iter = np.asarray((a1,a2)).T
    
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
def fisher_step(a1, a2):
    summation = []
    alpha = np.asarray((a1,a2)).T
    for i in range(n):
        summation.append((-1/(b[i]@alpha))*b[i].reshape(-1,1)@b[i].reshape(1,2))
    summation=np.asarray(summation)
    summation=summation.sum(axis=0)
    fisherinv = np.linalg.inv(-summation)
    fishers_hess.append(fisherinv)
    step = fisherinv@llprime(a1, a2)
    return step;
        

def fisher_method(a1, a2, max_iterations, print_option):
    num_iterations = 0
    alpha_ones = []
    alpha_twos = []
    alpha_iter = np.asarray((a1,a2)).T
    
    while num_iterations < max_iterations:
        alpha_ones.append(alpha_iter[0])
        alpha_twos.append(alpha_iter[1])
        alpha_iter = alpha_iter + fisher_step(alpha_iter[0],alpha_iter[1]) 
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