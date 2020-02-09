import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

data = pd.read_csv("oilspills.dat", delimiter = " ")

year = data['year'].to_numpy().reshape((26,1))
N = data['spills'].to_numpy().reshape((26,1))
b = data[['importexport', 'domestic']].to_numpy().reshape((26,2))
n = 26

def llprime(alpha):
    alpha = np.asarray(alpha).reshape((2,1))
    temp = []
    for i in range(n):
        dotproduct = np.dot(b[i], alpha)
        val = (N[i]/dotproduct) - 1
        temp.append(val*b[i])
    temp = np.asarray(temp)
    final = np.sum(temp, axis = 0)
    return final;

#def llprime2
def llprime2(alpha):
    alpha = np.asarray(alpha).reshape((2,1))
    temp = []
    for i in range(n):
        dotproduct2 = np.dot(b[i], alpha)**2
        val = -(N[i]/dotproduct2)
        bitrans = np.transpose(b[i])
        temp.append(val*np.outer(b[i],bitrans))
    temp = np.asarray(temp)
    final = np.sum(temp, axis = 0)
    return final;

alpha = [1, 1]

def newton_method(alpha, tol, max_iterations, print_option):
    alpha_vals = np.asarray([alpha])
    num_iterations = 0
    
    
    while abs(num_iterations < max_iterations):
        old_alpha = alpha_vals[num_iterations].reshape(2,1)
        new_alpha = old_alpha - step(old_alpha)
        num_iterations += 1
        np.append(alpha_vals, new_alpha)
        alpha = new_alpha
        
    alpha = np.asarray(alpha)
    # at this point we have broken out of the while loop    
    if num_iterations == max_iterations:
        print("Exceeded maximum number of iterations.")
        return
        
    # this is where we hope to be if newtons went well    
    if print_option == 1:
        print('\n')
        print("Starting value: " + str(alpha_vals[0]))
        sol = alpha_vals[-1]
        print("Number of iterations: " + str(num_iterations) + " \nTolerance: " + str(tol))
        print("Final solution: " + str(sol))
        return
    
    if print_option == 0:
        return alpha_vals[-1]
    

alpha = [0.5, 0.5]
print(newton_method(alpha, 1e-6, 1000, 1))