#DANILO COURA MOREIRA
from sklearn.linear_model import LinearRegression
import time as time
import numpy as np
import matplotlib.pyplot as plt

#norma < 0.04
def compute_norma(gradient):
	return np.sqrt(np.dot(gradient.T,gradient))

def compute_mse_vectorized(w,X,Y):
    res = Y - np.dot(X,w)
    totalError = np.dot(res.T,res)
    return totalError / float(len(Y))

def step_gradient_vectorized_old(w_current,X,Y,learningRate):
    m = np.size(w_current,0)
    res = Y - np.dot(X,w_current)
    gradient = np.zeros((m,1))
    w = np.zeros((m,1))
    for i in range(0, m):
        X1 = X[:,i][:,np.newaxis]
        gradient[i] = np.sum(np.multiply(res,X1))
    for i in range(0, m):
        w[i] = (w_current[i] + (2 * learningRate * gradient[i]))
    #print(w)
    #print(gradient)
    return [w,gradient]

def step_gradient_vectorized(w_current,X,Y,learningRate):
    m = np.size(w_current,0)
    res = Y - np.dot(X,w_current)
    gradient = np.zeros((m,1))
    w = np.zeros((m,1))
    gradient = np.sum(np.multiply(res,X),axis=0)[:,np.newaxis]
    w = w_current + (2 * learningRate * gradient)
    return [w,gradient]    	

def gradient_descent_runner_vectorized(w, X,Y, learning_rate, epsilon):
	i = 0
	norma = float('inf')
	while (norma>=epsilon):
		[w,gradient] = step_gradient_vectorized(w, X, Y, learning_rate)
		norma = compute_norma(gradient)
		if i % 10000 == 0:
			print("MSE na iteração {0} é de {1}".format(i,compute_mse_vectorized(w, X, Y)))
			print("epsilon na iteração {0} é de {1}".format(i,norma))
		i+= 1
	print("MSE na iteração final {0} é de {1}".format(i,compute_mse_vectorized(w, X, Y)))    
	print("epsilon na iteração final {0} é de {1}".format(i,norma))
	return w

points = np.genfromtxt("sample_treino.csv", delimiter=",", skip_header=1);
points = np.c_[np.ones(len(points)),points];
X = points[:,0:5];
Y = points[:,6][:,np.newaxis];
init_w = np.zeros((5,1));
learning_rate = 0.00004;
epsilon = 0.0001
w_do_zero = gradient_descent_runner_vectorized(init_w, X,Y, learning_rate, epsilon)
print("Os coeficientes obtidos na Regressão Múltipla do Zero são: \n {0}".format(w_do_zero))
clf = LinearRegression()
clf.fit(X,Y)
w_sklearn = clf.coef_.T
w_sklearn[0,0] = clf.intercept_

def compute_mse_coefs(w_dozero, w_sklearn):
    res = w_dozero - w_sklearn
    totalError = np.dot(res.T,res)
    return totalError / float(len(w_dozero))
print("Os coeficientes obtidos na Regressão Múltipla do sklearn são: \n {0}".format(w_sklearn))
print("A diferença entre o os vetores que representam os coeficientes de ambas as abordagens é: \n {0}".format(compute_mse_coefs(w_do_zero,w_sklearn)))