#DANILO COURA MOREIRA
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

global w_do_zero

def compute_norma(gradient):
    return np.sqrt(np.dot(gradient.T,gradient))

def sigmoid(z):
    return 1/(1+np.exp(-z))

def compute_cost_func(w,X,Y):
    m = np.size(w,0)
    hx = sigmoid(np.dot(X,w))
    return (np.dot(Y.T,np.log(hx)) + np.dot((1 - Y).T,np.log(1-hx)))*(-1/m)

def step_gradient_vectorized(w_current,X,Y,learningRate):
    m = np.size(w_current,0)
    hx = sigmoid(np.dot(X,w_current))
    gradient = np.zeros((m,1))
    w = np.zeros((m,1))
    gradient = (np.dot(X.T,hx-Y) * (1/m))
    w = w_current - (learningRate * gradient)
    return [w,gradient]    	

def gradient_descent_runner_vectorized(w, X,Y, learning_rate, epsilon):
	i = 0
	norma = float('inf')
	while (norma>=epsilon):
		i+= 1
		[w,gradient] = step_gradient_vectorized(w, X, Y, learning_rate)
		norma = compute_norma(gradient)
		#if i % 10000 == 0:
		if i % 50 == 0:
			print("Custo na iteração {0} é de {1}".format(i,compute_cost_func(w, X, Y)))
			print("epsilon na iteração {0} é de {1}".format(i,norma))
	print("Custo na iteração final {0} é de {1}".format(i,compute_cost_func(w, X, Y)))    
	print("epsilon na iteração final {0} é de {1}".format(i,norma))
	return w

def predict(X):
	hx = sigmoid(np.dot(X,w_do_zero))
	hx[hx>= 0.5] = 1
	hx[hx< 0.5] = 0
	return hx

def score(X,Y):
	hx = predict(X)
	Z = np.zeros((hx.shape[0],hx.shape[1]))
	print(hx.shape)
	print(Y.shape)
	Z[hx==Y] = 1	
	return np.sum(Z)/hx.shape[0]

def organize_data(file):
	data = pd.read_csv(file)
	data = np.c_[np.ones(len(data)),data];
	X = data[:,0:5].astype('long');
	Y = data[:,5][:,np.newaxis];
	Y[Y == 'Iris-setosa'] = 0
	Y[(Y == 'Iris-versicolor') | (Y == 'Iris-virginica')] = 1
	Y = Y.astype('long')
	return [X, Y]

print("sigmoid")
print((1- sigmoid(1)) * sigmoid(1))

[X, Y] = organize_data("iris\iris.data")
init_w = np.zeros((X.shape[1],1));
learning_rate = 0.25;
epsilon = 0.0005
w_do_zero = gradient_descent_runner_vectorized(init_w, X,Y, learning_rate, epsilon)
print("Os coeficientes obtidos na Regressão Múltipla do Zero são: \n {0}".format(w_do_zero))
clf = LogisticRegression(C=1e15)
clf.fit(X,Y.reshape(Y.shape[0]))
w_sklearn = clf.coef_.T
w_sklearn[0,0] = clf.intercept_

def compute_mse_coefs(w_dozero, w_sklearn):
    res = w_dozero - w_sklearn
    totalError = np.dot(res.T,res)
    return totalError / float(len(w_dozero))
print("Os coeficientes obtidos na Regressão Múltipla do sklearn são: \n {0}".format(w_sklearn))
print("A diferença entre o os vetores que representam os coeficientes de ambas as abordagens é: \n {0}".format(compute_mse_coefs(w_do_zero,w_sklearn)))
print(score(X,Y))
print(clf.score(X,Y))
