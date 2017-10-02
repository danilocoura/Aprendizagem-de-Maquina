#DANILO COURA MOREIRA

import time as time
import numpy as np
import matplotlib.pyplot as plt

#norma < 0.04
def compute_norma(w0_gradient, w1_gradient):
	return np.sqrt(w0_gradient**2 + w1_gradient**2)

def compute_error_for_given_points_vectorized(w0, w1, points, points_x_with_1):
	totalError = 0
	coef = np.append(w0,w1)
	#return np.sum(((points[:,1]) - (points[:,0]*w1 + w0))**2)/float(len(points))	TAMBÉM FUNCIONA
	return np.sum(((points[:,1]) - np.dot(coef,np.transpose(points_x_with_1)))**2)/float(len(points))	

def step_gradient_vectorized(w0_current, w1_current, points, points_x_with_1, learningRate):
	#gradient descent
	N = float(len(points))
	coef = np.append(w0_current,w1_current)
	w0_gradient = np.sum((points[:,1] - np.dot(coef,np.transpose(points_x_with_1)))* -(2/N)) 
	w1_gradient = (np.dot(np.transpose(points_x_with_1[:,1:2]),(points[:,1] - np.dot(coef,np.transpose(points_x_with_1))))* -(2/N))[0]
	new_w0 = w0_current - (learningRate * w0_gradient)		
	new_w1 = w1_current - (learningRate * w1_gradient)
	print('W0 = {0}'.format(new_w0))
	print('W1 = {0}'.format(new_w1))
	print('Gradiente (norma) = {0} \n'.format(compute_norma(w0_gradient, w1_gradient)))				
	return [new_w0, new_w1, compute_norma(w0_gradient, w1_gradient)]	

def gradient_descent_runner(points, starting_w0, starting_w1, learning_rate, num_iterations):
	w0 = starting_w0
	w1 = starting_w1
	it = 0
	#Pontos x precedidos de uma coluna de 1s
	points_x_with_1 = np.append(np.ones((len(points),1)),points[:,0:1],axis=1)
	var_RSS = []
	var_GDE = []
	### 5a e 6a QUESTAO
	e = float('inf')
	while e > 0.04:
		### 2a QUESTAO
		rss = compute_error_for_given_points_vectorized(w0, w1, points, points_x_with_1)
		var_RSS = np.append(var_RSS, rss)
		print('Iteração [{0}]'.format(it+1))
		print('RSS: {0}'.format(rss))
		w0, w1, e = step_gradient_vectorized(w0, w1, np.array(points), points_x_with_1, learning_rate)
		var_GDE = np.append(var_GDE, e) 
		#print('W0 = {0} - W1 = {1}'.format(w0,w1))
		it += 1
	return [w0, w1, it, var_RSS, var_GDE]

def run():
	tic = time.time()
	### 1a QUESTAO
	points = np.genfromtxt("income.csv", delimiter=",")
	#hyperparameters
	### 4a QUESTAO
	learning_rate = 0.003
	#y = w0 + w1x (slope formula)
	initial_w0 = 0
	initial_w1 = 0
	### 4a QUESTAO
	num_iterations = 16000
	[w0, w1, it, var_RSS, var_GDE] = gradient_descent_runner(points, initial_w0, initial_w1, learning_rate, num_iterations)
	tac = time.time()
	print('Tempo de processamento versão Vetorização: {0:0.2f} segundos'.format(tac-tic))
	axis_x = range(0,25)
	plt.figure(1)
	plt.xlabel('Escolaridade (anos)')
	plt.ylabel('Salário (mil/ano)')
	plt.title('Regressão Linear')
	plt.plot(points[:,0], points[:,1], 'ro')
	plt.plot(axis_x, w0+w1*axis_x)
	plt.text(5, 60, 'Y = {0:0.3f} + {1:0.3f}x'.format(w0,w1), fontsize=12)
	plt.show()
		
if __name__ == '__main__':
	run()