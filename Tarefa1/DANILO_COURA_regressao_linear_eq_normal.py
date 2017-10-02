#DANILO COURA MOREIRA

import time as time
import numpy as np
import matplotlib.pyplot as plt

def normal_equation(points):
	points_x_with_1 = np.append(np.ones((len(points),1)),points[:,0:1],axis=1)
	#w = (H'H)-1 H'y 
	return np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(points_x_with_1),points_x_with_1)),np.transpose(points_x_with_1)),points[:,1])

def run():
	tic = time.time()
	points = np.genfromtxt("income.csv", delimiter=",")
	#y = w0 + w1x (slope formula)
	[w0, w1] = normal_equation(points)
	tac = time.time()
	print('Tempo de processamento versão Equação Normal: {0:0.2f} segundos'.format(tac-tic))
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