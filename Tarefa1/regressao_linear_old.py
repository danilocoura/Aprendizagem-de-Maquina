import numpy as np
import matplotlib.pyplot as plt

#norma < 0.04
def compute_norma(b_gradient, m_gradient):
	return np.sqrt(b_gradient**2 + m_gradient**2)

def compute_error_for_given_points(b, m, points):
	totalError = 0
	for i in range(0, len(points)):
		x = points[i,0]
		y = points[i,1]
		totalError += (y - (b + m*x))**2	
	return totalError / float(len(points))

def compute_error_for_given_points_vector(b, m, points):
	totalError = 0
	return np.sum(((points[:,1]) - (points[:,0]*m + b))**2)/float(len(points))	

def step_gradient(b_current, m_current, points, learningRate):
	#gradient descent
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))	
	for i in range(0, len(points)):
		x = points[i,0]
		y = points[i,1]
		b_gradient += -(2/N) * (y - ((m_current *x) + b_current))
		m_gradient += -(2/N) * x * (y - ((m_current *x) + b_current))
	new_b = b_current - (learningRate * b_gradient)		
	new_m = m_current - (learningRate * m_gradient)
	print('GradienteB = {0} - GradienteM = {1}'.format(b_gradient,m_gradient))
	print('Norma = {0}'.format(compute_norma(b_gradient, m_gradient)))				
	return [new_b, new_m, compute_norma(b_gradient, m_gradient)]	

def step_gradient_vector(b_current, m_current, points, learningRate):
	#gradient descent
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))	
	b_gradient = np.sum(((points[:,1]) - (m_current * points[:,0]) + b_current)) * -(2/N)
	m_gradient = np.sum(( points[:,0] * ((points[:,1]) - (m_current * points[:,0]) + b_current))) * -(2/N)
	new_b = b_current - (learningRate * b_gradient)		
	new_m = m_current - (learningRate * m_gradient)		
	return [new_b, new_m, compute_norma(b_gradient, m_gradient)]

def plot_points(points):
	plt.ion()
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(points[:,0],points[:,1], 'ro')
	ax.axis([0, 25, 0, 100])
	variaveis = range(0,100)
	return fig, ax


def plot_graph(fig, ax, variaveis, b, m, rss, e, it):
	line1, = ax.plot(m*variaveis + b)
	fig.canvas.draw()
	return
	
def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
	b = starting_b
	m = starting_m
	it = 0
	RSS = []
	GDE = []
	### 5a e 6a QUESTAO
	e = float('inf')
	while e > 0.04:
		it += 1
		### 2a QUESTAO
		rss = compute_error_for_given_points(b, m, points)
		RSS = np.append(RSS, rss)
		print('Iteração {0} - RSS: {1}'.format(it+1, rss))
		b, m, e = step_gradient_vector(b, m, np.array(points), learning_rate)
		GDE = np.append(GDE, e) 
		print('W0 = {0} - W1 = {1}'.format(b,m))
	return [b, m, it, RSS, GDE]

def run():
	### 1a QUESTAO
	points = np.genfromtxt("income.csv", delimiter=",")
	#hyperparameters
	### 4a QUESTAO
	learning_rate = 0.003
	#y = mx + b (slope formula)
	initial_b = -39
	initial_m = 5
	### 4a QUESTAO
	num_iterations = 16000
	[b, m, it, RSS, GDE] = gradient_descent_runner(points, initial_m, initial_b, learning_rate, num_iterations)

	plt.plot(range(0,it), GDE)
	#plt.axis([0, it, 0, np.max(GDE)])
	plt.show()
		
if __name__ == '__main__':
	run()