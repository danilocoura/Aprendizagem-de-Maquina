import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import  validation_curve
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

inclinacao = 0;

def read_data(file):
	data = pd.read_csv(file)
	#reorganiza os dados
	data = pd.concat([data.loc[:,'Vetorial':'LP2'], data.loc[:,'Cálculo1'], data.loc[:,'cra']], axis=1)
	#separa os dados
	X = data.loc[:,'Vetorial':'Cálculo1']
	Y = data.loc[:,'cra']
	return[X,Y]

def getvalue(data, col_name):
	result = np.mean(data[data["DISCIPLINA"] == col_name].loc[:,"MAT_MEDIA_FINAL"].values)
	return 0 if math.isnan(result) else result

def organize_test_data(file):
	data_treino = pd.read_csv(file)
	#reorganiza os dados
	data_treino = pd.concat([data_treino.loc[:,"ALU_NOVAMATRICULA"], data_treino.loc[:,'DISCIPLINA':'MAT_MEDIA_FINAL']], axis=1)
	data_treino.loc[:,"MAT_MEDIA_FINAL"] = data_treino.loc[:,"MAT_MEDIA_FINAL"].fillna(0)
	matriculas = data_treino["ALU_NOVAMATRICULA"].unique()
	colunas = ["Vetorial","LPT","P1","IC","LP1","Cálculo2","Discreta","P2","Grafos","Fís.Clássica","LP2","Cálculo1","cra"]
	notas = np.zeros((matriculas.size,np.size(colunas)))
	count = 0
	for i in matriculas:
		result = data_treino[data_treino["ALU_NOVAMATRICULA"] == i]
		cred = result.loc[:,"CREDITOS"]  
		nota = result.loc[:,"MAT_MEDIA_FINAL"]
		cra = np.dot(cred.T,nota)/np.sum(cred)
		vetorial = getvalue(result,"Álgebra Vetorial e Geometria Analítica")
		lpt = getvalue(result,"Leitura e Produção de Textos")
		p1 = getvalue(result,"Programação I")
		ic = getvalue(result,"Introdução à Computação")
		lp1 = getvalue(result,"Laboratório de Programação I")
		calculo2 = getvalue(result,"Cálculo Diferencial e Integral II")
		discreta = getvalue(result,"Matemática Discreta")
		p2 = getvalue(result,"Programação I")
		grafos = getvalue(result,"Teoria dos Grafos")
		fisclassica = getvalue(result,"Fundamentos de Física Clássica")
		lp2 = getvalue(result,"Laboratório de Programação II")
		calculo1 = getvalue(result,"Cálculo Diferencial e Integral I")
		notas[count,:] = [vetorial,lpt,p1,ic,lp1,calculo2,discreta,p2,grafos,fisclassica,lp2,calculo1,cra]
		count += 1
	df = pd.DataFrame(notas, index=matriculas, columns=colunas)		
	pd.concat([df.loc[:,'Vetorial':'LP2'], df.loc[:,'Cálculo1'], df.loc[:,'cra']], axis=1)
	#separa os dados
	X = df.loc[:,'Vetorial':'Cálculo1']
	Y = df.loc[:,'cra']
	return[X,Y]

def add_mean(X):
	media = np.sum(X,axis=1)/12
	X = pd.concat([X,media], axis=1)
	return X

#aplica log nos dados quando houver inclinacao maior que o limite estabelecido
def skew_data(X,threshold):
	global inclinacao
	inclinacao = X[:].apply(lambda x: skew(x))
	inclinacao = inclinacao[abs(inclinacao) > threshold]
	inclinacao = inclinacao.index
	X[inclinacao] = np.log1p(X[inclinacao])
	return X

#disponibiliza criação de variaveis polinomiais
def add_poly_feat(X, degree):
	poly = PolynomialFeatures(degree)
	return poly.fit_transform(X)

def print_rmse(train,validation,param, range, name):
    result = np.concatenate((train[:,np.newaxis], validation[:,np.newaxis]), axis=1)
    result = pd.DataFrame(result, index=range, columns=['Train','Validation'])
    print("{0} - {1} versus RMSE \n {2} \n".format(name, param, result))

def plot_rmse(train,validation,param, range, name):
	plt.xlabel(param)
	plt.ylabel('RMSE')
	plt.title(name)
	line1, = plt.plot(range, validation, label="Validation")
	line2, = plt.plot(range, train, label="Train")
	first_legend = plt.legend(handles=[line1], loc=1)
	ax = plt.gca().add_artist(first_legend)
	plt.legend(handles=[line2], loc=4)
	plt.show()

def calc_rmse(model,param,range,name):
    [train,validation] = validation_curve(model, X, Y, scoring="neg_mean_squared_error", param_name=param, param_range=range, cv = 5)
    train = np.sqrt(-train)
    validation = np.sqrt(-validation)
    train = train.mean(axis=1)
    validation = validation.mean(axis=1)
    print_rmse(train,validation,param, range, name)
    plot_rmse(train,validation,param, range, name)
    #recupera o melhor parametro e retorna o melhor modelo
    params = {param:range[np.argmin(validation)]}
    model.set_params(**params)
    model.fit(X,Y)
    return model

def graph_res_pred(model,X,Y,name):
	preds = pd.DataFrame({"preds":model_ridge.predict(X), "true":Y})
	preds["residuals"] = preds["true"] - preds["preds"]
	preds.plot(x = "preds", y = "residuals",kind = "scatter")
	plt.title(name)
	plt.show()    

[X,Y] = read_data("treino.csv")
X = add_mean(X)
X = skew_data(X,0.4)

model_ridge = Ridge()
alphas = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 1, 3, 5, 10, 15, 30, 50, 60, 75, 100]
model_ridge = calc_rmse(model_ridge,"alpha",alphas,"Ridge Regression")

model_lasso = Lasso()
#alphas = [1, 0.5, 0.25, 0.1, 0.005, 0.001, 0.0005, 0.0001 ]
alphas = [0, 0.0001, 0.0005, 0.001, 0.005, 0.1, 0.25, 0.5, 1]
calc_rmse(model_lasso,"alpha",alphas,"Lasso Regression")

model_knn = KNeighborsRegressor()
n_neighbors = [1,3,5,7,9,11]
calc_rmse(model_knn,"n_neighbors",n_neighbors,"KNN Regression")

model_kernelridge = KernelRidge()
alphas = [0.0001, 0.001, 0.1, 0.5, 1, 5, 10, 20, 30, 50]
calc_rmse(model_kernelridge,"alpha",alphas,"Kernel Ridge Regression")

model_svr = SVR()
C = [5, 2, 1, 0.1, 0.01]
calc_rmse(model_svr,"C",C,"SVR")

model_tree = DecisionTreeRegressor()
min_samples_split  = [2, 5, 10, 15, 20, 25, 50]
calc_rmse(model_tree,"min_samples_split",min_samples_split,"Decision Tree Regression")

model_forest = RandomForestRegressor()
min_samples_split  = [2, 5, 10, 15, 20, 25]
calc_rmse(model_tree,"min_samples_split",min_samples_split,"Random Forest Regression")

#[X_t,Y_t] = organize_test_data("graduados_teste.csv")
[X_t,Y_t] = read_data("treino.csv")
X_t = add_mean(X_t)
X_t[inclinacao] = np.log1p(X_t[inclinacao])
model_ridge.fit(X,Y)
pred = model_ridge.predict(X_t)
print(np.sqrt(mean_squared_error(Y_t, pred)))
graph_res_pred(model_ridge,X,Y,"Ridge Regression")