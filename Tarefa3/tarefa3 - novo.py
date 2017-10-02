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

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

data = pd.read_csv("treino.csv")
#reorganiza os dados
data = pd.concat([data.loc[:,'Vetorial':'LP2'], data.loc[:,'Cálculo1'], data.loc[:,'cra']], axis=1)

#aplica log nos dados quando houver inclinacao maior que o limite estabelecido
inclinacao = data[:].apply(lambda x: skew(x))
limite_inclinacao = 0.4
inclinacao = inclinacao[abs(inclinacao) > limite_inclinacao]
inclinacao = inclinacao.index
data[inclinacao] = np.log1p(data[inclinacao])

X = data.loc[:,'Vetorial':'Cálculo1']
#calcula e adiciona a media
media = np.sum(X,axis=1)/12
X = pd.concat([X,media], axis=1)
Y = data.loc[:,'cra']

#disponibiliza criação de variaveis polinomiais
degree = 1
poly = PolynomialFeatures(degree)
X = poly.fit_transform(X)


def print_rmse(train,validation,param, range, name):
    result = np.concatenate((train[:,np.newaxis], validation[:,np.newaxis]), axis=1)
    result = pd.DataFrame(result, index=range, columns=['Train','Validation'])
    print("{0} - {1} versus RMSE \n {2} \n".format(name, param, result))

def plot_rmse(train,validation,param, range, name):
	plt.xlabel(param)
	plt.ylabel('RMSE')
	plt.title(name)
	line1, = plt.plot(range, train, label="Train")
	line2, = plt.plot(range, validation, label="Validation")
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

def graph_res_pred(model,X,Y,name):
	preds = pd.DataFrame({"preds":model_ridge.predict(X), "true":Y})
	preds["residuals"] = preds["true"] - preds["preds"]
	preds.plot(x = "preds", y = "residuals",kind = "scatter")
	plt.show()    

model_ridge = Ridge()
alphas = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 1, 3, 5, 10, 15, 30, 50, 60, 75, 100]
calc_rmse(model_ridge,"alpha",alphas,"Ridge Regression")
graph_res_pred(model_ridge,X,Y,"Ridge Regression")

model_lasso = Lasso()
alphas = [1, 0.5, 0.25, 0.1, 0.005, 0.001, 0.0005, 0.0001 ]
calc_rmse(model_lasso,"alpha",alphas,"Lasso Regression")

model_knn = KNeighborsRegressor()
n_neighbors = [1,3,5,7,9,11]
calc_rmse(model_knn,"n_neighbors",n_neighbors,"KNN Regression")

model_kernelridge = KernelRidge()
alphas = [0, 0.1, 0.5, 1, 3, 5, 10]
calc_rmse(model_kernelridge,"alpha",alphas,"Kernel Ridge Regression")

model_svr = SVR()
C = [1, 2, 5, 10]
calc_rmse(model_svr,"C",C,"SVR")

model_tree = DecisionTreeRegressor()
min_samples_split  = [2, 5, 10, 15, 20, 25]
calc_rmse(model_tree,"min_samples_split",min_samples_split,"Decision Tree Regression")

model_forest = RandomForestRegressor()
min_samples_split  = [2, 5, 10, 15, 20, 25]
calc_rmse(model_tree,"min_samples_split",min_samples_split,"Random Forest Regression")
