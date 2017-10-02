import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet,Lasso, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score, validation_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

media = 0
maximo = 0
minimo = 0

def my_normalize(x):
	global media
	media = np.mean(x)
	global maximo
	maximo = np.max(x)
	global minimo
	minimo = np.min(x)
	return (x-media)/(maximo-minimo)

def normalize_pred_in(x):
	global media
	global maximo
	global minimo
	return (x-media.loc['Vetorial':'Cálculo1'])/(maximo.loc['Vetorial':'Cálculo1']-minimo.loc['Vetorial':'Cálculo1'])

def normalize_pred_out(x):
	global media
	global maximo
	global minimo
	return (x)*(maximo.loc['cra']-minimo.loc['cra'])+media.loc['cra']

data_aux = pd.read_csv("treino.csv")
#reorganiza os dados
data_aux = pd.concat([data_aux.loc[:,'Vetorial':'LP2'], data_aux.loc[:,'Cálculo1'], data_aux.loc[:,'cra']], axis=1)
print("teste scaler")

print(data_aux)

#aplica log nos dados quando houver inclinacao maior que o limite estabelecido
inclinacao = data_aux[:].apply(lambda x: skew(x))
print(inclinacao)
limite_inclinacao = 0.4
inclinacao = inclinacao[abs(inclinacao) > limite_inclinacao]
inclinacao = inclinacao.index
print("inclinacao")
print(inclinacao)
data_aux[inclinacao] = np.log1p(data_aux[inclinacao])

#data_aux = my_normalize(data_aux)
#scaler = StandardScaler(with_mean=False)
#scaler.fit(data)
#data=scaler.transform(data)

X = data_aux.loc[:,'Vetorial':'Cálculo1']
mediaa = np.sum(X,axis=1)/12
X = pd.concat([X,mediaa], axis=1)
print(X)

poly = PolynomialFeatures(1)
X = poly.fit_transform(X)
Y = data_aux.loc[:,'cra']

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X, Y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

def rmse_train_validade(model,param,range):
    [train,validade] = validation_curve(model, X, Y, scoring="neg_mean_squared_error", param_name=param, param_range=range, cv = 5)
    train = np.sqrt(-train)
    validade = np.sqrt(-validade)
    return[train, validade]

model_ridge = Ridge(normalize=False)

alphas = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 1, 3, 5, 10, 15, 30, 50, 60, 75, 100]
train,validade =rmse_train_validade(model_ridge,"alpha",alphas)
train = train.mean(axis=1)
validade = validade.mean(axis=1)

print(train)
print(validade)

train = pd.Series(train, index = alphas)
train.plot(title = "Validation - Just Do It")
validade = pd.Series(validade, index = alphas)
validade.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show()

'''
X_predict = data_aux.loc[1,'Vetorial':'Cálculo1']
#print(media)
#print(minimo)
#print(maximo)
X_predict = normalize_pred_in(X_predict)
print("valores")
print(X_predict)
print(normalize_pred_out(model_ridge.predict(X_predict)))

#RESIDUAIS X PREDIÇÕES

preds = pd.DataFrame({"preds":model_ridge.predict(X), "true":Y})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter")
plt.show()

print(pd.concat([preds["true"], preds["preds"]], axis=1))

'''
model_lasso = Lasso()

alphas = [1, 0.5, 0.25, 0.1, 0.005, 0.001, 0.0005, 0.0001 ]
train,validade =rmse_train_validade(model_lasso,"alpha",alphas)
train = train.mean(axis=1)
validade = validade.mean(axis=1)

print(train)
print(validade)

train = pd.Series(train, index = alphas)
train.plot(title = "Lasso - Just Do It")
validade = pd.Series(validade, index = alphas)
validade.plot(title = "Lasso - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show() 

model_knn = KNeighborsRegressor()

n_neighbors = [1,3,5,7,9,11]
train,validade =rmse_train_validade(model_knn,"n_neighbors",n_neighbors)
validation_curve(model_knn, X, Y, scoring="neg_mean_squared_error", param_name="n_neighbors", param_range=n_neighbors, cv = 5)
train = train.mean(axis=1)
validade = validade.mean(axis=1)

print(train)
print(validade)

train = pd.Series(train, index = n_neighbors)
train.plot(title = "KNN - Just Do It")
validade = pd.Series(validade, index = n_neighbors)
validade.plot(title = "KNN - Just Do It")
plt.xlabel("neighbors")
plt.ylabel("rmse")
plt.show()

model_kernelridge = KernelRidge()

alphas = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 1, 3, 5, 10, 15, 30, 50, 60, 75, 100]
train,validade =rmse_train_validade(model_ridge,"alpha",alphas)
train = train.mean(axis=1)
validade = validade.mean(axis=1)

print(train)
print(validade)

train = pd.Series(train, index = alphas)
train.plot(title = "KernelRidge - Just Do It")
validade = pd.Series(validade, index = alphas)
validade.plot(title = "KernelRidge - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show()

model_svr = SVR()

C = [1, 2, 5, 10]
train,validade =rmse_train_validade(model_svr,"C",C)
train = train.mean(axis=1)
validade = validade.mean(axis=1)

print(train)
print(validade)

train = pd.Series(train, index = C)
train.plot(title = "SVR - Just Do It")
validade = pd.Series(validade, index = C)
validade.plot(title = "SVR - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show()

model_tree = DecisionTreeRegressor()

min_samples_split  = [2, 5, 10, 15, 20, 25]
train,validade =rmse_train_validade(model_tree,"min_samples_split",min_samples_split)
train = train.mean(axis=1)
validade = validade.mean(axis=1)

print(train)
print(validade)

train = pd.Series(train, index = min_samples_split)
train.plot(title = "Tree - Just Do It")
validade = pd.Series(validade, index = min_samples_split)
validade.plot(title = "Tree - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show()

model_forest = RandomForestRegressor()

min_samples_split  = [2, 5, 10, 15, 20, 25]
train,validade =rmse_train_validade(model_forest,"min_samples_split",min_samples_split)
train = train.mean(axis=1)
validade = validade.mean(axis=1)

print(train)
print(validade)

train = pd.Series(train, index = min_samples_split)
train.plot(title = "Forest - Just Do It")
validade = pd.Series(validade, index = min_samples_split)
validade.plot(title = "Forest - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show()