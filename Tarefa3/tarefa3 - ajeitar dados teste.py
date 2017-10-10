import math
import pandas as pd
import numpy as np
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

def getvalue(data, col_name):
	result = np.mean(data[data["DISCIPLINA"] == col_name].loc[:,"MAT_MEDIA_FINAL"].values)
	return 0 if math.isnan(result) else result

data_treino = pd.read_csv("graduados_teste.csv")
#reorganiza os dados
data_treino = pd.concat([data_treino.loc[:,"ALU_NOVAMATRICULA"], data_treino.loc[:,'DISCIPLINA':'MAT_MEDIA_FINAL']], axis=1)
data_treino.loc[:,"MAT_MEDIA_FINAL"] = data_treino.loc[:,"MAT_MEDIA_FINAL"].fillna(0)
matriculas = data_treino["ALU_NOVAMATRICULA"].unique()
colunas = ["Vetorial","LPT","P1","IC","LP1","Cálculo2","Discreta","P2","Grafos","Fís.Clássica","LP2","cra","Cálculo1"]
notas = np.zeros((matriculas.size,np.size(colunas)))
#print("notas")
#print(notas)
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
	notas[count,:] = [vetorial,lpt,p1,ic,lp1,calculo2,discreta,p2,grafos,fisclassica,lp2,cra,calculo1]
	count += 1
return pd.DataFrame(notas, index=matriculas, columns=colunas)	
