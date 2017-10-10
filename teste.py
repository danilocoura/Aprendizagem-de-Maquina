import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LogisticRegression
X = [[3,2],[5,2],[4,2],[10,7],[5,4],[5,1],[6,4],[8,6],[9,7],[8,5]]
#Y = [0,0,0,0,0,0,0,0,0,0]
model = OneClassSVM()
print(model.fit(X))
print(model.decision_function(X))
print("teste")
print(model.predict([4,2]))