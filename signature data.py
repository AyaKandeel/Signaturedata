import pandas as pd
import sklearn as sk
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn import svm
import os
heart = pd.read_csv('SAHeart.csv', sep=',',header=0)
print(heart.head())
y = heart.iloc[:,9]
X = heart.iloc[:,:9]
LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(X, y)
LR.predict(X.iloc[460:,:])
round(LR.score(X,y), 4)
SVM=svm.linearSVC()
SVM.fit(x,y)
SVM.predict(x.iloc[460:,:])
round(SVM.score(x,y),4)


 

