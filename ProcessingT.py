import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import seaborn as sns
from ROC import multiclass_ROC
from Learning_curve import plot_learning_curve
from sklearn.utils.testing import ignore_warnings
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from PLOT import Plots
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  confusion_matrix
from sklearn.model_selection import ShuffleSplit
from Grids import Grid
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from ROC import multiclass_ROC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import plotly.plotly as py
import plotly.graph_objs as go
from sklearn.svm import LinearSVC
from Transform import Transform
from sklearn.utils import check_random_state
from sklearn.preprocessing import KBinsDiscretizer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.model_selection import validation_curve
from sklearn.decomposition import PCA
import graphviz
from sklearn.model_selection import cross_validate
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from SPLIT import get_safe_balanced_split


#dot_data = tree.export_graphviz(d, out_file=None) 
#graph = graphviz.Source(dot_data)                      
#graph.render("iris")  

#IMPORT DATA
cols=['VP','VS','RHO','Y']
X=pd.read_csv('LOGS2.csv', sep=';') 
X=X[cols] 
Log=X.loc[X.VP < 1900 ].copy()  
X.drop(Log.loc[Log.Y==1].index,axis=0,inplace=True)
X.VP=X.VP+np.random.normal(0,X.VP.std(),X.shape[0])
X.VS=X.VS+np.random.normal(0,X.VS.std(),X.shape[0])
X.RHO=X.RHO+np.random.normal(0,X.RHO.std(),X.shape[0])

Xt=pd.read_csv('AVAIMP.txt',sep=';')
Xt=Xt[cols]
#Log=Xt.loc[Xt.VP < 1900 ].copy()
#Xt.drop(Log.loc[Log.Y==1].index,axis=0,inplace=True)


Xt=Test
X=X.sample(frac=1)
y=X.Y
yt=Xt.Y
T=Transform()
T.transform(X)
T.transform(Xt)
columns=['RHO','VP','VS','K','Zp','Zs','u','Lame']
X=X[columns]
Xt=Xt[columns]
#Xt=pd.read_csv('NUOVOTEST.txt', sep=';') 
#cols2=['Z','RHO','VP','VS','PIGE','SW','SH','Y']    
#Xt.columns=cols2
#Xt=Xt[cols2]

#colors = ["pale red", "denim blue", "medium green"]
#
#Xt2=pd.read_csv('Avares.txt', sep=';')     
#Xt2.columns=cols
#Xt2.RHO=Xt2.RHO/1000
#X=X.sample(random_state=1,frac=1).reset_index(drop=True)
#
#a=pd.read_csv('Test.txt', sep=';') 
#a.columns=['1','2','3']
#Test=pd.DataFrame()
#for i in a['1']:
#    Test=Test.append(X.loc[(np.abs(np.round(X.VP,2) - i)).argmin()].transpose())
#    X.drop((np.abs(np.round(X.VP,2) - i)).argmin(),inplace=True)
#Xt=Test
#
#Log=Xt.loc[Xt.VP < 1900 ].copy()
#Xt.drop(Log.loc[Log.Y==1].index,axis=0,inplace=True)
#
#random_state=1
#random_state=check_random_state(random_state)
#y=X.Y
#yt=Xt.Y
##yt2=Xt2.Y
#pl=Plots()
#T=Transform()
#T.transform(X) 
#T.transform(Xt)
#scale=StandardScaler()
#grids=Grid()
#columns=['VP','VS','RHO','K','Zp','Zs','u','Lame']
#
#X=X[columns]
#Xt=Xt[columns]
#Xt2=Xt2[columns]
#
#cv = StratifiedKFold(n_splits=5, random_state=random_state)
##
#estimators = [ ('scale',scale),('sm',sm),('clf',RandomForestClassifier(random_state=random_state))]
#
#
#pipe = imbPipeline(estimators1)
#
#param_grid = dict(sm__k_neighbors=range(2,4) ,sm__sampling_strategy=smotgrid ,clf__n_neighbors=Grid.KNN()['n_neighbors'],clf__p=Grid.KNN()['p'],clf__weights=Grid.KNN()['weights'])
#param_grid1 = dict(clf__n_neighbors=Grid.KNN()['n_neighbors'],clf__p=Grid.KNN()['p'],clf__weights=Grid.KNN()['weights'])
#
##title = "Learning Curves"
##plot_learning_curve(pipe,title, X,  y , ylim=None, cv=cv, 
##                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
##plt.show()
#
#rid_search = GridSearchCV(pipe, param_grid=param_grid1,cv=cv,n_jobs=-1,verbose=1,scoring='f1_micro')
##rid_search.fit(X,y)
#
#yp=rid_search.best_estimator_.predict(Xt)
#
#with open("Final_report.txt", "w") as text_file:
#    print(f"rep: {rep}", file=text_file)
#
#
#rep=classification_report(yt,yp,digits=5  ,target_names=['Shale','BrineSand','GasSand'])
#
#cvr=rid_search.cv_results_
#
#confusion_1 = confusion_matrix(yt, yp)
#
#
#f, ax = plt.subplots()      
#hm = sns.heatmap(confusion_1, annot=True, cmap="coolwarm",fmt='.2f',
#                 linewidths=.05)
#plt.show()
# 
#yp=np.asarray(yp)
#yp.shape=[yp.size,1]
#yy=np.concatenate([yp,yp,yp],axis=1)
#plt.imshow(yy, extent=[0,200,0,1400],aspect=1)
#plt.axis('scaled')
#
##if  hasattr(rid_search.best_estimator_, "predict_proba"):
##    yp=rid_search.best_estimator_.predict_proba(Xt)
##else:
##    yp=rid_search.best_estimator_.decision_function(Xt)
##
##multiclass_ROC(Xt,yt,yp) 
#ylim=[(0.6,1)]   
#title = "Learning Curves " 
#plot_learning_curve(pipe2, title,X, y, ylim=ylim, cv=cv,
#                          n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
#plt.show()

#yp2=rid_search.best_estimator_.predict(Xt2)

rep2=classification_report(yt,yp,digits=5  ,target_names=['Shale','BrineSand','GasSand'])
 
yp2=np.asarray(yt)
yp2.shape=[yp2.size,1]
yy=np.concatenate([yp2,yp2,yp2],axis=1)
plt.imshow(yy, extent=[0,200,0,1400],aspect=1)
plt.axis('scaled')
plt.savefig('Well_pred')
#
#
x=np.linspace(0,1,40)
plt.plot(x,-np.log(1-x),label='y=0')
plt.plot(x,-np.log(x),label='y=1')
plt.xlabel('P1')
plt.legend(loc=9)

a=pd.DataFrame(X.VP.loc[X.Y==1]) 
a.columns=['Shale']
b=pd.DataFrame(X.VP.loc[    X.Y==2]) 
b.columns=['Brine Sand']
c=pd.DataFrame(X.VP.loc[X.Y==3])
c.reset_index(inplace=True,drop=True)
c.columns=['Gas Sand']
x=X[columns]
x=scale.fit_transform(x)
x=pd.DataFrame(x,columns=columns)
x=pd.concat([x,X.Y],axis=1)
fig = plt.figure()
fig.suptitle('VS  Density estimation')
ax = fig.add_subplot(111)
sns.kdeplot(X['VS'].loc[X.Y==1],kernel='gau',color='r',shade=True,legend=False)
sns.kdeplot(X['VS'].loc[X.Y==2],kernel='gau',color='b',shade=True,legend=False)
sns.kdeplot(X['VS'].loc[X.Y==3],kernel='gau',color='g',shade=True,legend=False)
   