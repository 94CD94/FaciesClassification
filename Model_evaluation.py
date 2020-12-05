import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from Grids import Grid
from PLOT import Plots
from Transform import Transform
from export_dict import exportfromkeys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import model_selection 
from sklearn import feature_selection 
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.svm import LinearSVC

# load dataset
var = r'C:\Users\ridar\Desktop\Tesi_Carniani\X.txt'  
cols=['VP','VS','RHO','Y'] 
X=pd.read_csv(var,sep=';')

Log=X.loc[X.VP < 1900 ].copy()
X.drop(Log.loc[Log.Y==1].index,axis=0,inplace=True)
X=X[cols]
 
#sel=feature_selection.SelectKBest(k=20)
#X,Xt,y,yt = model_selection.train_test_split(X[['RHO','VP','VS']],X.Y, test_size=0.2,random_state=1, shuffle=True)        
Xt=pd.read_csv(r'C:\Users\ridar\Desktop\Tesi_Carniani\Avares.txt',sep=';')
Xt=Xt[cols]
Log=Xt.loc[Xt.VP < 1900 ].copy()
Xt.drop(Log.loc[Log.Y==1].index,axis=0,inplace=True)
Xt.RHO=Xt.RHO/1000
y=X.Y
yt=Xt.Y

T=Transform()
T.transform(X)     
T.transform(Xt)
#columns=['RHO', 'VP', 'Zs']
columns=['RHO','Zp','Zs']
X=X[columns]
Xt=Xt[columns]
n_classes=len(Counter(y))
scorer='balanced_accuracy'
# prepare models
cv =model_selection.StratifiedKFold(n_splits=5,random_state=1)
scale=StandardScaler()
sel=feature_selection.SelectKBest(k=20)
kb= KBinsDiscretizer(n_bins=30, strategy='uniform',encode='onehot-dense')
pca=PCA(svd_solver='full',random_state=1,n_components=4) 

estimators1 = [ ('scale',scale),('clf',KNeighborsClassifier())]
estimators2 = [ ('scale',scale),('clf',LogisticRegression(multi_class='ovr',solver='liblinear',random_state=1,max_iter=10000,fit_intercept=False,verbose=True,tol=10**-4))]
estimators3 = [ ('scale',scale),('clf',MLPClassifier(random_state=1,solver='lbfgs', hidden_layer_sizes=100))]
estimators4 = [ ('scale',scale),('clf',RandomForestClassifier(random_state=1))]
estimators5 = [ ('scale',scale),('clf',LinearSVC(random_state=1,dual=False,max_iter=10000,fit_intercept=False,tol=10**-4,verbose=True))]

pipe1  = imbPipeline(estimators1)
pipe2  = imbPipeline(estimators2)
pipe3  = imbPipeline(estimators3)
pipe4  = imbPipeline(estimators4)
pipe5  = imbPipeline(estimators5)

param_grid1 = dict(clf__n_neighbors=Grid.KNN()['n_neighbors'],clf__p=Grid.KNN()['p'],clf__weights=Grid.KNN()['weights'])
param_grid2 = dict(clf__C=Grid.LR()['C'],clf__penalty=Grid.LR()['penalty'])          
param_grid3 = dict(clf__alpha=Grid.NN()['alpha'],clf__activation=Grid.NN()['activation']) 
param_grid4 = dict(clf__n_estimators=Grid.RF()['n_estimators'],clf__max_depth=Grid.RF()['max_depth'],clf__min_samples_split=Grid.RF()['min_samples_split'],clf__min_samples_leaf=Grid.RF()['min_samples_leaf']) 
param_grid5 = dict(clf__C=Grid.SVM()['C'],clf__penalty=Grid.SVM()['penalty'])
   
grid1 = model_selection.GridSearchCV(pipe1, param_grid=param_grid1,cv=cv,n_jobs=-1,verbose=1,scoring='balanced_accuracy')
grid2 = model_selection.GridSearchCV(pipe2, param_grid=param_grid2,cv=cv,n_jobs=-1,verbose=1,scoring='balanced_accuracy')
grid3 = model_selection.GridSearchCV(pipe3, param_grid=param_grid3,cv=cv,n_jobs=-1,verbose=1,scoring='balanced_accuracy')
grid4 = model_selection.GridSearchCV(pipe4, param_grid=param_grid4,cv=cv,n_jobs=-1,verbose=1,scoring='balanced_accuracy')
grid5 = model_selection.GridSearchCV(pipe5, param_grid=param_grid5,cv=cv,n_jobs=-1,verbose=1,scoring='balanced_accuracy')

models = []
models.append(('KNN', grid1))
models.append(('LR', grid2))
models.append(('MLP', grid3))
models.append(('RF', grid4))
models.append(('SVM', grid5))

# evaluate each model in turn

Split_values=['split0_test_score','split1_test_score','split2_test_score','split3_test_score' ,'split4_test_score']
results = []
names = []
best_p=[]
results2=[]
scoring = [scorer]
Yp=[]
BestEstimator=[]
AveRank=[]

for name, model in models:
   model.fit(X,y) 
   best_p.append(name)
   best_p.append(model.best_params_)
   cv_results= exportfromkeys(model.cv_results_, Split_values, model.best_index_)
   Yp.append((name,model_selection.cross_val_predict(model.best_estimator_,X,y,cv=cv))) 
   BestEstimator.append((name,model.best_estimator_))
   results.append(cv_results)
   names.append(name)   
   msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
   print(msg)
   AveRank.append((cv_results.mean(),name))
   
# Confusion Matrix of cross validation  

for name, values in Yp :
    fig = plt.figure()
    fig.suptitle(name +' Confusion')
    ax = fig.add_subplot(111)
    Plots.confusion(y,values,n_classes)
    plt.savefig(name+" confusion.png")    
    plt.show()

# boxplot algorithm comparison   
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.ylabel('Balanced Accuracy')
plt.savefig("dispres.png")
plt.show()

with open("Log.txt", "w") as text_file:
    print(f"Best results: {best_p}", file=text_file)
with open("Results.txt", "w") as text_file:
    print(f"Best Scores: {AveRank}", file=text_file)

#BEST ALGORITHMS AND EVALUATION
AveRank.sort(reverse=True)
j=0
for name, pipe in BestEstimator: 
    if AveRank[0][1] is name:
        BestPipe=BestEstimator[j][1]
    j+=1

BestPipe.fit(X,y)
yp=BestPipe.predict(Xt)
classrep=metrics.classification_report(yt,yp,digits=5  ,target_names=['Shale','BrineSand','GasSand'])

with open("FinalReport.txt", "w") as text_file:
    print(f" {classrep}", file=text_file)


fig = plt.figure()
fig.suptitle( 'Test Confusion')
ax = fig.add_subplot(111)
Plots.confusion(yt,yp,n_classes) 
plt.savefig("Model_assessment_confusion.png")    
plt.show()

joblib_file = "Best_Model.pkl"          
joblib.dump(BestPipe, joblib_file)


yp2=np.asarray(yt)
yp2.shape=[yp2.size,1]
yy=np.concatenate([yp2,yp2,yp2],axis=1)
plt.imshow(yy, extent=[0,200,0,1400],aspect=1)
plt.axis('scaled')
plt.savefig('Well_pred')


#
#data=[]
#valori=range(5,55,5)
#for i in valori:
#    kb.n_bins=i
#    estimators3 =[ ('kb',kb),('clf',LogisticRegression(multi_class='ovr',C=10**10,max_iter=200000,solver='liblinear'))]
#    pipe3pca  = imbPipeline(estimators3)
#    data.append(model_selection.cross_validate(pipe3pca,X1,y1,scoring='balanced_accuracy',cv=cv))
#
#scoreslrb=[]
#timeslrb=[]
#
#for i in range(0,10):
#    timeslrb.append(exportfromkeys(data[i],['fit_time'],[0,1,2,3,4]).mean())
#    scoreslrb.append(exportfromkeys(data[i],['test_score'],[0,1,2,3,4]).mean())

#fig = plt.figure( figsize=(8, 6) )
#fig.suptitle( 'BINNING AND SCORES')
#i=[int(x) for x in valori] 
#plt.plot(i,scoressvmb,'-o')
#plt.plot(i,scoreslrb,'-o')
#plt.legend(['SVC','Logistic Regression'], loc=4     )
#plt.plot(i,times,'-o')
#plt.plot(i,timeskn,'-o')
#plt.plot(i,timesrf,'-o')
#plt.legend(['SVC','Logistic Regression','Neural Network','K Nearest','Random Forest'], loc=4)
#plt.ylabel('TEMPI MEDI DI FITTING')

#plt.plot(pca.explained_variance_ratio_,'-o')
#plt.ylabel('VARIANZA NORMALIZZATA')

