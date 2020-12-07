import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import  confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
plt.rc('font', family='serif')


class Plots(): 
    def boxcar(data): 
        f, (ax) = plt.subplots(1, 1, figsize=(12, 4))
        f.suptitle('Boxcar', fontsize=14)
        sns.boxplot(x=data.columns[1], y=data.columns[0], data=data,  ax=ax)

    def Pairplot(X,y):
    
        pp = sns.pairplot(X, size=1.5, aspect=1.5,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True),hue=y)
        plt.show()
    def Corrmatrix(self,X):
                
        f, ax = plt.subplots(figsize=(10, 6))
        corr = X.corr()
        hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                         linewidths=.05)
        f.subplots_adjust(top=0.93)
        t= f.suptitle('Log Attributes Correlation Heatmap', fontsize=14)
                
        plt.show()
    
    def Frequencies(X):
        title = fig.suptitle("Frequencies", fontsize=14)
        fig.subplots_adjust(top=0.85, wspace=0.3)
        ax = fig.add_subplot(1,1, 1)
        ax.set_xlabel("")
        ax.set_ylabel("Frequency") 
        ax.tick_params(axis='both', which='major', labelsize=8.5)
        sns.kdeplot(X, color='steelblue', ax=ax,shade=True)
        plt.show()
    
    def Barplot(X):
        fig = plt.figure(figsize = (6, 4))
        fig.suptitle("", fontsize=14)
        sns.barplot(data=X,palette=sns.xkcd_palette(['windows blue']))

    def confusion(y,yp,n_classes):    
        confusion_1 = confusion_matrix(y, yp)
        confusion_1.dtype=float
        for i in range(0,n_classes,1):
            confusion_1[i,:]=confusion_1[i,:]/sum(confusion_1[i,:])
        confusion_1=pd.DataFrame(confusion_1,columns=['Shale','Brine Sands','Gas Sands'], index=['Shale','    Brine Sands','Gas Sands'] )
        sns.heatmap(confusion_1, annot=True, cmap="coolwarm",fmt='.2f',
                         linewidths=.05,cbar=False)
        
        plt.rc('xtick', labelsize=12) 
        plt.rc('ytick', labelsize=12) 
        plt.yticks( va="center")
        
    def  SCATP3D(A,B,C,label):
         fig = plt.figure(figsize=(8, 6))
         ax = fig.add_subplot(111, projection='3d')
         xs = x
         ys = x
         zs = x
         ax.plot(a, b,  c)            
         ax.set_xlabel(label[0])
         ax.set_ylabel(label[1])
         ax.set_zlabel(label[2])

