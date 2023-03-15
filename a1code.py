import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier as GB
from sklearn.ensemble import AdaBoostClassifier as AB
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.svm import SVC


Dataset = pd.read_csv('Assignment_1_data.csv')
Dataset.iloc[:,0]=LabelEncoder().fit_transform(Dataset.iloc[:,0]) #represent gender as integer
X=Dataset.iloc[:,:-1]
Y=Dataset.iloc[:,-1]


#process missing values since some predictive model cannot train on nan
counts=Dataset.isna().sum()/len(Dataset)
clst=counts.index[counts.values>0.8].tolist()
        
x_m=X.drop(columns=clst) #feature selection1
for col in range(0,59-len(clst)):
    x_m.iloc[:,col]=x_m.iloc[:,col].fillna(x_m.iloc[:,col].median())
    
#SMOTE - process the imbalance data
oversample=SMOTE()
x,y=oversample.fit_resample(x_m,Y)


#feature selection2
fi=RFC(n_estimators=10,random_state=0).fit(x,y).feature_importances_
aq=pd.DataFrame()
aq['feature']=x.columns.values
aq['fi']=fi
aq.sort_values(by='fi')
l=list(aq.sort_values(by='fi',ascending=False)['feature'])
score=[]
for i in range(1,49):
    score.append(cross_val_score(RFC(n_estimators=10,random_state=0),x[l[0:i]],y,cv=5).mean())
plt.plot(range(1,49),score)
plt.show()
x_f=x.drop(columns=aq['feature'].iloc[0:33].tolist())

#sample for hyperparameter tuning
x_sample = x_f.sample(2000)
y_sample = y[x_sample.index]
y_sample.value_counts()


x_train, x_test, y_train, y_test = train_test_split(x_f,y, test_size=0.3)

#Random Forest Tree

modeltest=RFC(random_state=6)
modeltest.fit(x_train,y_train)
y_pred=modeltest.predict(x_test)
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test,y_pred))

param_test1 = {'n_estimators':range(10,201,10)}
gsearch1 = GridSearchCV(estimator = RFC(random_state=6), 
                       param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch1.fit(x_sample,y_sample)
gsearch1.best_params_, gsearch1.best_score_

param_test2 = {'max_depth':range(3,30,2)}
gsearch1 = GridSearchCV(estimator = RFC(n_estimators=190,random_state=6), 
                       param_grid = param_test2, scoring='roc_auc',cv=5)
gsearch1.fit(x_sample,y_sample)
gsearch1.best_params_, gsearch1.best_score_

param_test3 = {'min_samples_split':range(2,201,20), 'min_samples_leaf':range(1,60,10)}
gsearch1 = GridSearchCV(estimator = RFC(n_estimators=190,max_depth=21,random_state=6), 
                       param_grid = param_test3, scoring='roc_auc',cv=5)
gsearch1.fit(x_sample,y_sample)
gsearch1.best_params_, gsearch1.best_score_


modelRFC=RFC(n_estimators=190,max_depth=21,random_state=6)
modelRFC.fit(x_train,y_train)
y_pred=modelRFC.predict(x_test)
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test,y_pred))


#Decision Tree

modeltest=DT(random_state=6)
modeltest.fit(x_train,y_train)
y_pred=modeltest.predict(x_test)
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test,y_pred))


param_test1 = {'max_depth':range(1,100,2)}
gsearch2 = GridSearchCV(estimator = DT(random_state=6), 
                       param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch2.fit(x_sample,y_sample)
gsearch2.best_params_, gsearch2.best_score_

param_test2 = {'min_samples_split':range(2,201,20), 'min_samples_leaf':range(1,60,10)}
gsearch2 = GridSearchCV(estimator = DT(random_state=6), 
                       param_grid = param_test2, scoring='roc_auc',cv=5)
gsearch2.fit(x_sample,y_sample)
gsearch2.best_params_, gsearch2.best_score_

modelDT=DT(random_state=6)
modelDT.fit(x_train,y_train)
y_pred=modelDT.predict(x_test)
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test,y_pred))


#Logistic Regression

modeltest = LR(random_state=6)
modeltest.fit(x_train,y_train) 
y_pred=modeltest.predict(x_test)
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test,y_pred))


param_test1 = {'solver':('liblinear','sag','lbfgs','newton-cg')}
gsearch3 = GridSearchCV(estimator = LR(random_state=6), 
                       param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch3.fit(x_sample,y_sample)
gsearch3.best_params_, gsearch3.best_score_

param_test2 = {'C':(0.01,0.1,0.5,1,2,5,10,100,1000)}
gsearch3 = GridSearchCV(estimator = LR(random_state=6,solver='newton-cg'), 
                       param_grid = param_test2, scoring='roc_auc',cv=5)
gsearch3.fit(x_sample,y_sample)
gsearch3.best_params_, gsearch3.best_score_

param_test3 = {'max_iter':range(1,1001,10)}
gsearch3 = GridSearchCV(estimator = LR(random_state=6,solver='newton-cg',C=0.1), 
                       param_grid = param_test3, scoring='roc_auc',cv=5)
gsearch3.fit(x_sample,y_sample)
gsearch3.best_params_, gsearch3.best_score_

modelLR = LR(random_state=6,solver='newton-cg',C=0.1,max_iter=51)
modelLR.fit(x_train,y_train) 
y_pred=modelLR.predict(x_test)
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test,y_pred))






#SVC

#check if the dataset has linear relationship
from sklearn.linear_model import LinearRegression
regressor = LinearRegression().fit(x, y)
# Checking the accuracy
from sklearn.metrics import r2_score
print(r2_score(regressor.predict(x), y))

modeltest = SVC(random_state=6,kernel='sigmoid') #training dataset too large for other kernel, so only use sigmoid here 
modeltest.fit(x_train,y_train) 
y_pred=modeltest.predict(x_test)
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test,y_pred))


param_test1 = {'C':(0.01,0.1,0.5,1,2,5,10,100,1000)}
gsearch4 = GridSearchCV(estimator = SVC(random_state=6,kernel='sigmoid'), 
                       param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch4.fit(x_sample,y_sample)
gsearch4.best_params_, gsearch4.best_score_


param_test2 = {'gamma':(-5,-2,-1,-0.1,-0.01,-0.001,0.001,0.01,0.1,0.5,1,2,5,10,100,1000)}
gsearch4 = GridSearchCV(estimator = SVC(random_state=6,kernel='sigmoid',C=0.01), 
                       param_grid = param_test2, scoring='roc_auc',cv=5)
gsearch4.fit(x_sample,y_sample)
gsearch4.best_params_, gsearch4.best_score_

param_test3 = {'coef0':range(0,10,1)}
gsearch4 = GridSearchCV(estimator = SVC(random_state=6,kernel='sigmoid',C=0.01), 
                       param_grid = param_test3, scoring='roc_auc',cv=5)
gsearch4.fit(x_sample,y_sample)
gsearch4.best_params_, gsearch4.best_score_

modelSVC = SVC(random_state=6,kernel='sigmoid',C=0.01)
modelSVC.fit(x_train,y_train) 
y_pred=modelSVC.predict(x_test)
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test,y_pred))




#GradientBoosting


modeltest = GB(random_state=6)
modeltest.fit(x_train,y_train) 
y_pred=modeltest.predict(x_test)
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test,y_pred))


param_test1 = {'max_depth':range(1,25,2)}
gsearch5 = GridSearchCV(estimator = GB(random_state=6), 
                       param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch5.fit(x_sample,y_sample)
gsearch5.best_params_, gsearch5.best_score_

param_test2 = {'min_samples_split':range(2,201,20), 'min_samples_leaf':range(1,60,10)}
gsearch5 = GridSearchCV(estimator = GB(random_state=6,max_depth=5), 
                       param_grid = param_test2, scoring='roc_auc',cv=5)
gsearch5.fit(x_sample,y_sample)
gsearch5.best_params_, gsearch5.best_score_

param_test3 = {'learning_rate':(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)}
gsearch5 = GridSearchCV(estimator = GB(random_state=6,max_depth=5,min_samples_leaf=31, min_samples_split=182), 
                       param_grid = param_test3, scoring='roc_auc',cv=5)
gsearch5.fit(x_sample,y_sample)
gsearch5.best_params_, gsearch5.best_score_

modelGB = GB(random_state=6,max_depth=5,min_samples_leaf=31, min_samples_split=182)
modelGB.fit(x_train,y_train) 
y_pred=modelGB.predict(x_test)
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test,y_pred))



#AdaBoosting

modeltest = AB(random_state=6)
modeltest.fit(x_train,y_train) 
y_pred=modeltest.predict(x_test)
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test,y_pred))

param_test1 = {'n_estimators':range(10,201,10)}
gsearch6 = GridSearchCV(estimator = AB(random_state=6), 
                       param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch6.fit(x_sample,y_sample)
gsearch6.best_params_, gsearch6.best_score_

param_test2 = {'learning_rate':(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)}
gsearch6 = GridSearchCV(estimator = AB(random_state=6,n_estimators=90), 
                       param_grid = param_test2, scoring='roc_auc',cv=5)
gsearch6.fit(x_sample,y_sample)
gsearch6.best_params_, gsearch6.best_score_


modeltest = AB(random_state=6,n_estimators=90)
modeltest.fit(x_train,y_train) 
y_pred=modeltest.predict(x_test)
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test,y_pred))



