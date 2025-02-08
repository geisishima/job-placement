# -*- coding: utf-8 -*-
"""
Document: Project Appraisal 

@author: Geisi Shima, 070180912
"""

#########################################
#                                       #
# Importing all the necessary libraries #
#                                       #
#########################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel


from scipy.stats import zscore
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB


# Importing the data and setting the serial number as the index

df = pd.read_csv("Placement_Data_Full_Class.csv",sep = ",")
df = df.set_index('sl_no')

a = "Test"
##########################
#####               ######
##### DATA CLEANING ######
#####               ######
##########################


# Checking to see if the data has any missing variables

missingno.matrix(df)

# There are some missing variables in the salaries. This is understandable since
# not all people got hired, and this is what we aim to study in this data.

df.isnull().sum()

# This is a very consistent dataset. Out of 215 students, 67 were unhired, meaning
# 31.16 % of the sample are unhired. Their salaries are the null ones. To keep
# salaries in the analysis, or use it later, we code the unhired people as 
# having zero income from a job.

df['salary'] = df['salary'].fillna(0)

df['salary'].isnull().sum() #Now we have made sure that we have no missing values

# Some of the columns we have are unnecessary, like who is governing
# the high school or university. I will remove those columns.
# I will keep the sort of specialization the people had in high school to see
# if it has any effects on prospective hiring.

df.drop(['ssc_b', 'hsc_b'], axis = 1,inplace=True)


#####################################
plt.figure(figsize = (15, 10))
plt.style.use('seaborn-white')
ax=plt.subplot(321)
plt.boxplot(df['ssc_p'])
ax.set_title('Secondary school percentage')
ax=plt.subplot(322)
plt.boxplot(df['hsc_p'])
ax.set_title('Higher Secondary school percentage')
ax=plt.subplot(323)
plt.boxplot(df['degree_p'])
ax.set_title('UG Degree percentage')
ax=plt.subplot(324)
plt.boxplot(df['etest_p'])
ax.set_title('Employability Test percentage')
ax=plt.subplot(325)
plt.boxplot(df['mba_p'])
ax.set_title('MBA Degree percentage')
ax=plt.subplot(326)
plt.boxplot(df['salary'])
ax.set_title('Salary')

# I saw in the dataset description and previous work from people that outliers
# were a large problem in this dataset. I plotted the most important numerical
# variables in box plots. The largest number of outliers was in higher secondary
# school percentage. I checked these data points, and there were people that were
# scoring too low or too high in high school. This could bias the high school
# coefficient and show more correlation than there should be. So I decided to
# remove the outliers using the 1.5*IQR rule.

# This rule basically says that subtracting 1.5 times IQR from the first quartile
# and adding 1.5*IQR to the third quartile. Any number outside this created range is
# an outlier. Let us see if this will remove them.

Q1 = df['hsc_p'].quantile(0.25)
Q3 = df['hsc_p'].quantile(0.75)
IQR = Q3 - Q1 

filter = (df['hsc_p'] >= Q1 - 1.5 * IQR) & (df['hsc_p'] <= Q3 + 1.5 *IQR)
df=df.loc[filter]

plt.figure(figsize = (7.5, 5))
plt.title('High School Score after Outlier Removal')
plt.boxplot(df['hsc_p'])

# We see that there are no more outliers.

# From the plots above there was also an outlier in the salary, he had a much
# higher salary than the others. His other indicators were okay however so I
# will keep the observation since we are not using the salary in the main
# analysis, rather we are using the placement.

################################
#########              #########
######### DESCRIPTIVES #########
#########              #########
################################

descriptives = df.describe()

plt.figure(figsize = (15, 7))
plt.style.use('seaborn-darkgrid')
plt.subplot(231)
sns.distplot(df['ssc_p'])
fig = plt.gcf()
fig.set_size_inches(10,10)

plt.subplot(232)
sns.distplot(df['hsc_p'])
fig = plt.gcf()
fig.set_size_inches(10,10)

plt.subplot(233)
sns.distplot(df['degree_p'])
fig = plt.gcf()
fig.set_size_inches(10,10)

plt.subplot(234)
sns.distplot(df['etest_p'])
fig = plt.gcf()
fig.set_size_inches(10,10)

plt.subplot(235)
sns.distplot(df['mba_p'])
fig = plt.gcf()
fig.set_size_inches(10,10)

plt.subplot(236)
sns.distplot(df['salary'])
fig = plt.gcf()
fig.set_size_inches(10,10)

#######################################
#########                     #########
######### DATA VISUALISATIONS #########
#########                     #########
#######################################


sns.pairplot(df.iloc[:,1:])

###############################

sns.pairplot(df,vars=['ssc_p','hsc_p','degree_p','mba_p','etest_p'],hue="status")

###############################

sns.boxplot(x='workex',y='degree_p', hue = 'status',data=df)

# People who were employed had on average a lower degree score.
# People had more or less the same level of degree score regardless of experience.

###############################

sns.boxplot(x='gender',y='degree_p', hue = 'status',data=df)

# Non placed females had almost the same degree score as placed males.
# Could this be evidence of discrimination? Let us see more with a countplot.

sns.countplot(x="gender", data=df,hue="status")

# It seems like our sample has disproportionately more men. We can still make some
# inferences. The ratio of placed/non-placed is very different, with men having
# a much higher rate of employability.

df[df.gender == 'M'].groupby(df.status)['degree_p'].mean()
df[df.gender == 'F'].groupby(df.status)['degree_p'].mean()

# Women score on average higher, both placed and non-placed. If no other
# differences are noted in other variables, this could be a case of discrimination.

###############################

plt.figure(figsize = (15, 7))
plt.style.use('seaborn-white')
plt.subplot(221)
sns.countplot(x="gender", data=df, palette="Set3")
fig = plt.gcf()
fig.set_size_inches(10,10)

plt.subplot(222)
sns.countplot(x="workex", data=df, palette="Set3")
fig = plt.gcf()
fig.set_size_inches(10,10)

plt.subplot(223)
sns.countplot(x="specialisation", data=df, palette="Set3")
fig = plt.gcf()
fig.set_size_inches(10,10)

plt.subplot(224)
sns.countplot(x="degree_t", data=df, palette="Set3")
fig = plt.gcf()
fig.set_size_inches(10,10)


####################################
########                   #########
######## MODEL PREPARATION #########
########                   #########
####################################


# Turn all the variables into discrete, categorical or continuous

binary_cols=['gender','workex','specialisation','status']

# Apply label encoder to each column with categorical data of two types

label_encoder = LabelEncoder()
for col in binary_cols:
    df[col] = label_encoder.fit_transform(df[col])
    
df.loc[ df['hsc_s'] == 'Commerce', 'hsc_s'] = 1
df.loc[ df['hsc_s'] == 'Science', 'hsc_s'] = 2
df.loc[ df['hsc_s'] == 'Arts', 'hsc_s'] = 3

df.loc[ df['degree_t'] == 'Sci&Tech', 'degree_t'] = 1
df.loc[ df['degree_t'] == 'Comm&Mgmt', 'degree_t'] = 2
df.loc[ df['degree_t'] == 'Others', 'degree_t'] = 3

# For the others I will use 1,2,3 as categories. The two columns left
# are high school specialisation and degree type. I did not use the label encoder as 
# I did not want to use zero for these columns


# Correlation matrix

corr = df.corr()
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
plt.figure(figsize=(13,7))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, 1)] = True
a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f', cmap="Blues")
rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)

################### Grid Selection for Random Forest ######################
df_copy = df.drop('salary', axis=1)

y=df_copy.pop('status')
x=df_copy

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)


param_grid_cv = { 
    'n_estimators': [1,10],
    'max_features': ['auto', None],
    'max_depth' : [1,2,3,None],
    'criterion' :['gini', 'entropy']}

rfr=RandomForestClassifier(random_state=42)

CV_rfr = GridSearchCV(estimator=rfr, param_grid=param_grid_cv)
CV_rfr.fit(x_train, y_train)
CV_rfr.best_params_

# This code shows us the best parameters to run Random Forest

rfr1=RandomForestClassifier(random_state=42, max_features="auto", n_estimators= 10, max_depth=None, criterion='gini')
rfr1.fit(x_train, y_train)
pred=rfr1.predict(x_test)
print("R2 for Random Forest on Placement data: ",rfr1.score(x_train,y_train))

corr = np.corrcoef(pred,y_test)[1,0]

####### Feature Importance #######

def feature_importance(x_train,regressor_name):
    Importance = pd.DataFrame({"Importance": regressor_name.feature_importances_*100},
                         index = x_train.columns)
    importance_values=Importance.sort_values(by = "Importance", 
                           axis = 0, ascending=False)
    Importance.sort_values(by = "Importance", 
                           axis = 0, 
                           ascending = True).plot(kind ="barh", color = "r")
    plt.xlabel("Feature Importances")
    importance_index=importance_values.index
    return importance_index

# These are the best features in terms of feature selection

results=feature_importance(x,rfr1)
x_train[results[0:2]]


def rfe_selection(x_train,y_train,regressor_name): 
    rfe = RFE(estimator=regressor_name, n_features_to_select=1, step=1)
    rfe.fit(x_train, y_train)
    ranking=pd.DataFrame({"Importance":rfe.ranking_}, index = x_train.columns)
    importance_results=ranking.sort_values(by = "Importance", 
                           axis = 0, ascending=True)
    return importance_results

rfe_selection(x_train,y_train,rfr1)

# When we run recursive selection these are the best features


######## Grid Selection and Feature Importance for Decision Tree Classifier #######


param_grid_cv1 = { 
    'splitter': ['best','random'],
    'max_features': ['auto', None],
    'max_depth' : [1,2,3,None],
    'criterion' :['gini', 'entropy']}


dt=DecisionTreeClassifier(random_state=42)

CV_dt = GridSearchCV(estimator=dt, param_grid=param_grid_cv1)
CV_dt.fit(x_train, y_train)
CV_dt.best_params_

# The best parameters to run Decision Tree Classifier

dt1=DecisionTreeClassifier(criterion = 'entropy',max_depth=3, splitter= 'random', max_features=None)
dt1.fit(x_train, y_train)

pred1=dt1.predict(x_test)
print("R2 for Decision Tree on Placement data: ",dt1.score(x_train,y_train))

corr1 = np.corrcoef(pred1,y_test)[1,0]


results1=feature_importance(x,dt1)

# Best features according to feature importance

rfe_selection(x_train,y_train,dt1)  

# Recursive selection of the best features

# Low correlation and R squared. I will assume Random Forest Classifier works better here


######## Grid Selection and Feature Importance for KNN #######

param_grid_cv2 = { 
    'weights': ['uniform','distance'],
    'n_neighbors': [1,10],
    'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'metric' :['euclidean', 'minkowski']}

knn=KNeighborsClassifier()

CV_knn = GridSearchCV(estimator=knn, param_grid=param_grid_cv2)
CV_knn.fit(x_train, y_train)
CV_knn.best_params_

knn1=KNeighborsClassifier(algorithm= 'auto', n_neighbors=10, weights= 'distance', metric = 'euclidean' )
knn1.fit(x_train, y_train)

pred2=knn1.predict(x_test)
print("R2 for KNN on Placement data: ",knn1.score(x_train,y_train))

corr2 = np.corrcoef(pred2,y_test)[1,0]

# Oddly in this case the R squared is 1 but the correlation coefficient is low.
# The KNN has no attribute for feature importance so we do not apply it here.

###############################################################################
###############################################################################
######################                                    #####################
######################          MODELLING                 #####################  
######################                                    #####################
###############################################################################
###############################################################################


## For comments please refer to the report ##

###### Decision Tree Classifier #######

# First we fit the model according to the best parameters
dt_model=DecisionTreeClassifier(criterion = 'entropy',max_depth=3, splitter= 'random', max_features=None)
dt_model.fit(x_train, y_train)

# Accuracy score for the model

dt_model.score(x_test , y_test)


# We predict some values and compare them to the true values of y to see how it works.

y_predict_dt = dt_model.predict(x_test)
y_predict_dt[:5]

y_test.head(5)

# The following code is for the confusion matrix

from sklearn.metrics import confusion_matrix

confusion_matrix1 = confusion_matrix(y_test,y_predict_dt)

import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
class_names = np.array(['Not Placed','Placed'])    

plt.figure()
plot_confusion_matrix(confusion_matrix1,classes=class_names,
                      title='Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(confusion_matrix1, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')


# The following code is for the classification report

from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict_dt))

# I follow the same procedure for each of the models below. All comments are in the
# report.


##### Decision tree classifier with the six most important features ##########

df_copy1 = df.drop(['salary', 'specialisation','etest_p', 'degree_t', 'hsc_s', 'gender'], axis=1)

y1=df_copy1.pop('status')
x1=df_copy1

x_train1, x_test1,y_train1, y_test1 = train_test_split(x1,y1,test_size = 0.2,random_state = 42)


dt_model2=DecisionTreeClassifier(criterion = 'entropy',max_depth=3, splitter= 'random', max_features=None)
dt_model2.fit(x_train1, y_train1)

dt_model2.score(x_test1 , y_test1)

y_predict_dt2 = dt_model2.predict(x_test1)
y_predict_dt2[:5]

y_test1.head(5)

# Made two mistakes

confusion_matrix2 = confusion_matrix(y_test1,y_predict_dt2)


plt.figure()
plot_confusion_matrix(confusion_matrix2,classes=class_names,
                      title='Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(confusion_matrix2, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')


print(classification_report(y_test1,y_predict_dt2))


##### Decision tree classifier with the six most important features (recursive) ##########

df_copy2 = df.drop(['salary','degree_p' ,'hsc_p', 'hsc_s', 'gender'], axis=1)

y3=df_copy2.pop('status')
x3=df_copy2

x_train3, x_test3,y_train3, y_test3 = train_test_split(x3,y3,test_size = 0.2,random_state = 42)


dt_model3=DecisionTreeClassifier(criterion = 'entropy',max_depth=3, splitter= 'random', max_features=None)
dt_model3.fit(x_train3, y_train3)

dt_model3.score(x_test3 , y_test3)

y_predict_dt3 = dt_model3.predict(x_test3)
y_predict_dt3[:5]

y_test3.head(5)


confusion_matrix3 = confusion_matrix(y_test3,y_predict_dt3)


plt.figure()
plot_confusion_matrix(confusion_matrix3,classes=class_names,
                      title='Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(confusion_matrix3, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')


print(classification_report(y_test3,y_predict_dt3))


###### Random Forest Classifier #######


rfr_model=RandomForestClassifier(random_state=42, max_features="auto", n_estimators= 10, max_depth=None, criterion='gini')
rfr_model.fit(x_train, y_train)

rfr_model.score(x_test , y_test)

y_predict_rfr = rfr_model.predict(x_test)
y_predict_rfr[:5]

y_test.head(5)

confusion_matrix_rfr = confusion_matrix(y_test,y_predict_rfr)

plt.figure()
plot_confusion_matrix(confusion_matrix_rfr,classes=class_names,
                      title='Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(confusion_matrix_rfr, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')


print(classification_report(y_test,y_predict_rfr))


############## Random Forest Classifier with six most important features (recursive) ###############

df_copy3 = df.drop(['salary','specialisation' ,'degree_t', 'workex', 'gender'], axis=1)

y4=df_copy3.pop('status')
x4=df_copy3

x_train4, x_test4,y_train4, y_test4 = train_test_split(x4,y4,test_size = 0.2,random_state = 42)


rfr_model2=RandomForestClassifier(random_state=42, max_features="auto", n_estimators= 10, max_depth=None, criterion='gini')
rfr_model2.fit(x_train4, y_train4)

rfr_model2.score(x_test4 , y_test4)

y_predict_rfr2 = rfr_model2.predict(x_test4)
y_predict_rfr2[:5]

y_test4.head(5)

confusion_matrix_rfr2 = confusion_matrix(y_test4,y_predict_rfr2)

plt.figure()
plot_confusion_matrix(confusion_matrix_rfr2,classes=class_names,
                      title='Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(confusion_matrix_rfr2, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')


print(classification_report(y_test4,y_predict_rfr2))

############## Random Forest Classifier with six most important features ###############

df_copy4 = df.drop(['salary','specialisation' ,'degree_t', 'hsc_s', 'gender'], axis=1)

y5=df_copy4.pop('status')
x5=df_copy4

x_train5, x_test5,y_train5, y_test5 = train_test_split(x5,y5,test_size = 0.2,random_state = 42)


rfr_model3=RandomForestClassifier(random_state=42, max_features="auto", n_estimators= 10, max_depth=None, criterion='gini')
rfr_model3.fit(x_train5, y_train5)

rfr_model3.score(x_test5 , y_test5)

y_predict_rfr3 = rfr_model3.predict(x_test5)
y_predict_rfr3[:5]

y_test5.head(5)

confusion_matrix_rfr3 = confusion_matrix(y_test5,y_predict_rfr3)

plt.figure()
plot_confusion_matrix(confusion_matrix_rfr3,classes=class_names,
                      title='Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(confusion_matrix_rfr3, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')


print(classification_report(y_test5,y_predict_rfr3))


############### KNN #################

knn_model=KNeighborsClassifier(algorithm= 'auto', n_neighbors=10, weights= 'distance', metric = 'euclidean' )
knn_model.fit(x_train, y_train)

#Accuracy Score of the KNN

knn_model.score(x_test , y_test)

y_predict_knn = knn_model.predict(x_test)
y_predict_knn[:5]

y_test.head(5)

confusion_matrix_knn = confusion_matrix(y_test,y_predict_knn)

plt.figure()
plot_confusion_matrix(confusion_matrix_knn,classes=class_names,
                      title='Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(confusion_matrix_knn, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')


print(classification_report(y_test,y_predict_knn))



############### Naive Bayes #######################

naive_model = GaussianNB()
naive_model.fit(x_train, y_train)

prediction_naive = naive_model.predict(x_test)
naive_model.score(x_test, y_test)

confusion_matrix_nb = confusion_matrix(y_test, prediction_naive)

plt.figure()
plot_confusion_matrix(confusion_matrix_nb,classes=class_names,
                      title='Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(confusion_matrix_nb, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

print(classification_report(y_test, prediction_naive))


################## Logistic Regression ###################

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logit_model = LogisticRegression()
logit_model.fit(x_train, y_train)


logit_model.score(x_test , y_test)

y_predict_logit = logit_model.predict(x_test)
y_predict_logit[:5]

y_test.head(5)

confusion_matrix_logit = confusion_matrix(y_test,y_predict_logit)

plt.figure()
plot_confusion_matrix(confusion_matrix_logit,classes=class_names,
                      title='Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(confusion_matrix_logit, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')


print(classification_report(y_test,y_predict_logit))

################################################################################
######################           Comparison          ###########################
################################################################################

# With the following code I compare the algorithms of the models in terms of their
# accuracy score. I use Kfold to split the data ten times.

df_c = df.drop('salary', axis=1)

y_c=df_c.pop('status')
x_c=df_c


models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('Logit', LogisticRegression()))


results = []
names = []
scoring = 'accuracy'

for name, model in models:
	kfold = model_selection.KFold(n_splits=10)
	cv_results = model_selection.cross_val_score(model, x_c, y_c, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


######################################################################## 
#####################                             ######################
#####################        ROC curve            ######################
#####################                             ######################
########################################################################

from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict  

# In random forest the full model is clearly performing the best

randomforest_pred = cross_val_predict(rfr_model, x_train, y_train, cv=5)

knears_pred = cross_val_predict(knn_model, x_train, y_train, cv=5) 

naive_pred = cross_val_predict(naive_model, x_train, y_train, cv=5) 

tree_pred = cross_val_predict(dt_model, x_train, y_train, cv=5) 

tree_pred1 = cross_val_predict(dt_model3, x_train, y_train, cv=5) 

logit_pred = cross_val_predict(logit_model, x_train, y_train, cv=5) 

# The ROC curve surface (score) for each of the models.

from sklearn.metrics import roc_auc_score

print('Random Forest: ', roc_auc_score(y_train, randomforest_pred))
print('KNears Neighbors: ', roc_auc_score(y_train, knears_pred))
print('Naive: ', roc_auc_score(y_train, naive_pred))
print('Decision Tree Classifier: ', roc_auc_score(y_train, tree_pred))
print('Decision Tree Recursive Selection Classifier: ', roc_auc_score(y_train, tree_pred1))
print('Logistic Regression Classifier: ', roc_auc_score(y_train, logit_pred))




rf_fpr, rf_tpr, rf_thresold = roc_curve(y_train, randomforest_pred)
knear_fpr, knear_tpr, knear_threshold = roc_curve(y_train, knears_pred)
naive_fpr, naive_tpr, naive_threshold = roc_curve(y_train, naive_pred)
tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, tree_pred)
tree_fpr1, tree_tpr1, tree_threshold1 = roc_curve(y_train, tree_pred1)
logit_fpr, logit_tpr, logit_threshold = roc_curve(y_train, logit_pred)




def graph_roc_curve_multiple(rf_fpr, rf_tpr, knear_fpr, knear_tpr, naive_fpr, naive_tpr, tree_fpr, tree_tpr, tree_fpr1, tree_tpr1,logit_fpr,logit_tpr):
    plt.figure(figsize=(16,8))
    plt.title('ROC Curve \n Top 6 Classifiers', fontsize=18)
    plt.plot(rf_fpr, rf_tpr, label='Random Forest Classifier Score: {:.4f}'.format(roc_auc_score(y_train, randomforest_pred)))
    plt.plot(knear_fpr, knear_tpr, label='KNears Neighbors Classifier Score: {:.4f}'.format(roc_auc_score(y_train, knears_pred)))
    plt.plot(naive_fpr, naive_tpr, label='Naive Bayes Classifier Score: {:.4f}'.format(roc_auc_score(y_train, naive_pred)))
    plt.plot(tree_fpr, tree_tpr, label='Decision Tree Classifier Score: {:.4f}'.format(roc_auc_score(y_train, tree_pred)))
    plt.plot(tree_fpr1, tree_tpr1, label='Decision Tree Recursive Classifier Score: {:.4f}'.format(roc_auc_score(y_train, tree_pred1)))
    plt.plot(logit_fpr, logit_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_train, logit_pred)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                )
    plt.legend()
    
graph_roc_curve_multiple(rf_fpr, rf_tpr, knear_fpr, knear_tpr, naive_fpr, naive_tpr, tree_fpr, tree_tpr, tree_fpr1, tree_tpr1,logit_fpr,logit_tpr)
plt.show()























