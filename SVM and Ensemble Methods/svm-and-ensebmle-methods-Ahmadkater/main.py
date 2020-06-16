import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import tree
from sklearn import naive_bayes
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing , multiclass
from sklearn import ensemble
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("mnist.csv")


splitter = StratifiedShuffleSplit(n_splits=1, test_size=1/7, random_state=42)
data_set = None

for subset1_index, subset2_index in splitter.split(data,data["label"]):

    data_set = data.iloc[subset2_index].reset_index().drop("index",axis=1)


std = data_set[data_set.columns].std()
cols_to_drop = std[std == 0].index
new_data = data_set.drop(cols_to_drop, axis=1)

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)

train = None
test = None

scale = preprocessing.MinMaxScaler()
columns = new_data.columns

new_data[columns[1:]] = scale.fit_transform(new_data[columns[1:]])


for train_index , tes_index in splitter.split(new_data,new_data["label"]):

    train = new_data.iloc[train_index]
    test = new_data.iloc[tes_index]

y_train = train["label"]
x_train = train.drop("label",axis=1)

y_test = test["label"]
x_test = test.drop("label",axis=1)

nv1 = naive_bayes.GaussianNB()
nv2 = naive_bayes.MultinomialNB()
# multi_nv = multiclass.OneVsRestClassifier(nv1)

dt1 = tree.DecisionTreeClassifier()
# multi_dt = multiclass.OneVsRestClassifier(dt1)

# rf1 = ensemble.RandomForestClassifier(n_estimators= 500 , criterion='entropy')

svm1 = svm.LinearSVC(C=1, random_state=42, max_iter=10000)
# svm2 = svm.SVC(decision_function_shape='ovr')

v1 = ensemble.VotingClassifier(
estimators=[('nv', nv1), ('dt', dt1), ('svm',svm1)],
voting='hard'
)

bagging = ensemble.BaggingClassifier(
v1,
max_samples=0.3,
bootstrap=True
)

pasting = ensemble.BaggingClassifier(
v1,
max_samples=0.3,
bootstrap=False
)

clfs = ["Gaussian Naive Bayes", "Decision Tree ", "Support Vector Machine1 ", "Voting Classifier ", "Bagging ", "Pasting"]
clf = [nv1, dt1, svm1, v1, bagging, pasting]

for i in range(len(clf)):

    clf[i].fit(x_train,y_train)

    clf[i].predict(x_test)

    y_predict = clf[i].predict(x_test)

    score = clf[i].score(x_test, y_test) * 100

    print(clfs[i])

    print(score, ' %')


