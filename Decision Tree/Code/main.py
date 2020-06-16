import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from Tree import DTree

df = pd.read_csv("cardio_train.csv",delimiter=";")

df = df.drop("id",axis=1)[:10000]

print(df.head())

splitter = StratifiedShuffleSplit(n_splits=2,test_size=0.1, random_state=42)


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy * 100


decision_tree = DTree(max_tree_depth=5)
average = []

for train_index, test_index in splitter.split(df, df["cardio"]):

    train_set = df.loc[train_index]
    test_set = df.loc[test_index]

    x = train_set.drop("cardio", axis=1)
    y = train_set["cardio"]

    y_test = test_set["cardio"]
    x_test = test_set.drop("cardio", axis=1)

    decision_tree.build_tree(x,y)

    y_prediction = decision_tree.prediction(x_test)

    accuracy_percent = accuracy(y_test, y_prediction)
    average.append(accuracy_percent)
    print(accuracy_percent)

print(sum(average) / len(average), "%")



