import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split ,StratifiedShuffleSplit
import numpy as np
from sklearn.preprocessing import LabelEncoder
import brStarter as brs
import operator
import sklearn

flowers = pd.read_csv("iris.csv")

print("Dataset Sample\n")
print(flowers.head())
print("\nDataset Info\n")
print(flowers.info())
print("\nClasses count \n")
categories_count = flowers["species"].value_counts().sort_index()
print(categories_count)
print("\n Features description \n")
print(flowers.describe())

print("---------------------------------------------------------")

# scatter plots

c = ['g' for i in range(0,150)]
c[50:100] = ['r' for i in range(50)]
c[100:150] = ['b' for i in range(50)]

label = [k for k in categories_count.keys()]

ax = flowers[0:50].plot.scatter(x="sepal_length", y="sepal_width", c=c[0:50], label=label[0])
ax2 = flowers[50:100].plot.scatter(x="sepal_length", y="sepal_width", c=c[50:100], label=label[1], ax=ax)
flowers[100:150].plot.scatter(x="sepal_length", y="sepal_width", c=c[100:150], label=label[2], ax=ax2,alpha=0.5)
# plt.savefig("s-length-width.png")
plt.show()

ax3 = flowers[0:50].plot.scatter(x="petal_length", y="petal_width",c=c[0:50], label=label[0])
ax4 = flowers[50:100].plot.scatter(x="petal_length", y="petal_width",c=c[50:100], label=label[1], ax=ax3)
flowers[100:150].plot.scatter(x="petal_length", y="petal_width", c=c[100:150], label=label[2] , ax=ax4,alpha=0.5)
# plt.savefig("p-length-width.png")
plt.show()


# show histograms
flowers.hist()
plt.show()


# see correlation matrix of attributes
corr_matrix = flowers.corr()
print(corr_matrix)

# plot correlation
pd.plotting.scatter_matrix(flowers[flowers.columns], figsize=(12, 8),alpha=0.5,c=c)
# plt.savefig("all")
plt.show()

# encode species into numbers

flowers["species"] = LabelEncoder().fit_transform(flowers["species"])

# split the set into train - test using stratified shuffle split

splitter = StratifiedShuffleSplit(n_splits=5,test_size=0.2, random_state=42)

percentages = []
number_of_wrong_classifications = []
print(flowers)

for train_index, test_index in splitter.split(flowers, flowers["species"]):

    train_set = flowers.loc[train_index]
    test_set = flowers.loc[test_index]

    training_set = train_set.copy()

    accuracy, _ = brs.naive_bayes_classifier(training_set, test_set)

    number_of_wrong_classifications.append(_)

    percentages.append(accuracy)

print("wrong classifications", number_of_wrong_classifications)

max_number_of_error = max(number_of_wrong_classifications)
indices = [i+1 for i, x in enumerate(number_of_wrong_classifications) if x == max_number_of_error]

print("Max wrong classifications is ", max_number_of_error , "at split # ", indices)
print("total wrong classifications is ", sum(number_of_wrong_classifications))

print("Splits accuracy", percentages)
print("Average accuracy", brs.mean(percentages), "%")






