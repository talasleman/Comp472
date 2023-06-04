# decision tree and splitting criteria
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import preprocessing
import graphviz
import csv

# import from .csv file
with open("dataset.csv", "r") as csvfile:
    csv_reader = csv.reader(csvfile)
    rows = []
    for row in csv_reader:
        rows.append(row)

# concaterate and put everything into a 2D array in numpy format
dataset = np.array(rows)
# tabulate and put into pretty table
table = pd.DataFrame(dataset,
                     columns=['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est', 'WillWait'])
print(table)

# seperate the features from the labels
x = dataset[:, 0:10]
y = dataset[:, 10]

# encode all the strings
encoder = preprocessing.LabelEncoder()

# find length of array
counter = 0
for sublist in dataset[0, :]:
    counter += 1

# encode each string one by one, apparently only takes 1D arrays
for i in range(counter - 1):
    x[:, i] = encoder.fit_transform(x[:, i])
y = encoder.fit_transform(y)

# sets entropy for the tree
DecisionTree = tree.DecisionTreeClassifier(criterion="entropy")

# builds actual tree
DecisionTree.fit(x, y)
# should plot but dont know where it is
tree.plot_tree(DecisionTree)


dot_data = tree.export_graphviz(DecisionTree, out_file=None,
                                feature_names=['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est'],
                                class_names=encoder.classes_,
                                filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("mytree1")
