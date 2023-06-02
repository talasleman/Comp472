# decision tree and splitting criteria
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import preprocessing
import graphviz
import csv


with open("dataset.csv", "r") as csvfile:
    csv_reader = csv.reader(csvfile)
    rows = []
    for row in csv_reader:
        rows.append(row)

#dataset = np.array(", ".join(row))
        
        
dataset = np.array(rows)
table = pd.DataFrame(dataset,
                     columns=['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est', 'WillWait'])
print(table)

x = dataset[:, 0:10]
y = dataset[:, 10]

encoder = preprocessing.LabelEncoder()

counter = 0
for sublist in dataset[0, :]:
    counter += 1

for i in range(counter - 1):
    x[:, i] = encoder.fit_transform(x[:, i])

y = encoder.fit_transform(y)

DecisionTree = tree.DecisionTreeClassifier(criterion="entropy")

DecisionTree.fit(x, y)
tree.plot_tree(DecisionTree)

'''
dot_data = tree.export_graphviz(DecisionTree, out_file=None,
                                feature_names=['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est'],
                                class_names=encoder.classes_,
                                filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("mytree1")
'''