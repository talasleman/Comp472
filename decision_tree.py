# decision tree and splitting criteria
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import preprocessing
import graphviz
import csv

'''
with open("dataset.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        #dataset = np.array(", ".join(row))
        
        
dataset = np.array([])
table = pd.DataFrame(dataset,
                     columns=['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est', 'WillWait'])
table.index = ['']*len(table)
print(table.index)
'''

dataset = np.array([
    ['Yes', 'No', 'No', 'Yes', 'Some', '$$$', 'No', 'Yes', 'French', '0-10', 'Yes'],
    ['Yes', 'No', 'No', 'Yes', 'Full', '$', 'No', 'No', 'Thai', '30-60', 'No'],
    ['No', 'Yes', 'No', 'No', 'Some', '$', 'No', 'No', 'Burger', '0-10', 'Yes'],
    ['Yes', 'No', 'Yes', 'Yes', 'Full', '$', 'Yes', 'No', 'Thai', '10-30', 'Yes'],
    ['Yes', 'No', 'Yes', 'No', 'Full', '$$$', 'No', 'Yes', 'French', '>60', 'No'],
    ['No', 'Yes', 'No', 'Yes', 'Some', '$$', 'Yes', 'Yes', 'Italian', '0-10', 'Yes'],
    ['No', 'Yes', 'No', 'No', 'None', '$', 'Yes', 'No', 'Burger', '0-10', 'No'],
    ['No', 'No', 'No', 'Yes', 'Some', '$$', 'Yes', 'Yes', 'Thai', '0-10', 'Yes'],
    ['No', 'Yes', 'Yes', 'No', 'Full', '$', 'Yes', 'No', 'Burger', '>60', 'No'],
    ['Yes', 'Yes', 'Yes', 'Yes', 'Full', '$$$', 'No', 'Yes', 'Italian', '10-30', 'No'],
    ['No', 'No', 'No', 'No', 'None', '$', 'No', 'No', 'Thai', '0-10', 'No'],
    ['Yes', 'Yes', 'Yes', 'Yes', 'Full', '$', 'No', 'No', 'Burger', '30-60', 'Yes'],
])

table = pd.DataFrame(dataset,
                     columns=['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est', 'WillWait'])

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