# decision tree and splitting criteria
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import preprocessing
import graphviz
import csv
from sklearn.metrics import classification_report

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
#print(table)
with open("test.csv", "r") as testfile:
    csv_reader = csv.reader(testfile)
    rows = []
    for row in csv_reader:
        rows.append(row)

# concatenate and put everything into a 2D array in numpy format
testset = np.array(rows)
# tabulate and put into pretty table
test_table = pd.DataFrame(testset,
                     columns=['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est', 'WillWait'])

# seperate the features from the labels
x_train = dataset[:, 0:10]
y_train = dataset[:, 10]

x_test = testset[:, 0:10]
y_test = testset[:, 10]

# encode all the strings
encoder = preprocessing.LabelEncoder()

# find length of array
counter = 0
for sublist in dataset[0, :]:
    counter += 1

# encode each string one by one, apparently only takes 1D arrays
for i in range(counter - 1):
    x_train[:, i] = encoder.fit_transform(x_train[:, i])
    x_test[:, i] = encoder.fit_transform(x_test[:, i])
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

# sets entropy for the tree
DecisionTree = tree.DecisionTreeClassifier(criterion="entropy", random_state=0)

# builds actual tree
DecisionTree.fit(x_train, y_train)
tree.plot_tree(DecisionTree)

dot_data = tree.export_graphviz(DecisionTree, out_file=None,
                                feature_names=['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type',
                                               'Est'],
                                class_names=encoder.classes_,
                                filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("mytree1")

# predict the output based on test file
y_pred = DecisionTree.predict(x_test)
print("predicted: ", encoder.inverse_transform(y_pred))

print(classification_report(y_test, y_pred))
