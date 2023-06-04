# decision tree and splitting criteria
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import preprocessing
import graphviz
import csv
import pydotplus
from io import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz



def build_and_train_decision_tree():
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
    encoder = preprocessing.OneHotEncoder()
    x = encoder.fit_transform(x)

    # encode y
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y)

    # sets entropy for the tree
    DecisionTree = tree.DecisionTreeClassifier(criterion="entropy")

    # builds actual tree
    DecisionTree.fit(x, y)
    '''
    # should plot but dont know where it is
    tree.plot_tree(DecisionTree)

    dot_data = tree.export_graphviz(DecisionTree, out_file=None,
                                feature_names=['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est'],
                                class_names=encoder.classes_,
                                filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    graph.write_png('decision_tree.png')
    Image(graph.create_png())
    '''
    return DecisionTree, encoder


def classify_instance(DecisionTree, encoder, instance):
    # Reshape and encode instance with the encoder
    instance = np.array(instance).reshape(1, -1)
    instance = encoder.transform(instance)
    # Convert sparse matrix to dense
    instance = instance.toarray()
    # Predict instance
    prediction = DecisionTree.predict(instance)
    # return 'yes' if WillWait == 1 else 'no'
    return 'Yes' if prediction == 1 else 'No'




