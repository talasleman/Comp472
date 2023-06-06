# decision tree and splitting criteria
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import preprocessing
import graphviz
import csv
from io import StringIO
from IPython.display import Image
from sklearn.metrics import classification_report
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
    # print(table)
    with open("test.csv", "r") as testfile:
        csv_reader = csv.reader(testfile)
        rows = []
        for row in csv_reader:
            rows.append(row)

    # concatenate and put everything into a 2D array in numpy format
    testset = np.array(rows)
    # tabulate and put into pretty table
    test_table = pd.DataFrame(testset,
                              columns=['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est',
                                       'WillWait'])

    # seperate the features from the labels
    x_train = dataset[:, 0:10]
    y_train = dataset[:, 10]

    x_test = testset[:, 0:10]
    y_test = testset[:, 10]

    # encode all the strings
    encoder = preprocessing.OneHotEncoder()
    x_train = encoder.fit_transform(x_train)

    feature_names = encoder.get_feature_names_out(
        ['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est'])
    # encode y
    label_encoder = preprocessing.LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)

    # sets entropy for the tree
    DecisionTree = tree.DecisionTreeClassifier(criterion="entropy", random_state=0)

    # builds actual tree
    DecisionTree.fit(x_train, y_train)

    # should plot
    tree.plot_tree(DecisionTree)

    dot_data = tree.export_graphviz(DecisionTree, out_file=None,
                                    feature_names=feature_names,
                                    class_names=label_encoder.classes_,
                                    filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    # create png instead
    png_file_path = 'decision_tree'
    graph.format = 'png'
    graph.render(filename=png_file_path, cleanup=True)

    x_test = encoder.transform(x_test)
    y_test = label_encoder.transform(y_test)

    # predict the output based on test file
    y_pred = DecisionTree.predict(x_test)
    print("predicted: ", label_encoder.inverse_transform(y_pred))
    print(classification_report(y_test, y_pred))

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