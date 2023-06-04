from sklearn import tree

def build_and_train_decision_tree(X, y):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    return clf

# decision tree and splitting criteria

#import numpy as np

#class Node:
    #def __init__(self, feature=None, value=None, label=None):
       # self.feature = feature
       # self.value = value
       # self.label = label
       # self.children = {}
        # not exactly sure if will do this



