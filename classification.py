# classification
from sklearn import tree

def classify_instance(clf, instance):
    return clf.predict(instance)




#def classify_instance(node, instance):
    # if leaf node, return label
    #if node.label is not None:
     #   return node.label

    # else, look up value of feature in instance
    #feature_value = instance[node.feature]

    # if value has branch in tree, follow branch
    #if feature_value in node.children:
     #   return classify_instance(node.children[feature_value], instance)

    # else, return 'Unknown'
    #else:
     #   return 'Unknown'
