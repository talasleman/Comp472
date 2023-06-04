import tkinter as tk
from tkinter import filedialog
import pandas as pd

def browse_files():
    #open a file dialog to choose data set from
    #add csv filter
    filename = filedialog.askopenfilename(initialdir="/",
                                          filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")),
                                          title="Choose a file.")
    #check if file is empty with panda
    try:
        if filename:
            df = pd.read_csv(filename)
            print("File Loaded Successfully")
    except Exception as e:
        print("Error: ", e)
        print("Invalid file. Please choose a valid CSV file.")
        return

#initialize GUI
def run_ui():
    root = tk.Tk()

    root.title("Decision Tree Classifier")
    root.geometry("500x200")
    #add button with command
    browse_button = tk.Button(root, text="Browse files", command=browse_files)
    #display button
    browse_button.pack()

    root.mainloop()

'''
import tkinter as tk
from tkinter import filedialog
import pandas as pd

from entropy import entropy
from decision_tree import build_and_train_decision_tree
from classification import classify_instance


def browse_files(clf):
    filename = filedialog.askopenfilename(initialdir="/",
                                          filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")),
                                          title="Choose a file.")
    try:
        if filename:
            df = pd.read_csv(filename)
            print("File Loaded Successfully")
            instance = df[0]
            prediction = predict(clf, instance)
    except Exception as e:
        print("Error: ", e)
        print("Invalid file. Please choose a valid CSV file.")
        return

def predict(clf, instance):
    prediction = classify_instance(clf, instance)
    print(f"The predicted output for the input data is {prediction}")

def print_tree(tree, indent=""):
    # Assuming the tree is a nested dictionary
    for feature, branches in tree.items():
        if isinstance(branches, dict):
            # This is a subtree, print the feature and recurse
            print(f"{indent}{feature}?")
            print_tree(branches, indent + "  ")
        else:
            # This is a leaf node, print the outcome
            print(f"{indent}{branches}")

def run_ui():
    root = tk.Tk()

    root.title("Decision Tree Classifier")
    root.geometry("500x200")

    # Load training data and build the decision tree
    training_data = pd.read_csv('dataset.csv')

    df = pd.read_csv('dataset.csv')
    X = df.drop('WillWait', axis=1)
    y = df['WillWait']

    tree = build_and_train_decision_tree(X, y)
    print_tree(tree)

    instruction_label = tk.Label(root, text="Please browse for a file to predict:")
    instruction_label.grid(row=0, column=0, sticky=tk.W, pady=4)

    # Browse for new data
    browse_button = tk.Button(root, text="Browse files", command=lambda: browse_files())
    browse_button.grid(row=1, column=0, sticky=tk.W, pady=4)

    # Predict button
    predict_button = tk.Button(root, text="Predict", command=lambda: predict(tree, browse_files()))
    predict_button.grid(row=2, column=0, sticky=tk.W, pady=4)

    root.mainloop()
'''
