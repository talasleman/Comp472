import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from decision_tree import build_and_train_decision_tree, classify_instance
from PIL import ImageTk, Image
import matplotlib
from matplotlib import pyplot as plt

DecisionTree = None
encoder = None



def submit():
    global DecisionTree, encoder
    inputs = [e.get() for e in entries]
    
    result = classify_instance(DecisionTree, encoder, inputs)
    messagebox.showinfo("Prediction", "WillWait: " + result
                            )
    

#initialize GUI
def run_ui():
    global DecisionTree, encoder
    DecisionTree, encoder = build_and_train_decision_tree()
    root = tk.Tk()

    root.title("Decision Tree Classifier")
    root.geometry("500x500")
 
    
    labels = ['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est']
    
    global entries
    entries = []
    for i, label in enumerate(labels):
        tk.Label(root, text=label).grid(row=i)
        entry = tk.Entry(root)
        entry.grid(row=i, column=1)
        entries.append(entry)
  
    submit_button = tk.Button(root, text="Submit", command=submit)
    submit_button.grid(row=len(labels)+1, column=1)
    '''
    image = Image.open("decision_tree.png")
    photo = ImageTk.PhotoImage(image)
    label = tk.Label(root, image=photo)
    label.image = photo
    label.grid(row=len(labels)+2, column=0, columnspan=2)
    '''
    root.mainloop()
