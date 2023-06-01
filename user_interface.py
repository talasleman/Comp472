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
