# decision tree and splitting criteria
import numpy as np
import pandas as pd
import sklearn as sk
import csv

'''
lists = deque()
with open("dataset.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        lists.append(row)
        #dataset = np.array(", ".join(row))

dataset = np.array([])
for entry in lists:
    np.concatenate(dataset, entry)

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
data = np.array([
    ['sunny', 85, 85, 0, 'Don\'t Play'],
    ['sunny', 80, 90, 1, 'Don\'t Play'],
    ['overcast', 83, 78, 0, 'Play'],
    ['rain', 70, 96, 0, 'Play'],
    ['rain', 68, 80, 0, 'Play'],
    ['rain', 65, 70, 1, 'Don\'t Play'],
    ['overcast', 64, 65, 1, 'Play'],
    ['sunny', 72, 95, 0, 'Don\'t Play'],
    ['sunny', 69, 70, 0, 'Play'],
    ['rain', 75, 80, 0, 'Play'],
    ['sunny', 75, 70, 1, 'Play'],
    ['overcast', 72, 90, 1, 'Play'],
    ['overcast', 81, 75, 0, 'Play'],
    ['rain', 71, 80, 1, 'Don\'t Play'],
])

'''
table = pd.DataFrame(dataset,
                     columns=['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est', 'WillWait'])
blankIndex = [''] * len(table)
table.index = blankIndex
print(table)

table = pd.DataFrame(data,
                     columns=['Outlook', 'Temperature', 'Humidity', 'Windy', 'Play / Don’t Play'])
blankIndex = [''] * len(table)
table.index = blankIndex
print(table)
'''
print(dataset)
