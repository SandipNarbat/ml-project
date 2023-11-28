import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Function to predict using the trained model
def predict_banknote():
    new_banknote = [float(var_entry.get()), float(skew_entry.get()), float(curt_entry.get()), float(entr_entry.get())]
    new_banknote = scalar.transform([new_banknote])
    prediction = clf.predict(new_banknote)[0]
    probability = clf.predict_proba(new_banknote)[0]
    
    result_label.config(text=f'Prediction: Class {prediction}')
    probability_label.config(text=f'Probability [0/1]: {probability}')

# Load and preprocess the dataset
data = pd.read_csv('data_banknote_authentication.txt', header=None)
data.columns = ['var', 'skew', 'curt', 'entr', 'auth']

sns.set()
sns.set_palette("husl")

# Split the data into training and testing sets
x = data.loc[:, data.columns != 'auth']
y = data.loc[:, data.columns == 'auth']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

scalar = StandardScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)

# Train the logistic regression model
clf = LogisticRegression(solver='lbfgs', random_state=42, multi_class='auto')
clf.fit(x_train, y_train.values.ravel())

# Create a GUI window
window = tk.Tk()
window.title("Banknote Authentication Prediction")

# Labels and Entry Fields for user input
var_label = ttk.Label(window, text="Variance")
var_entry = ttk.Entry(window)
var_label.pack()
var_entry.pack()

skew_label = ttk.Label(window, text="Skewness")
skew_entry = ttk.Entry(window)
skew_label.pack()
skew_entry.pack()

curt_label = ttk.Label(window, text="Curtosis")
curt_entry = ttk.Entry(window)
curt_label.pack()
curt_entry.pack()

entr_label = ttk.Label(window, text="Entropy")
entr_entry = ttk.Entry(window)
entr_label.pack()
entr_entry.pack()

# Button to predict the banknote class
predict_button = ttk.Button(window, text="Predict", command=predict_banknote)
predict_button.pack()

# Labels to display the prediction result
result_label = ttk.Label(window, text="")
probability_label = ttk.Label(window, text="")
result_label.pack()
probability_label.pack()

window.mainloop()
