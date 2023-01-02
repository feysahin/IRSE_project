from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import tree
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix

import matplotlib.pyplot as plt


import pandas as pd
import numpy as np

VISITOR_DICT = {
    'Returning_Visitor': 0,
    'New_Visitor' : 1,
    'Other': 2
}
MONTH_DICT = {
    'May': 2,
    'Nov': 7,     
    'Mar': 1,     
    'Dec': 9,     
    'Oct': 3,
    'Sep': 8,    
    'Jul': 6,      
    'Aug': 5,      
    'June': 4,     
    'Feb': 0      
}

MODEL_DICT = {
    "NN": MLPClassifier,
    "DT": DecisionTreeClassifier,
    "RF": RandomForestClassifier, 
    "NB": GaussianNB,
    "SVM": SVC
}

numerical_columns = ['Administrative', 'Administrative_Duration', 'Informational',
       'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
       'BounceRates', 'ExitRates', 'PageValues']

categorical_columns = ['Month', 'VisitorType', 'TrafficType', 'Region', 'Browser', 'OperatingSystems']
binary_columns = ['Revenue', 'Weekend']

def transform_categorical_to_numerical(df):
  categorical_columns = df.select_dtypes(include='object').columns.tolist()
  boolean_columns = df.select_dtypes(include='bool').columns.tolist()
  cooumns = categorical_columns + boolean_columns
  # transform each column in the list
  for col in cooumns:
    # create a dictionary to map the unique values in the column to numeric labels
    mapping = {val:idx for idx, val in enumerate(df[col].unique())}
    df[col] = df[col].replace(mapping)
  return df


def boxplot_data(data_size, column):
  df = prepare_data(data_size)
  fig, ax = plt.subplots(figsize=(3, 3))
  ax.boxplot(df[column])
  ax.set_title(column)
  return fig


def clean_data(df, columns):
  #cleaning the data
  ini_length = len(df)
  print("Initial number of records:", len(df))
  columns = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 'PageValues']

  for column in columns:
    ini_length_1 = len(df)
    pst99 = df[column].quantile(0.99)
    df = df[df[column] <= pst99]
    print("Deleted observations with extream values in ", column, "\t deleted:", ini_length_1 - len(df))
  print("Total number of records deleted:", ini_length - len(df))
  return df


def prepare_data(ratio):
  df = pd.read_csv("online_shoppers_intention.csv")
  df = df.sample(frac = (ratio/100), random_state= 1, ignore_index = True)
  df = transform_categorical_to_numerical(df)
  columns_to_be_cleaned = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 'PageValues']
  df = clean_data(df, columns_to_be_cleaned)
  return df

def pie_chart(data_size):
    df = prepare_data(ratio = data_size)
    fig, ax = plt.subplots(figsize=(3, 3))
    df["Revenue"].value_counts().plot.pie(ax=ax, autopct='%1.1f%%')
    ax.set_title('Revenue')
    return fig

def histogram(df, columns):
  for column in columns:
    df.hist(column = column)

    

def train_model(model_name, ratio, train_size):
  df = prepare_data(ratio)
  X = df.copy()
  Y = X.pop("Revenue")
  # Split the data into training and testing sets
  x_train, x_test, y_train, y_test = train_test_split(
      X, Y, test_size= 1 - (train_size/100), random_state=42, shuffle=True
  )
  model = MODEL_DICT[model_name]()
  model = model.fit(x_train, y_train)
  predictions = model.predict(x_test)
  return predictions, y_test

def accuracy(predictions, y_test):
  return accuracy_score(y_test, predictions), precision_score(y_test, predictions), recall_score(y_test, predictions), f1_score(y_test, predictions)
