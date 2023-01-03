from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.utils import resample
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix, roc_curve, roc_auc_score

import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
import itertools

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


def resample_df(df):
  revenue_data = df[df['Revenue']==1]
  non_revenue_data = df[df['Revenue']==0]
  revenue_df = resample(revenue_data,
             replace=True,
             n_samples=len(non_revenue_data),
             random_state=42)
  data_upsampled = pd.concat([revenue_df, non_revenue_data])
  data_upsampled = data_upsampled.sample(frac = 1, random_state= 1, ignore_index = True)
  return data_upsampled

def boxplot_data(data_size, column, preprocess):
  if preprocess == "Yes":
      df = prepare_data(ratio = data_size)
  else:
      df = pd.read_csv("online_shoppers_intention.csv")
      df = df.sample(frac = (data_size/100), random_state= 1, ignore_index = True)
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

def select_features(df, k):
  X = df.copy()
  Y = X.pop("Revenue")
  # feature extraction
  selector = SelectKBest(score_func=chi2, k=k)
  fit = selector.fit(X, Y)
  # summarize scores
  np.set_printoptions(precision=3)
  print(fit.scores_)
  print("Slected columns: ", X.columns[selector.get_support()])
  features = fit.transform(X)
  # summarize selected features
  return features

def prepare_data(ratio):
  df = pd.read_csv("online_shoppers_intention.csv")
  df = df.sample(frac = (ratio/100), random_state= 1, ignore_index = True)
  df = transform_categorical_to_numerical(df)
  columns_to_be_cleaned = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 'PageValues']
  df = clean_data(df, columns_to_be_cleaned)
  df = resample_df(df)
  return df

def pie_chart(data_size, preprocess):
    if preprocess == "Yes":
        df = prepare_data(ratio = data_size)
    else:
        df = pd.read_csv("online_shoppers_intention.csv")
        df = df.sample(frac = (data_size/100), random_state= 1, ignore_index = True)
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
  X = select_features(df, 10)
  # Split the data into training and testing sets
  x_train, x_test, y_train, y_test = train_test_split(
      X, Y, test_size= 1 - (train_size/100), random_state=42, shuffle=True
  )
  if model_name != "NB":
    model = MODEL_DICT[model_name](random_state=42)
  else:
    model = MODEL_DICT[model_name]()
  model = model.fit(x_train, y_train)
  return model, X, x_test, Y, y_test

def compute_accuracy(model, x, y):
    predictions = model.predict(x)
    accuracy, precision, recall,f1=  accuracy_score(y, predictions), precision_score(y, predictions), recall_score(y, predictions), f1_score(y, predictions)
    return round(accuracy*100,2), round(precision*100,2), round(recall*100,2), round(f1*100,2)

def compute_confusion_matrix(model, x, y):
    predictions = model.predict(x)
    cm = confusion_matrix(y, predictions)
    return cm


def plot_confusion_matrix(model, x, y, classes):
    cm = compute_confusion_matrix(model, x, y)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title='Confusion matrix',
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


