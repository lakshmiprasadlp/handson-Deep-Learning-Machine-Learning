import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

calories = pd.read_csv('calories.csv')
exercise = pd.read_csv('exercise.csv')

df = exercise.merge(calories, on='User_ID')

# Bivariate and Multivariate Analysis

# Bar Plot (Numerical - Categorical)
# sns.barplot(df['Gender'], df['Calories'])

# Boxplot (numerical to categorical)
# sns.boxplot(df['Gender'], df['Age'])

# Distplot (Numerical - Categorical)
# sns.distplot(df[df['Gender']=='Male']['Age'])

# Lineplot (Numerical - Numerical)
# sns.lineplot(df['Age'], df['Calories'])

# Encoding
df['Gender'] = df['Gender'].map({'male': 1, 'female': 0})

# Train test split
X = df.drop(['User_ID', 'Calories'], axis=1)
y = df['Calories']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Model
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

models = {
    'lr': LinearRegression(),
    'rd': Ridge(),
    'ls': Lasso(),
    'dtr': DecisionTreeRegressor(),
    'rfr': RandomForestRegressor()
}

for name, mod in models.items():
    mod.fit(X_train, y_train)
    y_pred = mod.predict(X_test)

    print(f"{name}  MSE: {mean_squared_error(y_test, y_pred)}, Score: {r2_score(y_test, y_pred)}")

rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_test)

import pickle

pickle.dump(rfr, open('rfr.pkl', 'wb'))
X_train.to_csv('X_train.csv')