import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("iris_dataset.csv")
print(data.head())

# Task one
mean_sepal_length = data["sepal_length"].mean()
print(f"The mean of sepal_length {mean_sepal_length:.2f}")