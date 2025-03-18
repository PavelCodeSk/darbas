import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

"""Uzkrauname dataframe"""
df = pd.read_csv("df_yes_no.csv")

df_mixed = df.sample(frac=1, random_state=42).reset_index(drop=True)
# one_hot_encoding = pd.get_dummies(df_mixed["yes_no"])
column = ["yes_no"]

df_one_hot_encoded = pd.get_dummies(df_mixed, columns=column, drop_first=True, dtype=int)
col_name = df_one_hot_encoded.columns[-1]
col_data = df_one_hot_encoded.pop(col_name)
df_one_hot_encoded.insert(0, col_name, col_data)

# df_one_hot_encoded = df_one_hot_encoded.rename(columns={"yes_no_yes": "yes_no"})
# print(df_mixed.head())
print(df_one_hot_encoded.columns)
print(df_one_hot_encoded.head())
# print(df.tail())

label_target = df_one_hot_encoded.pop("yes_no_yes",)

# print(label_target.head())

x_train,x_test,y_train,y_test = train_test_split(df_one_hot_encoded, label_target, test_size=0.2, random_state=42)

"""Normalizacija"""
x_train_normalized = x_train / 255.0
x_test_normalized = x_test / 255.0

"""Sukuriame modeli"""

rfc = RandomForestClassifier(n_estimators=30, random_state=42)

rfc.fit(x_train_normalized, y_train)
y_pred = rfc.predict(x_test_normalized)

accuracy = accuracy_score(y_test, y_pred)

classification_reports = classification_report(y_test, y_pred)

print("Accuracy: ", accuracy)
print("Classification", classification_reports)