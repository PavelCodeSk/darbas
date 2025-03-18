import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import ast
from sklearn.preprocessing import LabelEncoder


"""Uzkrauname dataframe"""
df = pd.read_csv("df_yes_no.csv")

encoder = LabelEncoder()

df['yes_no'] = encoder.fit_transform(df['yes_no'])

target_test = df['yes_no']

print(target_test.shape)

# print(df.head())

# print(target_test.head())



df_pixel = pd.read_csv("pixels_df.csv")

# print(df_pixel.head())


# df_mixed = df.sample(frac=1, random_state=42).reset_index(drop=True)
# one_hot_encoding = pd.get_dummies(df_mixed["yes_no"])

# ====================================================================
# column = ["yes_no"]

# df_one_hot_encoded = pd.get_dummies(df, columns=column, drop_first=True, dtype=int)
# col_name = df_one_hot_encoded.columns[-1]
# col_data = df_one_hot_encoded.pop(col_name)
# df_one_hot_encoded.insert(0, col_name, col_data)
# ====================================================================

# label_target = df_one_hot_encoded.pop("yes_no_yes",)


df_to_numpy = df_pixel.to_numpy()
# ====================================================================


# df_one_hot_encoded = df_one_hot_encoded.rename(columns={"yes_no_yes": "yes_no"})
# print(df_mixed.head())
# print(df_one_hot_encoded.columns)
# print(df_one_hot_encoded.head())
# print(df.tail())
# df["pixels"] = df['pixels'].apply(lambda x: np.array(ast.literal_eval(x)))


# df_to_numpy = df_one_hot_encoded.to_numpy()

# print(label_target.head())

x_train_val, x_test, y_train_val, y_test = train_test_split(df_to_numpy, target_test, test_size=0.2, random_state=42, shuffle=True)

x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2, random_state=42, shuffle=True)


# x_train = x_train.reshape(-1, 128, 128, 1)
# x_test = x_test.reshape(-1, 128, 128, 1)
# x_val = x_val.reshape(-1, 128, 128, 1)
# y_val = y_val.reshape(-1, 128, 128, 1)
# y_train = y_train.reshape(-1, 128, 128, 1)
# y_test = y_test.reshape(-1, 128, 128, 1)
print(f"y_train shape: {x_train.shape}")
print(f"y_test shape: {x_test.shape}")
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"x_val shape: {x_val.shape}")
print(f"y_val shape: {y_val.shape}")

# reshape = x_train.reshape(-1)
# to_flatten = x_train.flatten().tolist()
# print(to_flatten)

# to_list = to_flatten.to_list()


# plt.imshow(x_train[0])
# plt.title(y_train[0])

batch_size = 32
learning_rate = 0.0003
epochs = 20


"""Normalizacija"""
x_train_normalized = x_train / 255.0
x_test_normalized = x_test / 255.0
# y_train_normalized = y_train / 255.0
# y_test_normalized = y_test / 255.0
x_val_normalized = x_val / 255.0
y_val_normalized = y_val / 255.0

# 2. Modelio sukūrimas
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# 3. Modelio kompiliavimas
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

# 4. Modelio treniravimas
history = model.fit(x_train_normalized, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size)

# 5. Modelio vertinimas
val_preds = model.predict(x_test_normalized)
val_preds = (val_preds > 0.5).astype(int)
print(classification_report(y_test, val_preds, target_names=['Ne', 'Taip']))

# 6. Atvaizduojame mokymosi kreives
plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Modelio Tikslumo Kreivė')
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Modelio Nuostolio Kreivė')
plt.show()