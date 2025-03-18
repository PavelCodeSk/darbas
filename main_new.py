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
from utilities.utilities import parse_pixels, create_df_image_yes_no, train_datagen
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.callbacks import ReduceLROnPlateau

"""Sukuriam kintajami kuriame nurodysim kelia i nuotraukas"""
images_folder_malignant_melanoma = "dataset/new_dt_malignant/*.*"
images_folder_belign_melanoma = "dataset/new_dt_belign/*.*"
images_folder_malignant_melanoma_test = "dataset/new_dt_malignant_test/*.*"
images_folder_belign_melanoma_test = "dataset/new_dt_belign_test/*.*"


# """Uzkrauname dataframe"""
# df1 = pd.read_csv("df_yes_no.csv")
df = create_df_image_yes_no(images_folder_malignant_melanoma, images_folder_belign_melanoma, "yes", "no", "yes_no", no_color=False, add_filtr=False, lache=True)
# df_test = create_df_image_yes_no(images_folder_malignant_melanoma_test, images_folder_belign_melanoma_test, "yes", "no", "yes_no", no_color=False, add_filtr=False, lache=True)
df_val = create_df_image_yes_no(images_folder_malignant_melanoma_test, images_folder_belign_melanoma_test, "yes", "no", "yes_no", no_color=False, add_filtr=False, lache=True)

print(df.head())
print(df.tail())
print("="*100)
print(df_val.head())
print(df_val.tail())
# print(df1.head())

# ===========================================================

# encoder = LabelEncoder()

# df['yes_no'] = encoder.fit_transform(df['yes_no'])
# df_test['yes_no'] = encoder.fit_transform(df_test['yes_no'])
# ===========================================================
# encoder = LabelEncoder()
# encoder.fit(df['yes_no'])  # Treniruojame tik su train duomenimis
# df['yes_no'] = encoder.transform(df['yes_no'])
# df_test['yes_no'] = encoder.transform(df_test['yes_no'])
# -------------------------------
encoder = LabelEncoder()
encoder.fit(df['yes_no'])  # Treniruojame tik su train duomenimis
df['yes_no'] = encoder.transform(df['yes_no'])
df_val['yes_no'] = encoder.transform(df_val['yes_no'])

# ===========================================================



# target_test = df['yes_no']

# pixel_arrays = df.pop('pixels')
pixel_arrays = df['pixels']
# pixel_arrays_test = df_test['pixels']
pixel_arrays_val = df_val['pixels']



# pixel_arrays = pixel_arrays.to_numpy()
# print(pixel_arrays.head())
# pixel_arrays = df['pixels'].apply(parse_pixels)

X_pixel = np.stack(pixel_arrays.values)
# X_pixel_test = np.stack(pixel_arrays_test.values)
X_pixel_val = np.stack(pixel_arrays_val.values)


y = df['yes_no'].values
# y_test = df_test['yes_no'].values
y_val = df_val['yes_no'].values



# print(target_test.shape)

# print(df.head())

# print(target_test.head())

# df_pixel = pd.read_csv("pixels_df.csv")

# df_to_numpy = df_pixel.to_numpy()
# ====================================================================
# X_train_val, X_test, y_train_val, y_test = train_test_split(X_pixel, y, random_state=42, shuffle=True)

X_train, X_test, y_train, y_test = train_test_split(X_pixel, y, test_size=0.2, random_state=42, shuffle=True)


# X_train, X_val, y_train, y_val = train_test_split(X_pixel, y, test_size=0.2, random_state=42, shuffle=True)

# X_train_val, X_test, y_train_val, y_test = train_test_split(X_pixel, y, test_size=0.2, random_state=42, shuffle=True)

# sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# for train_index, test_index in sss.split(X_pixel, y):
#     X_train_val, X_test = X_pixel[train_index], X_pixel[test_index]
#     y_train_val, y_test = y[train_index], y[test_index]

# X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)


# x_train = x_train.reshape(-1, 128, 128, 1)
# x_test = x_test.reshape(-1, 128, 128, 1)
# x_val = x_val.reshape(-1, 128, 128, 1)
# y_val = y_val.reshape(-1, 128, 128, 1)
# y_train = y_train.reshape(-1, 128, 128, 1)
# y_test = y_test.reshape(-1, 128, 128, 1)
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"x_train shape: {X_train.shape}")
# print(f"x_test shape: {X_test.shape}")
# print(f"x_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")
# print(f"x_test shape: {X_pixel_test.shape}")
print(f"x_test shape: {X_pixel_val.shape}")



img_size = int(np.sqrt(X_train.shape[1]))

# Reshape for CNN
X_train = X_train.reshape(-1, 128, 128, 3)
X_test = X_test.reshape(-1, 128, 128, 3)
# X_val = X_val.reshape(-1, 128, 128, 3)
# X_pixel_test = X_pixel_test.reshape(-1, 128, 128, 3)
X_pixel_val = X_pixel_val.reshape(-1, 128, 128, 3)


# reshape = x_train.reshape(-1)
# to_flatten = x_train.flatten().tolist()
# print(to_flatten)

# to_list = to_flatten.to_list()


# plt.imshow(x_train[0])
# plt.title(y_train[0])

batch_size = 4
learning_rate = 0.00002
epochs = 30
no_color = 1
color = 3
# ======================================================
# val_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=False)
# # val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size, shuffle=False)

# test_datagen = ImageDataGenerator(rescale=1./255)

# # 2️⃣ Sukuriame test duomenų srautą
# test_generator = test_datagen.flow(X_test, batch_size=batch_size, shuffle=False)
# ======================================================
early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=5,  # Sustabdo, jei per 5 epochas validacijos nuostolis nesumažėja
    restore_best_weights=True
)
# ======================================================

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,       # Sumažina `learning rate` perpus
    patience=3,       # Laukia 3 epochas prieš mažinimą

    min_lr=1e-6       # Nenusileidžia žemiau 0.000001
)
# ======================================================




"""Normalizacija"""
X_train_normalized = X_train / 255.0
X_test_normalized = X_test / 255.0
# X_val_normalized = X_val / 255.0
# X_pixel_test_normalized = X_pixel_test / 255.0
X_pixel_val_normalized = X_pixel_val / 255.0


# =======================================================

# 2. Modelio sukūrimas

# model = Sequential([
#     Conv2D(32, (3,3), activation='relu', kernel_regularizer=l2(0.001), input_shape=(128, 128, 3)),
#     BatchNormalization(),
#     MaxPooling2D((2,2)),
#     Dropout(0.25),

#     Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(0.001)),
#     BatchNormalization(),
#     MaxPooling2D((2,2)),
#     Dropout(0.25),

#     Flatten(),
#     Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')
# ])
# =======================================================
# Modelio sukūrimas
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])



# 3. Modelio kompiliavimas
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

# 4. Modelio treniravimas
# history = model.fit(X_train_normalized, y_train, validation_data=(X_val_normalized, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])

history = model.fit(X_train_normalized, y_train, validation_data=(X_pixel_val_normalized, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping, reduce_lr])

# history = model.fit(
#     train_generator,
#     # validation_data=val_generator,
#     validation_data=(X_val_normalized, y_val),
#     callbacks=[early_stopping, reduce_lr],
#     epochs=epochs,
#     steps_per_epoch=len(X_train) // batch_size,
#     validation_steps=len(X_val) // batch_size
#     )

model.save("melanoma_classifier.h5")


# 5. Modelio vertinimas
# val_preds = model.predict(X_test_normalized)
val_preds = model.predict(X_test_normalized)

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