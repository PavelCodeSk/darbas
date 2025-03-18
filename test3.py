import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the data
df = pd.read_csv("df_yes_no.csv")

# Encode the target variable
encoder = LabelEncoder()
df['yes_no'] = encoder.fit_transform(df['yes_no'])

def parse_pixels(pixel_str):
    # Remove brackets and split by commas
    pixels = pixel_str.strip('[]').split(',')
    # Convert to integers
    return np.array([int(p.strip()) for p in pixels])

# Apply the function to create a list of numpy arrays
pixel_arrays = df['pixels'].apply(parse_pixels)

# Convert to a 2D numpy array
X = np.stack(pixel_arrays.values)
y = df['yes_no'].values

# First split: Training+Validation vs Test (80% vs 20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Second split: Training vs Validation (75% vs 25% of the training+validation set)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42
)

# Print the shapes to verify
print(f"y_train shape: {y_train.shape}")
print(f"X_train shape: {X_train.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")


img_size = int(np.sqrt(X_train.shape[1]))

# Normalize pixel values to range [0, 1]
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0



# Reshape for CNN

X_train = X_train.reshape(-1, img_size, img_size, 1)
X_test = X_test.reshape(-1, img_size, img_size, 1)
X_val = X_val.reshape(-1, img_size, img_size, 1)


# Print the shapes to verify
# print(f"y_train shape: {y_train.shape}")
# print(f"X_train shape: {X_train.shape}")
# print(f"y_test shape: {y_test.shape}")
# print(f"X_test shape: {X_test.shape}")
# print(f"X_val shape: {X_val.shape}")
# print(f"y_val shape: {y_val.shape}")

 