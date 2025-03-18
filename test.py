import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np

# number = 1

# for i in range(10):
#     number += 1
#     print(number)

# df = pd.read_csv("df_yes_no.csv")
# print(df.head())

# list_pixel = df["pixels"]
# print(list_pixel)
# print(list_pixel)
# df_pixel = pd.DataFrame(list_pixel)
# print(df_pixel.head())

# print(df.head())
# print(df.dtypes)
# print(df.columns)

# df_test = df['pixels'].apply(lambda x: np.array(ast.literal_eval(x)))
# print("+"*100)
# print(df_test)
# def pixels_list(x):
#     data = []
#     data.append(x)
#     # df = pd.DataFrame(x)
#     # df.to_csv("df_pixels.csv", index=False)
#     return data

# def create_df_pixels(x):
#     df = pd.DataFrame(x)
#     df.to_csv("df_pixels.csv", index=False)

df = pd.read_csv("df_yes_no.csv")
data = []
def convert_pixels(x):
    print(f"Konvertuojamas irasas {x}")
    # change = np.array(ast.literal_eval(x))
    change = ast.literal_eval(x)

    data.append(change)
    return data


df["pixels"] = df["pixels"].apply(convert_pixels)

df_pixel = pd.DataFrame(data=data)

df_pixel.to_csv("pixels_df.csv", index=False)

# print(df["pixels"])




