import pandas as pd
import cv2
import numpy as np
import glob as gb
import os
import skimage.filters as filters
import albumentations as A
import albumentations.augmentations.transforms as T
import ast
from utilities.filtr_for_images import lache_filtr, clahe_fitr

def concat_df():
    df_yes = pd.read_csv("img_yes.csv")
    df_no = pd.read_csv("img_no.csv")

    df_yes_no = pd.concat([df_yes,df_no], ignore_index=True, axis=0)
    df_mixed = df_yes_no.sample(frac=1, random_state=42).reset_index(drop=True)

    df_mixed.to_csv("df_yes_no.csv", index=False)
    return df_mixed

"""Sukuriam kintajami kuriame nurodysim kelia i nuotraukas"""
images_folder_yes_melanoma = "dataset/yesmelanoma/*.*"
images_folder_no_melanoma = "dataset/nomelanoma/*.*"
images_folder_no_melanoma_test = "dataset/test_images/nomelanoma/*.*"

"""Sukuriam kintamaji kuriame nurodysim nuotrauku ikelimo kelia (papke vieta)
        preprocessintas nuotraukas kelsim i nauja papke"""
output_folder_yes_melanoma = "dataset/output_folder_yes_melanoma"
output_folder_no_melanoma = "dataset/output_folder_no_melanoma"
output_folder_no_melanoma_test = "dataset/test_images/output_folder_no_melanoma_test"

"""Sukuriam salyga - patikrina ar yra tokia papke(kelias) jeigu nera tada sukuria"""
if not os.path.exists(output_folder_yes_melanoma):
    os.makedirs(output_folder_yes_melanoma)

if not os.path.exists(output_folder_no_melanoma):
    os.makedirs(output_folder_no_melanoma)

if not os.path.exists(output_folder_no_melanoma_test):
    os.makedirs(output_folder_no_melanoma_test)

"""grąžina sąrašą failų, kurie atitinka pateiktą šabloną.
Jei images_folder_no_melanoma yra:
tada gb.glob(images_folder_no_melanoma) suras visus .jpg
failus aplanke dataset/no_melanoma/ ir grąžins jų pilnus kelius sąraše."""
images_paths_yes = gb.glob(images_folder_yes_melanoma)
images_paths_no = gb.glob(images_folder_no_melanoma)
images_paths_no_test = gb.glob(images_folder_no_melanoma_test)



def preprocessing_images(image_path, row_name, color=True):
    data = []

    for path in image_path:
        image = cv2.imread(path)
        if image is None:
            print(f"Error: Could not open {path}")
            continue
        if color:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_resize = cv2.resize(image_gray, (128, 128))
        pixels = image_resize.flatten()

        data.append([row_name, pixels.tolist()])

    return data

def create_csv_from_images(image_path_yes, image_path_no,row_name_yes, row_name_no, label_name, color=True):
    data1 = preprocessing_images(image_path_yes, row_name_yes, color)
    data2 = preprocessing_images(image_path_no, row_name_no, color)
    columns_names = [label_name, "pixels"]
    df_yes = pd.DataFrame(data1, columns=columns_names)
    df_no = pd.DataFrame(data2, columns=columns_names)

    df_concat = pd.concat([df_yes, df_no], ignore_index=True, axis=0)
    # df_mixed = df_concat.sample(frac=1, random_state=42).reset_index(drop=True)
    df_concat.to_csv(f"df_{label_name}.csv", index=False)

# create_csv_from_images(images_paths_yes, images_paths_no, row_name_yes="yes", row_name_no="no", label_name="yes_no", color=True)
# print("failas sukurtas")

# ==========================================================
# df = pd.read_csv("df_yes_no.csv")
# data = []
# def convert_pixels(x):
#     print(f"Konvertuojamas irasas {x}")
#     change = np.array(ast.literal_eval(x))
#     data.append(change)
#     return data


# df["pixels"] = df["pixels"].apply(convert_pixels)

# df_pixel = pd.DataFrame(data=data)

# df_pixel.to_csv("pixels_df.csv", index=False)
# ==========================================================



def convert_string_to_list(csv_file:str, column: str, exmport=True):
    data = []
    df = pd.read_csv(csv_file)

    def convert_pixels(x):
        print(f"Konvertuojamas irasas {x}")
        change = np.array(ast.literal_eval(x))
        # change = ast.literal_eval(x)

        data.append(change)
        return data
    
    df[column] = df[column].apply(convert_pixels)

    df_pixel = pd.DataFrame(data=data)
    if exmport:
        df_pixel.to_csv("pixels_df.csv", index=False)
    return df_pixel

# df = convert_string_to_list("df_yes_no.csv", column="pixels", exmport=True)

# print(df.head())


# def augmentation_img():

def parse_pixels(pixel_str):
    # Remove brackets and split by commas
    pixels = pixel_str.strip('[]').split(',')
    # Convert to integers
    return np.array([int(p.strip()) for p in pixels])

def preprocessing_imgs(image_path, row_name=None, no_color=None, add_filtr=None, lache=None):
    """preprocessing_imgs(image_path, row_name, no_color=None, add_filtr=None, lache"""
    data = []
    number = 0

    for path in image_path:
        image = cv2.imread(path)
        if image is None:
            print(f"Error: Could not open {path}")
            continue
        if no_color is False and add_filtr is False and lache:
            image_resize = cv2.resize(image, (128, 128))
            karnel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            filtered_img2 = cv2.morphologyEx(image_resize, cv2.MORPH_BLACKHAT, karnel2)
            image_clahe = clahe_fitr(image_resize)
            laplacian_colored = lache_filtr(image_clahe)
            pixels = laplacian_colored.flatten()
            data.append([row_name, pixels.tolist()])
            number += 1
            print(number)
            

        if no_color and add_filtr:
            image_gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_resize1 = cv2.resize(image_gray1, (128, 128))
            image_equalized1 = cv2.equalizeHist(image_resize1)
            karnel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            filtered_img1 = cv2.morphologyEx(image_equalized1, cv2.MORPH_BLACKHAT, karnel1)
            pixels = filtered_img1.flatten()
            data.append([row_name, pixels.tolist()])
            number += 1
            print(number)

        if no_color and add_filtr is False:
            image_gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_resize2 = cv2.resize(image_gray2, (128, 128))
            pixels = image_resize2.flatten()
            data.append([row_name, pixels.tolist()])
            number += 1
            print(number)

        # if no_color and add_filtr:
        #     pixels = filtered_img.flatten()
        #     data.append([row_name, pixels.tolist()])
        # if no_color and add_filtr is False:
        #     pixels = image_resize.flatten()
        #     data.append([row_name, pixels.tolist()])
                

        if no_color is False and add_filtr:
            image_resize3 = cv2.resize(image, (128, 128))
            # image_equalized2 = cv2.equalizeHist(image_resize3)
            karnel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            filtered_img2 = cv2.morphologyEx(image_resize3, cv2.MORPH_BLACKHAT, karnel2)
            pixels = filtered_img2.flatten()
            data.append([row_name, pixels.tolist()])
            number += 1
            print(number)
        if no_color is False and add_filtr is False and lache is False or None:
            image_resize4 = cv2.resize(image, (128, 128))
            pixels = image_resize4.flatten()
            data.append([row_name, pixels.tolist()])
            number += 1
            print(number)

        # elif no_color is False and add_filtr:
        #     pixels = filtered_img.flatten()
        #     data.append([row_name, pixels.tolist()])
        # elif no_color is False and add_filtr is False:
        #     pixels = image_gray.flatten()
        #     data.append([row_name, pixels.tolist()])
        # pixels = image_resize.flatten()

        # data.append([row_name, pixels.tolist()])
    return data

def create_df_image_yes_no(image_folder_yes, image_folder_no, row_name_yes, row_name_no, label_name, no_color=None, add_filtr=None, lache=None):
    """create_df_image_yes_no(image_folder_yes, image_folder_no, row_name_yes, row_name_no, label_name, no_color=True, add_filtr )"""
    images_paths_yes = gb.glob(image_folder_yes)
    images_paths_no = gb.glob(image_folder_no)
    number = 0

    data_yes = preprocessing_imgs(images_paths_yes, row_name=row_name_yes, no_color=no_color, add_filtr=add_filtr, lache=lache)
    data_no = preprocessing_imgs(images_paths_no, row_name=row_name_no, no_color=no_color, add_filtr=add_filtr, lache=lache)

    columns_names = [label_name, "pixels"]
    df_yes = pd.DataFrame(data_yes, columns=columns_names)
    df_no = pd.DataFrame(data_no, columns=columns_names)
    df_concat = pd.concat([df_yes, df_no], ignore_index=True, axis=0)
    df_mixed = df_concat.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_mixed



from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range=20,    # Pasuka iki 20 laipsnių
    width_shift_range=0.2, # Perkelia horizontaliai iki 20% 
    height_shift_range=0.2, # Perkelia vertikaliai iki 20%
    shear_range=0.2,      # Šlyties transformacija
    zoom_range=0.2,       # Priartinimas iki 20%
    horizontal_flip=True, # Atsitiktinis apvertimas horizontaliai
    fill_mode='nearest'   # Užpildymas kraštuose
)

# train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)

def get_image_for_test_model(img_folder, no_color=None, add_filtr=None, lache=None):
    # Load the image using OpenCV

    images_path = gb.glob(img_folder)


    # image = cv2.imread(img_folder)
    preprocess_img = preprocessing_imgs(image_path=images_path, no_color=no_color, add_filtr=add_filtr, lache=lache)

    data = list(preprocess_img)
    image = None
    for a in data:
        del a[0]
        image = a[0]
        image = np.array(image).reshape(-1, 128, 128, 3)
        image = image / 255.0
    return image


img_test_folder = "dataset/test_model_image/*.*"

result = get_image_for_test_model(img_test_folder)
print(result)


