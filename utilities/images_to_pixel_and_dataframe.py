import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob as gb
import os

images_folder_yes = "dataset/output_folder_yes_melanoma/*.*"
images_folder_no = "dataset/output_folder_no_melanoma/*.*"
images_folder_no_test = "dataset/test_images/nomelanoma/*.*"

output_csv_yes = "img_yes.csv"
output_csv_no = "img_no.csv"
output_csv_no_test = "img_no_test.csv"

image_path_yes = gb.glob(images_folder_yes)
image_path_no = gb.glob(images_folder_no)
image_path_no_test = gb.glob(images_folder_no_test)


def images_to_csv(image_folder, output_csv, label_name: str, row_name: str):
    """Si funkcija ima is folderio nuotrauka vercia ja i pixelius surusioja
      kaip sarasa ir deda i sarasa kuri veliau verciam i dataframe ir exportuojam 
        i csv faila"""

    image_paths = gb.glob(image_folder)
    data = []
    number = 0
    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Could not read image: {path}")
            continue
        pixels = image.flatten()

        row = [row_name] + pixels.tolist()
        data.append(row)

        column_names = [label_name] + [f"pixel{i}" for i in range(len(data[0]) -1)]
        df = pd.DataFrame(data=data, columns=column_names)

        df.to_csv(output_csv, index=False)
        number += 1
        print(f"CSV failas issaugotas kaip: {output_csv}, number: {number}")
    print("Failas pilnai sukurtas")

images_to_csv(images_folder_no_test, output_csv_no_test, label_name="yes_no", row_name="no")
