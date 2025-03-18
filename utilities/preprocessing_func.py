import pandas
import cv2
import glob as gb
import os
import skimage.filters as filters
import albumentations as A
import albumentations.augmentations.transforms as T

"""Sukuriam kintajami kuriame nurodysim kelia i nuotraukas"""
images_folder_yes_melanoma = "dataset/nomelanoma/*.*"
images_folder_no_melanoma = "dataset/yesmelanoma/*.*"
images_folder_no_melanoma_test = "dataset/test_images/nomelanoma/*.*"


# C:\Users\pavel\Desktop\python_darbai\2_baigiamasis_darbas\darbas\dataset\nomelanoma



"""Sukuriam kintamaji kuriame nurodysim nuotrauku ikelimo kelia (papke vieta)
        preprocessintas nuotraukas kelsim i nauja papke"""
output_folder_yes_melanoma = "dataset/output_folder_yes_melanoma"
output_folder_no_melanoma = "dataset/output_folder_no_melanoma"
output_folder_no_melanoma_test = "dataset/test_images/output_folder_no_melanoma_test"


# ================================================================

"""Sukuriam salyga - patikrina ar yra tokia papke(kelias) jeigu nera tada sukuria"""
if not os.path.exists(output_folder_yes_melanoma):
    os.makedirs(output_folder_yes_melanoma)

if not os.path.exists(output_folder_no_melanoma):
    os.makedirs(output_folder_no_melanoma)

if not os.path.exists(output_folder_no_melanoma_test):
    os.makedirs(output_folder_no_melanoma_test)

# ================================================================
"""grąžina sąrašą failų, kurie atitinka pateiktą šabloną.
    Jei images_folder_no_melanoma yra:
    tada gb.glob(images_folder_no_melanoma) suras visus .jpg
    failus aplanke dataset/no_melanoma/ ir grąžins jų pilnus kelius sąraše."""
images_paths_yes = gb.glob(images_folder_yes_melanoma)
images_paths_no = gb.glob(images_folder_no_melanoma)
images_paths_no_test = gb.glob(images_folder_no_melanoma_test)

# ================================================================

# Augmentacijos transformacijos
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),  # 50% tikimybė apversti horizontaliai
    A.VerticalFlip(p=0.2),    # 20% tikimybė apversti vertikaliai
    T.RandomBrightnessContrast(p=0.2),  # Kontrasto ir ryškumo pakeitimai
    A.Rotate(limit=40, p=0.3),  # Pasukimas nuo -40 iki 40 laipsnių
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.3)  # Atsitiktinis mastelio ir padėties keitimas
])

def proccesed_images_and_upload_new_cataloge(image_path, output_folder, augment=True, num_augmented=3):
    """Sia funkcija naudojam tam ,
      kad paimti nuotraukas preprocessinti ir ikelti i nauja vieta

      example:
      proccesed_images_and_upload_new_cataloge(image_path, output_folder)"""

    for path in image_path:
        image = cv2.imread(path) # uzkeliam nuotrauka is saraso
        if image is None:
            print(f"Error: Could not read image from {path}") # patikrinam ar yra tokia nuotrauka , jeigu nera praleidziame
            continue
        image_resize = cv2.resize(image, (128, 128)) # keiciam nuotraukos dydi
        image_gray = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)
        image_equalized = cv2.equalizeHist(image_gray)
        karnel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        filtered_img = cv2.morphologyEx(image_gray, cv2.MORPH_BLACKHAT, image_equalized)
        # image_blur = cv2.GaussianBlur(image_equalized, (9,9), 0)
        # image_sharped = filters.unsharp_mask(image_blur, 3, 1.5) * 255
        # grad_x = cv2.Sobel(filtered_img, cv2.CV_64F, 1, 0, ksize=3)  # X kryptimi
        # grad_y = cv2.Sobel(filtered_img, cv2.CV_64F, 0, 1, ksize=3)  # Y kryptimi
        # grad_x = cv2.convertScaleAbs(grad_x)
        # grad_y = cv2.convertScaleAbs(grad_y)
        # sobel_combined = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)

        filename = os.path.basename(path)

        save_path = os.path.join(output_folder, filename)

        cv2.imwrite(save_path, filtered_img) # ikeliam pakeistas nuotraukas i nauja vieta
        print(f"Processed: {filename}")
        # ===========================================
        """Augmentacija"""
        if augment:
            for i in range(1, num_augmented + 1):
                aug_result = augmentations(image=filtered_img)
                aug_img = aug_result["image"]

                # Pridedame prie failo pavadinimo `_aug1`, `_aug2`, ...
                aug_filename = f"{os.path.splitext(filename)[0]}_aug{i}.jpg"
                aug_save_path = os.path.join(output_folder, aug_filename)
                
                cv2.imwrite(aug_save_path, aug_img)
                print(f"Saved augmented: {aug_filename}")


    print(f"All images processed and saved in {output_folder}")


proccesed_images_and_upload_new_cataloge(images_paths_no_test, output_folder_no_melanoma_test, augment=False)

print("tdddd")