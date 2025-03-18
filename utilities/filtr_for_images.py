import cv2
import numpy as np

def clahe_fitr(image):
    # image = cv2.imread("paveikslelis.jpg")

    # Konvertuojame į LAB spalvų erdvę
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Išskaidome į L, A, B kanalus
    l, a, b = cv2.split(lab)

    # Pritaikome CLAHE tik šviesumo kanalui (L)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)

    # Sujungiame kanalus atgal
    lab_clahe = cv2.merge((l_clahe, a, b))

    # Konvertuojame atgal į BGR formatą
    image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return image_clahe

def lache_filtr(image):
    # image = cv2.imread("paveikslelis.jpg")

    # Išskaidome į BGR kanalus
    b, g, r = cv2.split(image)

    # Pritaikome Laplacian filtrą kiekvienam kanalui
    laplacian_b = cv2.Laplacian(b, cv2.CV_64F)
    laplacian_g = cv2.Laplacian(g, cv2.CV_64F)
    laplacian_r = cv2.Laplacian(r, cv2.CV_64F)

    # Konvertuojame atgal į uint8
    laplacian_b = np.uint8(np.absolute(laplacian_b))
    laplacian_g = np.uint8(np.absolute(laplacian_g))
    laplacian_r = np.uint8(np.absolute(laplacian_r))

    # Sujungiame atgal į vieną vaizdą
    laplacian_colored = cv2.merge([laplacian_b, laplacian_g, laplacian_r])
    return laplacian_colored



# if no_color is False and add_filtr is False and lache:
#             image_resize = cv2.resize(image, (128, 128))

#             b, g, r = cv2.split(image_resize)

#             # Pritaikome Laplacian filtrą kiekvienam kanalui
#             laplacian_b = cv2.Laplacian(b, cv2.CV_64F)
#             laplacian_g = cv2.Laplacian(g, cv2.CV_64F)
#             laplacian_r = cv2.Laplacian(r, cv2.CV_64F)

#             # Konvertuojame atgal į uint8
#             laplacian_b = np.uint8(np.absolute(laplacian_b))
#             laplacian_g = np.uint8(np.absolute(laplacian_g))
#             laplacian_r = np.uint8(np.absolute(laplacian_r))

#             laplacian_colored = cv2.merge([laplacian_b, laplacian_g, laplacian_r])
#             pixels = laplacian_colored.flatten()
#             data.append([row_name, pixels.tolist()])
#             number += 1
#             # print(number)