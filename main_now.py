from tensorflow.keras.models import load_model
from utilities.utilities import get_image_for_test_model
import numpy as np

img_test_folder_no = "dataset/test_model_image/nomelanoma/*.*"
img_test_folder_yes = "dataset/test_model_image/melanoma/*.*"



model = load_model("2_melanoma_classifier.h5")

# Test the loaded model with a sample image

# prediction = model.predict()

result = get_image_for_test_model(img_test_folder_yes, no_color=False, add_filtr=False, lache=True)
print(result)

# img = np.array(result)

prediction = model.predict(result)

print("Modelio išvestis:", prediction)

if prediction[0][0] > 0.03:
    print("Melanoma: TAIP")
else:
    print("Melanoma: NE")

