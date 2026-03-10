import numpy as np
import wikipedia
from tensorflow.keras.models import load_model
from keras.preprocessing import image

# Load saved model
cnn = load_model("flower_model.h5")
print("Model loaded successfully!")

# Load image
test_image = image.load_img('Prediction/f.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = test_image / 255.0
test_image = np.expand_dims(test_image, axis=0)

# Predict
result = cnn.predict(test_image)

class_labels = ['daisy','dandelion','rose','sunflower','tulip']

predicted_index = np.argmax(result)
confidence = np.max(result) * 100
predicted_flower = class_labels[predicted_index]

print("\nPredicted Flower:", predicted_flower.capitalize())
print("Confidence:", round(confidence,2), "%")

# Wikipedia Info
def get_flower_info(name):
    try:
        summary = wikipedia.summary(name + " flower", sentences=3)
        page = wikipedia.page(name + " flower")
        return summary, page.url
    except:
        return "No information found.", None

summary, url = get_flower_info(predicted_flower)

print("\nWikipedia Information:\n")
print(summary)

if url:
    print("\nRead more:", url)