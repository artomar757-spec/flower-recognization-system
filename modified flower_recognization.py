# =========================
# IMPORT LIBRARIES
# =========================
import numpy as np
import tensorflow as tf
import wikipedia
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# =========================
# IMAGE PREPROCESSING
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

training_set = train_datagen.flow_from_directory(
    'training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    'test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# =========================
# BUILD CNN MODEL
# =========================
cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Dropout(0.5))

cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=5, activation='softmax'))

cnn.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# TRAIN MODEL
# =========================
cnn.fit(x=training_set, validation_data=test_set, epochs=30)
cnn.save("flower_model.h5")
print("Model saved successfully!")
import json

# Save class labels
with open("class_labels.json", "w") as f:
    json.dump(training_set.class_indices, f)

print("Class labels saved successfully!")
# =========================
# FUNCTION TO FETCH WIKIPEDIA DATA
# =========================
def get_flower_info_from_wikipedia(flower_name):
    try:
        wikipedia.set_lang("en")
        
        # Add "flower" keyword for better search
        search_query = flower_name + " flower"
        
        summary = wikipedia.summary(search_query, sentences=3)
        page = wikipedia.page(search_query)
        
        return summary, page.url
    
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            summary = wikipedia.summary(e.options[0], sentences=3)
            page = wikipedia.page(e.options[0])
            return summary, page.url
        except:
            return "No clear information found.", None
    
    except Exception:
        return "No information found on Wikipedia.", None


# =========================
# PREDICTION SECTION
# =========================
test_image = image.load_img('Prediction/f.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = test_image / 255.0
test_image = np.expand_dims(test_image, axis=0)

result = cnn.predict(test_image)

# Get class labels
class_indices = training_set.class_indices
class_names = list(class_indices.keys())

# Get predicted flower
predicted_index = np.argmax(result[0])
predicted_flower = class_names[predicted_index]

print("\n Predicted Flower:", predicted_flower.capitalize())

# =========================
# FETCH LIVE WIKIPEDIA DATA
# =========================
summary, url = get_flower_info_from_wikipedia(predicted_flower)

print("\n Wikipedia Information:\n")
print(summary)

if url:
    print("\n Read more:", url)