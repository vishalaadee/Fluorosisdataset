import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt

def load_and_resize_images(folder, target_size=(224, 224)):
    images = []
    for filename in os.listdir(folder):
        img = image.load_img(os.path.join(folder, filename), target_size=target_size, grayscale=True)
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        images.append((filename, img_array))

    return images

normal_images = load_and_resize_images("/Users/er.vishalmishra/Downloads/Fluorosisdataset/NormalTeeth")
patient_images = load_and_resize_images("/Users/er.vishalmishra/Downloads/Fluorosisdataset/sample data")

min_samples = min(len(normal_images), len(patient_images))

print(f"Number of normal samples: {len(normal_images)}")
print(f"Number of patient samples: {len(patient_images)}")
print(f"Minimum number of samples: {min_samples}")

normal_labels = np.zeros(min_samples)
patient_labels = np.ones(min_samples)

X = np.array([img[1] for img in normal_images[:min_samples]] + [img[1] for img in patient_images[:min_samples]])
y = np.concatenate((normal_labels, patient_labels))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 1))
model = Sequential([
    base_model,
    Flatten(),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

user_input = input("Enter the path of the image you want to predict: ")
user_image = image.load_img(user_input, target_size=(224, 224), grayscale=True)
user_image_array = image.img_to_array(user_image)
user_image_array = preprocess_input(user_image_array)

user_image_array = np.expand_dims(user_image_array, axis=0)

user_prediction = model.predict(user_image_array)

plt.imshow(user_image, cmap='gray')
plt.title(f"Predicted: {'Patient' if user_prediction >= 0.5 else 'Normal'}")
plt.show()
