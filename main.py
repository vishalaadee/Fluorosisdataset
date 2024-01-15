import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

def load_and_resize_images(folder, target_size=(100, 100)):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, target_size)

            img_flat = img_resized.flatten()

            print(f"Loaded and Resized Image: {filename}, Original Shape: {img.shape}, Resized Shape: {img_resized.shape}, Flattened Shape: {img_flat.shape}")

            images.append((filename, img_flat))

    return images

normal_images = load_and_resize_images("/Users/er.vishalmishra/Downloads/Fluorosisdataset/NormalTeethJan24")
patient_images = load_and_resize_images("/Users/er.vishalmishra/Downloads/Fluorosisdataset/DentalFluorosisJan24")

min_samples = min(len(normal_images), len(patient_images))

print(f"Number of normal samples: {len(normal_images)}")
print(f"Number of patient samples: {len(patient_images)}")
print(f"Minimum number of samples: {min_samples}")

normal_labels = np.zeros(min_samples)
patient_labels = np.ones(min_samples)

X = np.vstack((np.array([img[1] for img in normal_images[:min_samples]]),
               np.array([img[1] for img in patient_images[:min_samples]])))

print(f"Shape of X after stacking: {X.shape}")

y = np.concatenate((normal_labels, patient_labels))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

classifier = SVC(kernel='linear')
classifier.fit(X_scaled, y)

joblib.dump((classifier, scaler), 'svm_model_with_scaling.pkl')

loaded_model, loaded_scaler = joblib.load('svm_model_with_scaling.pkl')

user_input = input("Enter the path of the image you want to predict: ")
user_image = cv2.imread(user_input, cv2.IMREAD_GRAYSCALE)

if user_image is None:
    print("Error: Unable to load the image.")
else:
    user_image_resized = cv2.resize(user_image, (100, 100))

    user_features = user_image_resized.flatten()
    user_features_scaled = loaded_scaler.transform(user_features.reshape(1, -1))

    user_prediction = loaded_model.predict(user_features_scaled)

    plt.imshow(user_image_resized, cmap='gray')
    plt.title(f"Predicted: {'Patient' if user_prediction == 1 else 'Normal'}")
    plt.show()
